"""whisper.cpp-based Whisper benchmark — TTFT, TPOT, throughput on
LibriSpeech / TED-LIUM.

Uses pywhispercpp (the Python binding for ggerganov/whisper.cpp) and GGML/GGUF
model files for CPU/Metal/CUDA inference. Mirrors the single/batch/offline
modes from speech_benchmark.py so results are directly comparable.

Important: whisper.cpp does *not* fuse multiple audios into one inference
the way HF-transformers / vLLM do. In production, whisper.cpp is deployed as
N concurrent worker contexts, each transcribing one audio at a time. So
"batch" here is a thread pool of `--batch-size` workers, each with its own
whisper_context. Memory scales linearly with batch_size — at large model
sizes you'll want quantized GGML variants (q5_0, q8_0) before scaling out.

Per-sample metrics (preferred path — when the _pywhispercpp C binding exposes
whisper_get_timings, which is the common case):
  - e2e_ms:    wall clock for one model.transcribe() call
  - ttft_ms:   encode_ms + avg sample/decode step (i.e. time-to-first-token
               approximated as encode + one decoder step)
  - tpot_ms:   (decode_ms + sample_ms) / n_decode (whisper.cpp's actual
               per-token decode cost, not a wall-clock estimate)
  - n_output_tokens: n_decode from whisper.cpp's counter

Fallback path (older pywhispercpp without get_timings binding):
  - e2e_ms only; ttft_ms / tpot_ms set to NaN, n_output_tokens estimated from
    the transcribed text via WhisperTokenizer.

Output: one CSV per (model, precision, dataset, mode) into output/whispercpp/,
named {model}_{precision}_{dataset}_{mode_tag}_{YYYYMMDD_HHMM}.csv.
"""
import argparse
import csv
import datetime as _dt
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np

_HERE = Path(__file__).resolve().parent
GGML_MODELS_DIR = _HERE / "models" / "whisper.cpp"
OUTPUT_DIR = _HERE / "output" / "whispercpp"
DATA_DIR = _HERE / "data"
HF_CACHE_DIR = DATA_DIR / "hf_datasets"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
GGML_MODELS_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HOME", str(DATA_DIR / "hf_home"))

EAST = ZoneInfo("America/New_York")

# GGML files hosted at https://huggingface.co/ggerganov/whisper.cpp
GGML_REGISTRY = {
    "whisper-tiny":             "ggml-tiny.bin",
    "whisper-base":             "ggml-base.bin",
    "whisper-small":            "ggml-small.bin",
    "whisper-medium":           "ggml-medium.bin",
    "whisper-large-v3":         "ggml-large-v3.bin",
    "whisper-large-v3-q5_0":    "ggml-large-v3-q5_0.bin",
    "whisper-large-v3-q8_0":    "ggml-large-v3-q8_0.bin",
    "whisper-large-v3-turbo":   "ggml-large-v3-turbo.bin",
    "whisper-large-v3-turbo-q5_0": "ggml-large-v3-turbo-q5_0.bin",
}
GGML_HF_REPO = "ggerganov/whisper.cpp"


@dataclass
class SampleRow:
    sample_id: str
    audio_seconds: float
    n_output_tokens: int
    ttft_ms: float
    tpot_ms: float
    e2e_ms: float


@dataclass
class Summary:
    model: str
    precision: str
    dataset: str
    mode: str
    batch_size: int
    n_threads: int
    n_samples: int
    total_audio_s: float
    total_wall_s: float
    throughput_xrt: float
    throughput_tokens_per_s: float
    avg_ttft_ms: float
    p50_ttft_ms: float
    p95_ttft_ms: float
    avg_tpot_ms: float
    p50_tpot_ms: float
    p95_tpot_ms: float


# --- dataset / audio helpers ---------------------------------------------------

def _iter_dataset(name: str, max_samples: Optional[int], streaming: bool):
    from speech_benchmark import _iter_librispeech, _iter_tedlium
    if name == "librispeech":
        return _iter_librispeech(max_samples=max_samples, streaming=streaming)
    if name == "tedlium":
        return _iter_tedlium(max_samples=max_samples, streaming=streaming)
    raise ValueError(f"unknown dataset: {name}")


def _resample(audio: np.ndarray, sr: int, target_sr: int = 16000) -> np.ndarray:
    if sr == target_sr:
        return audio.astype(np.float32, copy=False)
    import librosa
    return librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)


# --- model / precision helpers -------------------------------------------------

def _infer_precision(filename: str) -> str:
    n = filename.lower()
    for tag in ("q4_0", "q4_1", "q5_0", "q5_1", "q6_k", "q8_0", "fp16", "fp32"):
        if tag in n:
            return tag
    return "fp16"  # unquantized ggml whisper ships in fp16


def _resolve_ggml_model(model_arg: str) -> Path:
    """Resolve --model to a local .bin path. Downloads from HF if absent."""
    p = Path(model_arg)
    if p.is_absolute() and p.exists():
        return p
    if model_arg in GGML_REGISTRY:
        filename = GGML_REGISTRY[model_arg]
    elif model_arg.endswith(".bin"):
        filename = model_arg
    else:
        filename = f"ggml-{model_arg}.bin"
    target = GGML_MODELS_DIR / filename
    if target.exists():
        return target
    print(f"[download] {filename} from {GGML_HF_REPO} → {GGML_MODELS_DIR}")
    from huggingface_hub import hf_hub_download
    downloaded = hf_hub_download(
        repo_id=GGML_HF_REPO, filename=filename,
        local_dir=str(GGML_MODELS_DIR),
        local_dir_use_symlinks=False,
    )
    return Path(downloaded)


# --- tokenizer (lazy, shared) --------------------------------------------------

_TOK_LOCK = threading.Lock()
_TOK = None


def _count_tokens(text: str) -> int:
    """Token count via WhisperTokenizer for apples-to-apples with the HF/vLLM
    benchmarks. Falls back to word count if transformers isn't usable."""
    global _TOK
    if not text:
        return 0
    if _TOK is None:
        with _TOK_LOCK:
            if _TOK is None:
                try:
                    from transformers import WhisperTokenizer
                    _TOK = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")
                except Exception as e:
                    print(f"[warn] couldn't load WhisperTokenizer ({e}); "
                          f"falling back to word-split for n_tokens")
                    _TOK = "fallback"
    if _TOK == "fallback":
        return max(1, len(text.split()))
    return len(_TOK.encode(text))


# --- internal timing counters from whisper.cpp ---------------------------------

def _read_ctx_timings(model) -> Optional[dict]:
    """Pull whisper.cpp's internal cumulative timing counters via the
    _pywhispercpp C binding. Returns a dict with encode_ms / decode_ms /
    sample_ms / n_decode / n_sample, or None if this build doesn't expose it.

    Counters accumulate on the context across transcribe() calls, so caller
    diffs before/after to get per-call timings.
    """
    try:
        import _pywhispercpp as pwcpp
    except ImportError:
        return None
    getter = getattr(pwcpp, "whisper_get_timings", None)
    if getter is None:
        return None
    ctx = None
    for attr in ("_ctx", "context", "ctx"):
        ctx = getattr(model, attr, None)
        if ctx is not None:
            break
    if ctx is None:
        return None
    try:
        t = getter(ctx)
    except Exception:
        return None
    # Field names changed between whisper.cpp versions (t_*_ms vs *_ms).
    def _f(name):
        for k in (name, f"t_{name}", f"{name}_ms", f"t_{name}_ms"):
            v = getattr(t, k, None)
            if v is not None:
                return float(v)
        return 0.0
    def _i(name):
        v = getattr(t, name, None)
        return int(v) if v is not None else 0
    return {
        "encode_ms": _f("encode_ms"),
        "decode_ms": _f("decode_ms"),
        "sample_ms": _f("sample_ms"),
        "n_decode": _i("n_decode"),
        "n_sample": _i("n_sample"),
    }


def _diff_timings(before: Optional[dict], after: Optional[dict]) -> Optional[dict]:
    if before is None or after is None:
        return None
    return {k: after.get(k, 0) - before.get(k, 0) for k in after}


# --- whisper.cpp model pool (one Model per worker thread) ----------------------

_TLS = threading.local()


def _get_thread_model(model_path: Path, n_threads: int, use_gpu: bool):
    """Each worker thread gets its own whisper_context. They're not safe to
    share concurrently — whisper.cpp keeps decoder state on the context."""
    if not hasattr(_TLS, "model"):
        from pywhispercpp.model import Model
        kwargs = dict(
            model=str(model_path),
            n_threads=n_threads,
            print_progress=False,
            print_realtime=False,
            print_timestamps=False,
        )
        # Only try use_gpu when explicitly requested. Some pywhispercpp builds
        # accept the kwarg on Model() but then try to set it on
        # whisper_full_params (which lacks the field) and raise AttributeError.
        if use_gpu:
            try:
                _TLS.model = Model(**kwargs, use_gpu=True)
            except (TypeError, AttributeError) as e:
                print(f"[warn] --use-gpu not supported by this pywhispercpp "
                      f"build ({type(e).__name__}: {e}); using CPU")
                _TLS.model = Model(**kwargs)
        else:
            _TLS.model = Model(**kwargs)
    return _TLS.model


def _transcribe_one(model_path: Path, n_threads: int, use_gpu: bool,
                    sample_id: str, audio: np.ndarray, sr: int,
                    language: str) -> SampleRow:
    audio16 = _resample(audio, sr, 16000)
    audio_s = len(audio16) / 16000.0
    model = _get_thread_model(model_path, n_threads, use_gpu)

    before = _read_ctx_timings(model)

    t0 = time.perf_counter()
    segments = model.transcribe(audio16, language=language)
    t1 = time.perf_counter()
    e2e_ms = (t1 - t0) * 1000.0

    after = _read_ctx_timings(model)
    delta = _diff_timings(before, after)

    if delta and delta.get("n_decode", 0) > 0:
        # whisper.cpp's internal counters. encode_ms is per-encoder-pass total;
        # decode_ms / sample_ms accumulate across n_decode steps. TTFT ≈ encode
        # + one decode step + one sample step; TPOT = avg decode+sample per
        # token. This avoids the wall-clock noise from worker contention.
        n_dec = delta["n_decode"]
        n_smp = max(1, delta.get("n_sample", n_dec))
        avg_decode_step = delta["decode_ms"] / max(1, n_dec)
        avg_sample_step = delta["sample_ms"] / n_smp
        ttft_ms = delta["encode_ms"] + avg_decode_step + avg_sample_step
        tpot_ms = (delta["decode_ms"] + delta["sample_ms"]) / n_dec
        n_tok = n_dec
    else:
        # Fallback when whisper_get_timings isn't bound. We can't separate
        # TTFT from total wall, so report NaN rather than mislead.
        text = " ".join(getattr(s, "text", "") for s in segments) if segments else ""
        n_tok = _count_tokens(text)
        ttft_ms = float("nan")
        tpot_ms = float("nan")

    return SampleRow(sample_id, audio_s, n_tok, ttft_ms, tpot_ms, e2e_ms)


# --- run loop ------------------------------------------------------------------

def _run_one(model_path: Path, n_threads: int, use_gpu: bool, language: str,
             dataset: str, max_samples: Optional[int], streaming: bool,
             *, mode: str, batch_size: int,
             ) -> tuple[list[SampleRow], float]:
    """Single → serial transcription. Batch/offline → thread-pool concurrency.

    'offline' is treated as 'batch with all-at-once submission', which for
    whisper.cpp is functionally identical to batch (no scheduler in between);
    we just keep the name parity with the other benchmarks.
    """
    samples = list(_iter_dataset(dataset, max_samples, streaming))
    print(f"[{dataset}] mode={mode} batch_size={batch_size} "
          f"n_threads={n_threads} samples={len(samples)}")

    rows: list[Optional[SampleRow]] = [None] * len(samples)
    t_start = time.perf_counter()

    if mode == "single":
        for i, (sid, audio, sr, _ref) in enumerate(samples):
            rows[i] = _transcribe_one(model_path, n_threads, use_gpu,
                                      sid, audio, sr, language)
    else:
        workers = max(1, batch_size)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_idx = {
                pool.submit(_transcribe_one, model_path, n_threads, use_gpu,
                            sid, audio, sr, language): i
                for i, (sid, audio, sr, _ref) in enumerate(samples)
            }
            for f in as_completed(future_to_idx):
                i = future_to_idx[f]
                rows[i] = f.result()

    wall = time.perf_counter() - t_start
    return [r for r in rows if r is not None], wall


def _summarize(rows: list[SampleRow], wall_s: float, *,
               model: str, precision: str, dataset: str,
               mode: str, batch_size: int, n_threads: int) -> Summary:
    if not rows:
        return Summary(model, precision, dataset, mode, batch_size, n_threads,
                       0, 0.0, wall_s, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    audio = sum(r.audio_seconds for r in rows)
    toks = sum(r.n_output_tokens for r in rows)
    ttfts = np.array([r.ttft_ms for r in rows])
    tpot_vals = [r.tpot_ms for r in rows if r.n_output_tokens > 1]
    tpots = np.array(tpot_vals) if tpot_vals else np.array([0.0])
    return Summary(
        model=model, precision=precision, dataset=dataset,
        mode=mode, batch_size=batch_size, n_threads=n_threads,
        n_samples=len(rows),
        total_audio_s=audio,
        total_wall_s=wall_s,
        throughput_xrt=audio / max(1e-9, wall_s),
        throughput_tokens_per_s=toks / max(1e-9, wall_s),
        avg_ttft_ms=float(ttfts.mean()),
        p50_ttft_ms=float(np.percentile(ttfts, 50)),
        p95_ttft_ms=float(np.percentile(ttfts, 95)),
        avg_tpot_ms=float(tpots.mean()),
        p50_tpot_ms=float(np.percentile(tpots, 50)),
        p95_tpot_ms=float(np.percentile(tpots, 95)),
    )


def _write_csv(rows: list[SampleRow], summary: Summary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# summary"])
        sfields = [fld.name for fld in fields(Summary)]
        w.writerow(sfields)
        w.writerow([getattr(summary, n) for n in sfields])
        w.writerow([])
        w.writerow(["# per-sample"])
        rfields = [fld.name for fld in fields(SampleRow)]
        w.writerow(rfields)
        for r in rows:
            w.writerow([getattr(r, n) for n in rfields])


def _print_summary(s: Summary) -> None:
    rows = [
        ("model", s.model),
        ("precision", s.precision),
        ("dataset", s.dataset),
        ("mode", f"{s.mode} (batch_size={s.batch_size})"
                 if s.mode != "single" else s.mode),
        ("n_threads (per worker)", str(s.n_threads)),
        ("samples", str(s.n_samples)),
        ("audio (s)", f"{s.total_audio_s:.1f}"),
        ("wall (s)", f"{s.total_wall_s:.1f}"),
        ("throughput (xRT)", f"{s.throughput_xrt:.2f}"),
        ("throughput (tok/s)", f"{s.throughput_tokens_per_s:.1f}"),
        ("TTFT avg / p50 / p95 (ms)",
         f"{s.avg_ttft_ms:.1f} / {s.p50_ttft_ms:.1f} / {s.p95_ttft_ms:.1f}"),
        ("TPOT avg / p50 / p95 (ms)",
         f"{s.avg_tpot_ms:.2f} / {s.p50_tpot_ms:.2f} / {s.p95_tpot_ms:.2f}"),
    ]
    width = max(len(k) for k, _ in rows)
    for k, v in rows:
        print(f"{k.ljust(width)}  {v}")


def benchmark(model_arg: str, *, precision: Optional[str], datasets: list[str],
              max_samples: Optional[int], n_threads: int, use_gpu: bool,
              language: str, mode: str, batch_size: int,
              streaming: bool) -> list[Path]:
    model_path = _resolve_ggml_model(model_arg)
    precision = precision or _infer_precision(model_path.name)
    model_slug = model_path.stem  # e.g. ggml-large-v3-q5_0

    # Physical-core-aware n_threads autotune. whisper.cpp's CPU kernels are
    # memory-bandwidth bound; SMT/HT siblings share the same execution units
    # and L1/L2, so logical cores rarely help and often hurt. We size per
    # whisper.cpp's own recommendation: n_threads_per_worker * workers ≈
    # physical_cores, never logical_cores.
    try:
        import psutil
        physical = psutil.cpu_count(logical=False) or os.cpu_count() or 4
    except ImportError:
        physical = os.cpu_count() or 4
    workers = batch_size if mode != "single" else 1
    if n_threads <= 0:
        n_threads = max(1, physical // max(1, workers))
        print(f"[autotune] n_threads not set → {n_threads} "
              f"(physical_cores={physical} / workers={workers})")
    elif n_threads * workers > physical:
        clamped = max(1, physical // max(1, workers))
        print(f"[autotune] n_threads={n_threads} × workers={workers} = "
              f"{n_threads * workers} > physical_cores={physical}; "
              f"clamping n_threads to {clamped} to avoid oversubscription")
        n_threads = clamped

    print(f"Loading whisper.cpp model: {model_path}  "
          f"(precision_label={precision}, mode={mode}, "
          f"batch_size={batch_size}, n_threads={n_threads}, use_gpu={use_gpu})")

    written: list[Path] = []
    for ds in datasets:
        print(f"\n=== {model_slug} | {precision} | {ds} | {mode} ===")
        rows, wall = _run_one(
            model_path, n_threads, use_gpu, language,
            ds, max_samples, streaming,
            mode=mode, batch_size=batch_size,
        )
        summary = _summarize(rows, wall, model=model_slug, precision=precision,
                             dataset=ds, mode=mode, batch_size=batch_size,
                             n_threads=n_threads)
        _print_summary(summary)

        stamp = _dt.datetime.now(EAST).strftime("%Y%m%d_%H%M")
        mode_tag = f"batch{batch_size}" if mode == "batch" else mode
        out_path = OUTPUT_DIR / f"{model_slug}_{precision}_{ds}_{mode_tag}_{stamp}.csv"
        _write_csv(rows, summary, out_path)
        print(f"wrote {out_path}")
        written.append(out_path)
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="whisper-large-v3",
                   help=f"GGML registry key, filename under models/whisper.cpp/, "
                        f"or absolute path. Known: {list(GGML_REGISTRY)}")
    p.add_argument("--precision", default=None,
                   help="Label for CSV filename (auto-inferred from ggml filename if omitted)")
    p.add_argument("--datasets", nargs="+", choices=["librispeech", "tedlium"],
                   default=["librispeech", "tedlium"])
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap samples per dataset (default: full split)")
    p.add_argument("--language", default="en",
                   help="2-letter language code passed to whisper.cpp (default: en)")
    p.add_argument("--n-threads", type=int, default=0,
                   help="whisper.cpp n_threads per worker. Default 0 = "
                        "auto-split physical cores across --batch-size workers.")
    p.add_argument("--use-gpu", action="store_true",
                   help="Try to enable GPU offload (requires whisper.cpp built "
                        "with CUDA/Metal AND pywhispercpp that exposes use_gpu).")
    p.add_argument("--mode", default="single",
                   choices=["single", "batch", "offline"],
                   help="single = 1 audio at a time (serial); "
                        "batch  = --batch-size audios in parallel through a "
                        "thread pool of whisper contexts (production deploy "
                        "shape for whisper.cpp); "
                        "offline = same as batch but submits everything upfront.")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Number of concurrent whisper.cpp worker contexts "
                        "(default: 4). Memory scales linearly — prefer "
                        "quantized GGML (q5_0/q8_0) at higher concurrency.")
    p.add_argument("--no-streaming", action="store_true",
                   help="Download datasets fully instead of streaming")
    args = p.parse_args()

    benchmark(
        args.model,
        precision=args.precision,
        datasets=args.datasets,
        max_samples=args.max_samples,
        n_threads=args.n_threads,
        use_gpu=args.use_gpu,
        language=args.language,
        mode=args.mode,
        batch_size=args.batch_size,
        streaming=not args.no_streaming,
    )


if __name__ == "__main__":
    main()
