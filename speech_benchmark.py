import argparse
import csv
import datetime as _dt
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional

import numpy as np
import psutil

_HERE = Path(__file__).resolve().parent
DATA_DIR = _HERE / "data"
OUTPUT_DIR = _HERE / "output"
HF_CACHE_DIR = DATA_DIR / "hf_datasets"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HOME", str(DATA_DIR / "hf_home"))

try:
    from .model_loader import MODEL_REGISTRY, load_model
except ImportError:
    from model_loader import MODEL_REGISTRY, load_model


@dataclass
class SampleResult:
    sample_id: str
    audio_seconds: float
    total_seconds: float
    ttft_seconds: float
    n_new_tokens: int
    wer: float = 0.0
    n_ref_words: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    reference: str = ""
    hypothesis: str = ""
    reference_normalized: str = ""
    hypothesis_normalized: str = ""


@dataclass
class BenchReport:
    timestamp: str
    model: str
    dataset: str
    device: str
    resolved_device: str
    dtype: str
    n_samples: int
    total_audio_s: float
    total_wall_s: float
    throughput_xrt: float
    avg_ttft_ms: float
    p50_ttft_ms: float
    p95_ttft_ms: float
    avg_tpot_ms: float
    wer: float
    sub_rate: float
    del_rate: float
    ins_rate: float
    peak_rss_mb: float
    peak_cuda_mb: Optional[float]


class MemoryMonitor:
    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.peak_rss = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc = psutil.Process(os.getpid())

    def _run(self):
        while not self._stop.is_set():
            rss = self._proc.memory_info().rss
            if rss > self.peak_rss:
                self.peak_rss = rss
            self._stop.wait(self.interval)

    def __enter__(self):
        self.peak_rss = self._proc.memory_info().rss
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)


def _make_streamer():
    from transformers.generation.streamers import BaseStreamer

    class TimingStreamer(BaseStreamer):
        def __init__(self):
            self.first_token_time: Optional[float] = None
            self.n_new_tokens: int = 0
            self.start: float = 0.0

        def begin(self, t0: float):
            self.start = t0
            self.first_token_time = None
            self.n_new_tokens = 0

        def put(self, value):
            if self.first_token_time is None:
                self.first_token_time = time.perf_counter() - self.start
            try:
                n = int(value.shape[-1]) if value.ndim >= 1 else 1
            except (AttributeError, IndexError):
                n = 1
            self.n_new_tokens += n

        def end(self):
            pass

    return TimingStreamer()


def _resample(audio: np.ndarray, sr: int, target_sr: int = 16000) -> np.ndarray:
    if sr == target_sr:
        return audio.astype(np.float32, copy=False)
    import librosa
    return librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)


def _decode_audio_field(a):
    """Return (np.float32 mono array, sample_rate). Accepts a HF Audio cell
    with decode=False ({path, bytes}) or already-decoded ({array, sampling_rate})."""
    import io
    import soundfile as sf
    if isinstance(a, dict) and "array" in a:
        return np.asarray(a["array"], dtype=np.float32), int(a["sampling_rate"])
    if a.get("bytes"):
        audio, sr = sf.read(io.BytesIO(a["bytes"]))
    else:
        audio, sr = sf.read(a["path"])
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32, copy=False), int(sr)


def _sample_id_from_hf(ex: dict, audio_cell: dict, fallback_idx: int) -> str:
    for key in ("id", "utt_id", "audio_id"):
        if ex.get(key):
            return str(ex[key])
    if isinstance(audio_cell, dict) and audio_cell.get("path"):
        return str(audio_cell["path"])
    return f"sample_{fallback_idx}"


def _iter_librispeech(split: str = "test.clean", max_samples: Optional[int] = None,
                      streaming: bool = True):
    from datasets import Audio
    ds = _load_hub_dataset("openslr/librispeech_asr", None, split, streaming)
    ds = ds.cast_column("audio", Audio(decode=False))
    for i, ex in enumerate(ds):
        if max_samples and i >= max_samples:
            return
        audio, sr = _decode_audio_field(ex["audio"])
        yield _sample_id_from_hf(ex, ex["audio"], i), audio, sr, ex["text"]


def _load_hub_dataset(repo: str, config: Optional[str], split: str, streaming: bool):
    from datasets import load_dataset
    from datasets.exceptions import DatasetNotFoundError
    load_kwargs = dict(split=split, streaming=streaming, cache_dir=str(HF_CACHE_DIR))
    try:
        return (load_dataset(repo, config, **load_kwargs)
                if config else load_dataset(repo, **load_kwargs))
    except DatasetNotFoundError as e:
        msg = str(e).lower()
        if "gated" in msg or "auth" in msg or "401" in msg or "403" in msg:
            raise SystemExit(
                f"\n[auth] '{repo}' is gated on the HuggingFace Hub.\n"
                f"  1. Visit https://huggingface.co/datasets/{repo} and click "
                f"'Agree and access repository'.\n"
                f"  2. Run: huggingface-cli login   (paste a Read token from "
                f"https://huggingface.co/settings/tokens)\n"
                f"  3. Re-run this command.\n"
            ) from e
        raise


def _iter_tedlium(split: str = "test", max_samples: Optional[int] = None,
                  streaming: bool = True, repo: str = "chengan/tedlium_small",
                  config: Optional[str] = None):
    from datasets import Audio
    ds = _load_hub_dataset(repo, config, split, streaming)
    ds = ds.cast_column("audio", Audio(decode=False))
    for i, ex in enumerate(ds):
        if max_samples and i >= max_samples:
            return
        text = ex.get("text") or ex.get("transcription") or ex.get("sentence") or ""
        if text.strip().lower() in ("ignore_time_segment_in_scoring", ""):
            continue
        audio, sr = _decode_audio_field(ex["audio"])
        yield _sample_id_from_hf(ex, ex["audio"], i), audio, sr, text


def _iter_chime6(manifest_path: str, max_samples: Optional[int] = None):
    """CHiME-6 isn't on HF; supply a JSONL manifest:
    {"audio": "/abs/path/utt.wav", "text": "reference transcript", "id": "<optional>"}
    """
    import soundfile as sf
    with open(manifest_path) as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                return
            entry = json.loads(line)
            audio, sr = sf.read(entry["audio"])
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            sid = entry.get("id") or entry.get("audio") or f"sample_{i}"
            yield str(sid), audio.astype(np.float32), int(sr), entry["text"]


DATASET_LOADERS = {
    "librispeech": _iter_librispeech,
    "tedlium": _iter_tedlium,
    "chime6": _iter_chime6,
}


def _is_ov_model(model) -> bool:
    return type(model).__module__.startswith("optimum.intel")


def _is_openvino_repo(model_name: str) -> bool:
    name = MODEL_REGISTRY.get(model_name, model_name).lower()
    return "-ov" in name or "openvino" in name


def _resolve_device(device: str, gpu: int, model_name: str) -> str:
    """Translate (device, gpu) → backend-specific string.
    torch:    cpu | cuda:N
    OpenVINO: CPU | GPU.N  (NB: for OV "GPU" means Intel iGPU/dGPU, not CUDA)
    """
    d = device.lower()
    if d not in ("cpu", "cuda"):
        raise ValueError(f"--device must be 'cpu' or 'cuda', got {device!r}")
    is_ov = _is_openvino_repo(model_name)
    if d == "cpu":
        return "CPU" if is_ov else "cpu"
    if is_ov:
        return f"GPU.{gpu}" if gpu else "GPU"
    return f"cuda:{gpu}"


def _run_one_dataset(
    *, model, processor, dataset: str, sample_iter,
    model_name: str, device: str, resolved_device: str,
    is_ov: bool, torch, torch_device, model_dtype,
    dtype_arg: Optional[str], language: str,
    samples_csv_path: Optional[Path] = None,
) -> BenchReport:
    import jiwer
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    spelling = getattr(processor.tokenizer, "english_spelling_normalizer", {})
    normalizer = EnglishTextNormalizer(spelling)

    gen_kwargs = dict(
        language=language,
        task="transcribe",
        max_new_tokens=440,
        return_timestamps=False,
    )

    if not is_ov and torch_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    streamer = _make_streamer()
    results: list[SampleResult] = []
    total_audio_s = 0.0
    total_wall_s = 0.0

    with MemoryMonitor() as mon:
        for sample_id, audio, sr, ref in sample_iter:
            audio_seconds = len(audio) / float(sr)
            audio = _resample(audio, sr, 16000)
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features
            if not is_ov:
                input_features = input_features.to(torch_device)
                if model_dtype is not None and input_features.dtype != model_dtype:
                    input_features = input_features.to(model_dtype)

            t0 = time.perf_counter()
            streamer.begin(t0)
            with torch.inference_mode():
                out = model.generate(
                    input_features=input_features,
                    streamer=streamer,
                    **gen_kwargs,
                )
            if not is_ov and torch_device.type == "cuda":
                torch.cuda.synchronize()
            total = time.perf_counter() - t0

            hyp = processor.batch_decode(out, skip_special_tokens=True)[0]
            results.append(SampleResult(
                sample_id=sample_id,
                audio_seconds=audio_seconds,
                total_seconds=total,
                ttft_seconds=streamer.first_token_time or total,
                n_new_tokens=streamer.n_new_tokens,
                reference=ref,
                hypothesis=hyp,
            ))
            total_audio_s += audio_seconds
            total_wall_s += total

    for r in results:
        r.reference_normalized = normalizer(r.reference)
        r.hypothesis_normalized = normalizer(r.hypothesis)
        ref_n = r.reference_normalized or "<empty>"
        hyp_n = r.hypothesis_normalized or "<empty>"
        wo_one = jiwer.process_words([ref_n], [hyp_n])
        r.n_ref_words = wo_one.hits + wo_one.substitutions + wo_one.deletions
        r.substitutions = wo_one.substitutions
        r.deletions = wo_one.deletions
        r.insertions = wo_one.insertions
        r.wer = float(wo_one.wer)
    refs_norm = [r.reference_normalized or "<empty>" for r in results]
    hyps_norm = [r.hypothesis_normalized or "<empty>" for r in results]
    wo = jiwer.process_words(refs_norm, hyps_norm)

    if samples_csv_path is not None:
        _write_samples_csv(results, samples_csv_path)
    total_ref_words = wo.hits + wo.substitutions + wo.deletions
    denom = max(1, total_ref_words)

    ttfts = np.array([r.ttft_seconds for r in results]) if results else np.array([0.0])
    tpots = [
        (r.total_seconds - r.ttft_seconds) / (r.n_new_tokens - 1)
        for r in results if r.n_new_tokens > 1
    ]

    peak_cuda_mb = None
    if not is_ov and torch_device.type == "cuda":
        peak_cuda_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    effective_dtype = (
        str(model_dtype).replace("torch.", "")
        if model_dtype is not None
        else (dtype_arg or "ov-native")
    )

    return BenchReport(
        timestamp=_dt.datetime.now().isoformat(timespec="seconds"),
        model=model_name,
        dataset=dataset,
        device=device,
        resolved_device=resolved_device,
        dtype=effective_dtype,
        n_samples=len(results),
        total_audio_s=total_audio_s,
        total_wall_s=total_wall_s,
        throughput_xrt=total_audio_s / max(1e-9, total_wall_s),
        avg_ttft_ms=float(ttfts.mean() * 1000),
        p50_ttft_ms=float(np.percentile(ttfts, 50) * 1000),
        p95_ttft_ms=float(np.percentile(ttfts, 95) * 1000),
        avg_tpot_ms=float(np.mean(tpots) * 1000) if tpots else 0.0,
        wer=float(wo.wer),
        sub_rate=wo.substitutions / denom,
        del_rate=wo.deletions / denom,
        ins_rate=wo.insertions / denom,
        peak_rss_mb=mon.peak_rss / (1024 ** 2),
        peak_cuda_mb=peak_cuda_mb,
    )


def benchmark_model(
    model_name: str,
    *,
    device: str = "cpu",
    gpu: int = 0,
    dtype: Optional[str] = None,
    datasets: Optional[list] = None,
    max_samples: Optional[int] = None,
    chime6_manifest: Optional[str] = None,
    librispeech_split: str = "test.clean",
    tedlium_split: str = "test",
    streaming: bool = True,
    language: str = "english",
    output_csv: Optional[str] = None,
) -> list:
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch

    if device == "cuda" and hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)

    datasets = datasets or list(DATASET_LOADERS)
    resolved_device = _resolve_device(device, gpu, model_name)
    model, processor = load_model(model_name, device=resolved_device, dtype=dtype)
    is_ov = _is_ov_model(model)
    torch_device = torch.device("cpu")
    model_dtype = None
    if not is_ov:
        model.eval()
        torch_device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype

    date = _dt.date.today().strftime("%Y%m%d")
    reports: list[BenchReport] = []
    for ds in datasets:
        if ds == "chime6":
            if not chime6_manifest:
                print(f"[skip] {ds}: --chime6-manifest not provided")
                continue
            sample_iter = _iter_chime6(chime6_manifest, max_samples)
        elif ds == "librispeech":
            sample_iter = _iter_librispeech(librispeech_split, max_samples, streaming)
        elif ds == "tedlium":
            sample_iter = _iter_tedlium(tedlium_split, max_samples, streaming)
        else:
            raise ValueError(f"unknown dataset: {ds}")

        samples_path = OUTPUT_DIR / f"{_slug(model_name)}_{ds}_{date}.csv"
        print(f"\n=== running {model_name} on {ds} ===")
        report = _run_one_dataset(
            model=model, processor=processor,
            dataset=ds, sample_iter=sample_iter,
            model_name=model_name, device=device, resolved_device=resolved_device,
            is_ov=is_ov, torch=torch, torch_device=torch_device, model_dtype=model_dtype,
            dtype_arg=dtype, language=language,
            samples_csv_path=samples_path,
        )
        _print_report(report)
        print(f"per-sample CSV: {samples_path}")
        reports.append(report)

    csv_path = Path(output_csv) if output_csv else _default_csv_path(model_name, date)
    _write_csv(reports, csv_path)
    print(f"\nsummary CSV: {csv_path}")
    return reports


def _slug(s: str) -> str:
    return s.replace("/", "__").replace(" ", "_")


def _default_csv_path(model_name: str, date: Optional[str] = None) -> Path:
    date = date or _dt.date.today().strftime("%Y%m%d")
    return OUTPUT_DIR / f"{_slug(model_name)}_{date}.csv"


def _write_csv(reports: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [f.name for f in fields(BenchReport)]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in reports:
            w.writerow([getattr(r, h) for h in header])


def _write_samples_csv(results: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [f.name for f in fields(SampleResult)]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in results:
            w.writerow([getattr(r, h) for h in header])


def _print_report(r: BenchReport) -> None:
    rows = [
        ("model", r.model),
        ("dataset", r.dataset),
        ("device", f"{r.device} ({r.resolved_device})"),
        ("dtype", r.dtype),
        ("samples", str(r.n_samples)),
        ("audio (s)", f"{r.total_audio_s:.1f}"),
        ("wall (s)", f"{r.total_wall_s:.1f}"),
        ("throughput (xRT)", f"{r.throughput_xrt:.2f}"),
        ("WER", f"{r.wer * 100:.2f}%"),
        ("substitution rate", f"{r.sub_rate * 100:.2f}%"),
        ("deletion rate", f"{r.del_rate * 100:.2f}%"),
        ("insertion rate", f"{r.ins_rate * 100:.2f}%"),
        ("TTFT avg / p50 / p95 (ms)",
         f"{r.avg_ttft_ms:.1f} / {r.p50_ttft_ms:.1f} / {r.p95_ttft_ms:.1f}"),
        ("TPOT avg (ms)", f"{r.avg_tpot_ms:.2f}"),
        ("peak RSS (MB)", f"{r.peak_rss_mb:.0f}"),
        ("peak CUDA (MB)", f"{r.peak_cuda_mb:.0f}" if r.peak_cuda_mb else "n/a"),
    ]
    width = max(len(k) for k, _ in rows)
    for k, v in rows:
        print(f"{k.ljust(width)}  {v}")


def main():
    p = argparse.ArgumentParser(
        description="Whisper benchmark — runs LibriSpeech + TED-LIUM + CHiME-6 for a single model"
    )
    p.add_argument("--model", required=True,
                   help=f"Registry key or HF repo. Known: {list(MODEL_REGISTRY)}")
    p.add_argument("--datasets", nargs="+", choices=list(DATASET_LOADERS),
                   default=list(DATASET_LOADERS),
                   help="Datasets to run (default: all three)")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--gpu", type=int, default=0,
                   help="GPU index when --device cuda (0=first, 1=second, ...)")
    p.add_argument("--dtype", default=None, help="fp16|fp32|bf16|auto (transformers path only)")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap samples per dataset (default: full split)")
    p.add_argument("--chime6-manifest", help="JSONL with {audio, text} per line")
    p.add_argument("--librispeech-split", default="test.clean")
    p.add_argument("--tedlium-split", default="test")
    p.add_argument("--no-streaming", action="store_true",
                   help="Download datasets fully into data/ instead of streaming")
    p.add_argument("--language", default="english")
    p.add_argument("--output-csv", default=None,
                   help="CSV path (default: output/{model}_{YYYYMMDD}.csv, overwritten)")
    args = p.parse_args()

    benchmark_model(
        args.model,
        device=args.device,
        gpu=args.gpu,
        dtype=args.dtype,
        datasets=args.datasets,
        max_samples=args.max_samples,
        chime6_manifest=args.chime6_manifest,
        librispeech_split=args.librispeech_split,
        tedlium_split=args.tedlium_split,
        streaming=not args.no_streaming,
        language=args.language,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
