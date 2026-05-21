"""vLLM-based Whisper benchmark — throughput, TTFT, TPOT on LibriSpeech / TED-LIUM.

Mirrors Intel's MLPerf Whisper SUT engine config and prompt format (see
whisper_mlperf/code/workspace/SUT.py and QSL.py). This is the path in the
project that exercises real CPU INT8 / INT4 GEMM kernels:

  - vLLM auto-detects the compressed-tensors `quantization_config` in
    whisper-large-v3-w8a8 / -w4a16 and dispatches Intel oneDNN INT8/INT4
    kernels on CPU. No re-quantization needed; load the existing checkpoint.
  - kv_cache_dtype='fp8' matches the MLPerf SUT for additional memory savings.
  - --numa-bind sets VLLM_CPU_OMP_THREADS_BIND (the env var the MLPerf SUT
    uses for per-instance core pinning).

Per-sample TTFT / TPOT come from driving llm_engine.step() ourselves and
recording arrival → first-token → last-token timestamps per request, same
pattern as the MLPerf SUT (so numbers are directly comparable).

Output: one CSV per (model, precision, dataset) under output/vllm/, named
    {model}_{precision}_{dataset}_{mode_tag}_{YYYYMMDD_HHMM}.csv
with HHMM in US-Eastern (matches the other benchmarks in this repo).

Requires a CPU-built vLLM (`pip install vllm` ships the CUDA build by
default; for CPU INT8 you need a vLLM built with VLLM_TARGET_DEVICE=cpu).
"""
import argparse
import csv
import datetime as _dt
import os
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np

_HERE = Path(__file__).resolve().parent
MODELS_DIR = _HERE / "models"
OUTPUT_DIR = _HERE / "output" / "vllm"
DATA_DIR = _HERE / "data"
HF_CACHE_DIR = DATA_DIR / "hf_datasets"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HOME", str(DATA_DIR / "hf_home"))

EAST = ZoneInfo("America/New_York")

# Same multimodal prompt template the MLPerf SUT uses (QSL.py:37-42).
WHISPER_PROMPT = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"

_VLLM_NVCC_MSG = """\
vLLM failed to find `nvcc` — a CUDA feature is JIT-compiling a kernel and
the CUDA Toolkit (which provides nvcc/ptxas) is missing. CUDA *driver* +
*runtime* alone aren't enough; nvcc only ships with the full Toolkit.

If the traceback mentions `(EngineCore pid=...)`, the trigger is almost
always `torch.compile` — vLLM v1 enables it by default and it JIT-generates
Triton/Inductor kernels at first forward pass. If the traceback mentions
fp8 cache or fp8 quant kernels, that's the second-most-common trigger.

Three ways out, in increasing order of effort:

  1. Disable torch.compile — pass `--enforce-eager`. Skips the JIT entirely
     while keeping the precompiled INT8/INT4 GEMM kernels active. Throughput
     is somewhat lower than the fully-compiled path but the model runs.

  2. Disable FP8 KV cache — pass `--kv-cache-dtype auto`. Only helps if FP8
     was the trigger; usually #1 is the real fix on v1.

  3. Install the CUDA Toolkit matching your driver (the full fix):
       nvidia-smi  # check "CUDA Version" top-right
       sudo apt install cuda-toolkit-<major>-<minor>
       export CUDA_HOME=/usr/local/cuda
       export PATH=$CUDA_HOME/bin:$PATH

vLLM's device target is compile-time (separate CUDA / CPU wheels); you can't
switch at runtime via flags.
"""


def _import_vllm():
    """Single entry point for vLLM imports. Converts the cryptic
    'Could not find nvcc' RuntimeError into actionable guidance."""
    try:
        from vllm import LLM, SamplingParams
        return LLM, SamplingParams
    except ImportError as e:
        raise RuntimeError(
            f"vLLM is not installed: {e}\n"
            "Install via the official docs: https://docs.vllm.ai/en/latest/getting_started/installation.html"
        ) from e
    except RuntimeError as e:
        msg = str(e).lower()
        if "nvcc" in msg or "cuda_home" in msg:
            raise RuntimeError(_VLLM_NVCC_MSG) from e
        raise

_DTYPE_TO_TAG = {
    "bfloat16": "bf16", "float16": "fp16",
    "float32": "fp32", "auto": "auto",
}


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
    timestamp: str
    model: str
    precision: str
    dataset: str
    mode: str
    batch_size: int
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


# --- dataset iterators (reused from speech_benchmark.py) -----------------------

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
    return librosa.resample(audio.astype(np.float32),
                            orig_sr=sr, target_sr=target_sr)


# --- model / precision helpers -------------------------------------------------

def _infer_precision(model_dir: Path) -> Optional[str]:
    name = model_dir.name.lower()
    for tag in ("w8a8", "w4a16", "w4a4", "int8", "int4", "fp16", "bf16", "fp32"):
        if tag in name:
            return tag
    return None


def _resolve_model_path(model_arg: str) -> Path:
    """Accept an absolute path, a folder under models/, or a registry
    short-name from model_loader.MODEL_REGISTRY (e.g. whisper-large-v3-w8a8)."""
    p = Path(model_arg)
    if p.is_absolute() and p.exists():
        return p
    try:
        from model_loader import MODEL_REGISTRY, _local_path, _resolve_repo_id
        if model_arg in MODEL_REGISTRY:
            local = _local_path(_resolve_repo_id(model_arg))
            if local.exists():
                return local
    except ImportError:
        pass
    candidate = MODELS_DIR / model_arg
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Could not find model '{model_arg}'. Tried registry, {p}, and "
        f"{candidate}. Available under {MODELS_DIR}: "
        f"{sorted(x.name for x in MODELS_DIR.iterdir())}"
    )


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _maybe_set_numa_pinning(numa_bind: Optional[str]) -> None:
    """Honor the MLPerf SUT pattern: bind vLLM CPU OMP threads to a core range
    via VLLM_CPU_OMP_THREADS_BIND. Must be set BEFORE the vLLM engine is built
    — the CPU backend reads it once at startup."""
    if numa_bind:
        os.environ["VLLM_CPU_OMP_THREADS_BIND"] = numa_bind
        print(f"[numa] VLLM_CPU_OMP_THREADS_BIND={numa_bind}")


# --- engine build --------------------------------------------------------------

def _make_llm_kwargs(model_path: Path, dtype: str, kv_cache_dtype: str,
                     max_num_seqs: int, max_model_len: int,
                     max_num_batched_tokens: int, device: str,
                     enforce_eager: bool) -> dict:
    """Build the kwargs dict for vllm.LLM(...). Pulled out so worker
    processes in multi-instance mode can rebuild an identical engine.
    Mirrors MLPerf Whisper SUT engine config (SUT.py:137-150)."""
    kwargs = dict(
        model=str(model_path),
        dtype=dtype,
        skip_tokenizer_init=False,
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        limit_mm_per_prompt={"audio": 1},
        kv_cache_dtype=kv_cache_dtype,
        enforce_eager=enforce_eager,
    )
    if device == "cuda":
        kwargs["gpu_memory_utilization"] = 0.95
    return kwargs


def _build_llm(**llm_kwargs):
    LLM, _ = _import_vllm()
    return LLM(**llm_kwargs)


# --- benchmark core ------------------------------------------------------------

def _materialize_dataset(dataset: str, max_samples: Optional[int],
                         streaming: bool) -> list[dict]:
    """Pull all audio samples into memory once. In multi-instance mode the
    parent does this, then ships slices to workers — saves N×download cost."""
    samples = []
    for sid, audio, sr, _ref in _iter_dataset(dataset, max_samples, streaming):
        audio16 = _resample(audio, sr, 16000)
        samples.append({
            "sample_id": sid,
            "audio_seconds": len(audio16) / 16000.0,
            "prompt": {
                "prompt": WHISPER_PROMPT,
                "multi_modal_data": {"audio": (audio16, 16000)},
            },
        })
    return samples


def _process_prepared_samples(llm, samples: list[dict], *,
                              mode: str, batch_size: int,
                              max_new_tokens: int,
                              log_prefix: str = "") -> list[SampleRow]:
    """Drive the engine step-by-step over a pre-loaded sample list.
    Returns SampleRow per input. Same step-loop pattern as MLPerf SUT."""
    _, SamplingParams = _import_vllm()

    sampling = SamplingParams(temperature=0, top_p=1.0,
                              max_tokens=max_new_tokens)

    if mode == "single":
        chunk_size = 1
    elif mode == "batch":
        chunk_size = max(1, batch_size)
    elif mode == "offline":
        chunk_size = len(samples) or 1
    else:
        raise ValueError(f"unknown mode: {mode}")

    n_chunks = (len(samples) + chunk_size - 1) // chunk_size
    print(f"{log_prefix}mode={mode} chunk_size={chunk_size} → "
          f"{len(samples)} requests in {n_chunks} chunk(s)")

    engine = llm.llm_engine
    rows: list[SampleRow] = []

    for chunk_idx, chunk_start in enumerate(range(0, len(samples), chunk_size)):
        chunk = samples[chunk_start:chunk_start + chunk_size]

        arrival: dict[str, float] = {}
        first_tok: dict[str, float] = {}
        last_tok: dict[str, float] = {}
        final_n_tok: dict[str, int] = {}

        for i, s in enumerate(chunk):
            rid = f"c{chunk_idx}_r{i}"
            engine.add_request(rid, s["prompt"], sampling)
            arrival[rid] = time.perf_counter()

        while engine.has_unfinished_requests():
            step_outputs = engine.step()
            now = time.perf_counter()
            for out in step_outputs:
                rid = out.request_id
                if rid not in arrival:
                    continue
                n_tok = len(out.outputs[0].token_ids) if out.outputs else 0
                if n_tok > 0 and rid not in first_tok:
                    first_tok[rid] = now
                if n_tok > final_n_tok.get(rid, 0):
                    last_tok[rid] = now
                    final_n_tok[rid] = n_tok

        for i, s in enumerate(chunk):
            rid = f"c{chunk_idx}_r{i}"
            a = arrival[rid]
            ft = first_tok.get(rid, a)
            lt = last_tok.get(rid, ft)
            n_tok = final_n_tok.get(rid, 0)
            ttft = (ft - a) * 1000.0
            tpot = ((lt - ft) * 1000.0 / (n_tok - 1)) if n_tok > 1 else 0.0
            e2e = (lt - a) * 1000.0
            rows.append(SampleRow(s["sample_id"], s["audio_seconds"],
                                  n_tok, ttft, tpot, e2e))

    return rows


# --- multi-instance machinery --------------------------------------------------

def _physical_cores() -> int:
    try:
        import psutil
        return psutil.cpu_count(logical=False) or os.cpu_count() or 1
    except ImportError:
        return os.cpu_count() or 1


def _compute_core_bindings(num_instances: int,
                           override_range: Optional[str] = None) -> list[str]:
    """Return ['lo0-hi0', 'lo1-hi1', ...] — one core range per instance.
    If override_range is 'A-B', subdivide that range; otherwise use all
    physical cores. Cores are split contiguously so each instance stays
    on the same NUMA node when the system topology is socket-contiguous
    (the typical Xeon layout)."""
    if override_range:
        lo, hi = (int(x) for x in override_range.split("-"))
        total = hi - lo + 1
        base = lo
    else:
        total = _physical_cores()
        base = 0
    if num_instances <= 0:
        num_instances = 1
    per = max(1, total // num_instances)
    out = []
    for i in range(num_instances):
        a = base + i * per
        b = base + (i + 1) * per - 1 if i < num_instances - 1 else base + total - 1
        out.append(f"{a}-{b}")
    return out


# Top-level (picklable) worker for multi-instance mode. Must NOT close over
# state from benchmark(); everything comes through args.
def _instance_worker(instance_id: int, core_bind: str,
                     samples_shard: list[dict], llm_kwargs: dict,
                     mode: str, batch_size: int, max_new_tokens: int,
                     cpu_kvcache_gib: int, result_queue):
    try:
        os.environ["VLLM_CPU_OMP_THREADS_BIND"] = core_bind
        # vLLM CPU autosizes KV cache from total available RAM. On big-memory
        # boxes that overshoots wildly (e.g. ~1 TB requested per instance) and
        # OOMs in multi-instance mode. Whisper's 448-token context needs <5 GiB
        # even at batch=64, so we cap it explicitly.
        os.environ["VLLM_CPU_KVCACHE_SPACE"] = str(cpu_kvcache_gib)
        prefix = f"[inst {instance_id} cores={core_bind}] "
        print(f"{prefix}VLLM_CPU_KVCACHE_SPACE={cpu_kvcache_gib} GiB; "
              f"loading vLLM, {len(samples_shard)} samples to process")
        llm = _build_llm(**llm_kwargs)
        print(f"{prefix}engine ready, starting transcribe loop")
        t0 = time.perf_counter()
        rows = _process_prepared_samples(llm, samples_shard,
                                         mode=mode, batch_size=batch_size,
                                         max_new_tokens=max_new_tokens,
                                         log_prefix=prefix)
        worker_wall = time.perf_counter() - t0
        print(f"{prefix}done — {len(rows)} samples in {worker_wall:.1f}s")
        result_queue.put({"instance_id": instance_id, "rows": rows,
                          "worker_wall_s": worker_wall})
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        result_queue.put({"instance_id": instance_id, "error": str(e),
                          "traceback": tb})


def _run_multi_instance(num_instances: int, samples: list[dict],
                        llm_kwargs: dict, mode: str, batch_size: int,
                        max_new_tokens: int,
                        numa_bind: Optional[str],
                        cpu_kvcache_gib: int
                        ) -> tuple[list[SampleRow], float]:
    """Spawn `num_instances` vLLM workers, one per pinned core range, and
    shard the sample list round-robin across them. Returns aggregated rows
    and end-to-end wall (parent's perspective: spawn → all workers done)."""
    import multiprocessing as mp

    bindings = _compute_core_bindings(num_instances, override_range=numa_bind)
    shards = [samples[i::num_instances] for i in range(num_instances)]

    print(f"[multi] num_instances={num_instances} bindings={bindings}")
    print(f"[multi] shard sizes: {[len(s) for s in shards]}")

    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    procs = []
    t_start = time.perf_counter()
    for i in range(num_instances):
        p = ctx.Process(
            target=_instance_worker,
            args=(i, bindings[i], shards[i], llm_kwargs,
                  mode, batch_size, max_new_tokens, cpu_kvcache_gib, result_q),
        )
        p.start()
        procs.append(p)

    results = []
    errors = []
    for _ in range(num_instances):
        r = result_q.get()
        if "error" in r:
            errors.append(r)
        else:
            results.append(r)

    for p in procs:
        p.join()

    wall = time.perf_counter() - t_start

    if errors:
        for e in errors:
            print(f"\n[inst {e['instance_id']}] FAILED:\n{e['traceback']}")
        raise RuntimeError(
            f"{len(errors)}/{num_instances} instances failed (see tracebacks above)"
        )

    # Reassemble in input order (round-robin shards → interleave back)
    by_id = {r["instance_id"]: r["rows"] for r in results}
    all_rows: list[SampleRow] = []
    cursors = [0] * num_instances
    for i in range(len(samples)):
        inst = i % num_instances
        rows_i = by_id.get(inst, [])
        if cursors[inst] < len(rows_i):
            all_rows.append(rows_i[cursors[inst]])
            cursors[inst] += 1

    return all_rows, wall


def _summarize(rows: list[SampleRow], wall_s: float, *,
               model: str, precision: str, dataset: str,
               mode: str, batch_size: int) -> Summary:
    ts = _dt.datetime.now().isoformat(timespec="seconds")
    if not rows:
        return Summary(ts, model, precision, dataset, mode, batch_size, 0,
                       0.0, wall_s, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    audio = sum(r.audio_seconds for r in rows)
    toks = sum(r.n_output_tokens for r in rows)
    ttfts = np.array([r.ttft_ms for r in rows])
    tpot_vals = [r.tpot_ms for r in rows if r.n_output_tokens > 1]
    tpots = np.array(tpot_vals) if tpot_vals else np.array([0.0])
    return Summary(
        timestamp=ts,
        model=model, precision=precision, dataset=dataset,
        mode=mode, batch_size=batch_size,
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


def benchmark(model_arg: str, *, precision: Optional[str], dtype: str,
              kv_cache_dtype: str, datasets: list[str],
              max_samples: Optional[int], max_new_tokens: int,
              max_num_seqs: int, max_model_len: int,
              max_num_batched_tokens: int, device: str,
              mode: str, batch_size: int, streaming: bool,
              numa_bind: Optional[str],
              enforce_eager: bool,
              num_instances: int,
              cpu_kvcache_gib: int) -> list[Path]:
    model_path = _resolve_model_path(model_arg)
    precision = (precision or _infer_precision(model_path)
                 or _DTYPE_TO_TAG.get(dtype, dtype))
    model_slug = model_path.name
    resolved_device = _resolve_device(device)

    # Whisper's audio encoder is 1500 tokens / 30s clip. For batched prefill
    # we need max_num_batched_tokens ≥ N * 1500 + a bit of decoder headroom,
    # otherwise vLLM serializes encoders within a batch (huge TTFT hit).
    concurrency = (batch_size if mode == "batch"
                   else (1 if mode == "single" else 0))
    if concurrency >= 2:
        needed = concurrency * 1500 + 256
        if max_num_batched_tokens < needed:
            print(f"[autotune] bumping max_num_batched_tokens "
                  f"{max_num_batched_tokens} → {needed} so {concurrency} "
                  f"encoder prefills (1500 tok each) can co-batch")
            max_num_batched_tokens = needed

    llm_kwargs = _make_llm_kwargs(model_path, dtype, kv_cache_dtype,
                                  max_num_seqs, max_model_len,
                                  max_num_batched_tokens,
                                  resolved_device, enforce_eager)

    print(f"vLLM config: {model_path}  "
          f"(device={resolved_device}, dtype={dtype}, "
          f"kv_cache_dtype={kv_cache_dtype}, precision_label={precision}, "
          f"mode={mode}, batch_size={batch_size}, "
          f"max_num_batched_tokens={max_num_batched_tokens}, "
          f"num_instances={num_instances}, "
          f"cpu_kvcache_gib={cpu_kvcache_gib})")
    if resolved_device == "cpu":
        print(f"[cpu] Each instance reserves VLLM_CPU_KVCACHE_SPACE="
              f"{cpu_kvcache_gib} GiB. Total: "
              f"{cpu_kvcache_gib * max(1, num_instances)} GiB across "
              f"{max(1, num_instances)} instance(s).")

    # Single-instance path: build the engine once and reuse across datasets.
    # Multi-instance: parent doesn't build the engine — workers do — so we
    # materialize the dataset per-loop and dispatch through _run_multi_instance.
    llm = None
    if num_instances <= 1:
        if numa_bind:
            _maybe_set_numa_pinning(numa_bind)
        # Cap CPU KV cache size to avoid vLLM auto-sizing from total RAM
        # and OOMing (see _instance_worker for the multi-instance equivalent).
        if resolved_device == "cpu":
            os.environ["VLLM_CPU_KVCACHE_SPACE"] = str(cpu_kvcache_gib)
            print(f"[cpu] VLLM_CPU_KVCACHE_SPACE={cpu_kvcache_gib} GiB")
        llm = _build_llm(**llm_kwargs)

    written: list[Path] = []
    for ds in datasets:
        print(f"\n=== {model_slug} | {precision} | {ds} | {mode} "
              f"| instances={num_instances} ===")
        samples = _materialize_dataset(ds, max_samples, streaming)
        print(f"[{ds}] materialized {len(samples)} samples")

        if num_instances <= 1:
            t0 = time.perf_counter()
            rows = _process_prepared_samples(
                llm, samples, mode=mode, batch_size=batch_size,
                max_new_tokens=max_new_tokens, log_prefix=f"[{ds}] ",
            )
            wall = time.perf_counter() - t0
        else:
            rows, wall = _run_multi_instance(
                num_instances, samples, llm_kwargs,
                mode=mode, batch_size=batch_size,
                max_new_tokens=max_new_tokens, numa_bind=numa_bind,
                cpu_kvcache_gib=cpu_kvcache_gib,
            )

        summary = _summarize(rows, wall, model=model_slug, precision=precision,
                             dataset=ds, mode=mode, batch_size=batch_size)
        _print_summary(summary)

        stamp = _dt.datetime.now(EAST).strftime("%Y%m%d_%H%M")
        mode_tag = f"batch{batch_size}" if mode == "batch" else mode
        if num_instances > 1:
            mode_tag = f"{mode_tag}_inst{num_instances}"
        out_path = OUTPUT_DIR / f"{model_slug}_{precision}_{ds}_{mode_tag}_{stamp}.csv"
        _write_csv(rows, summary, out_path)
        print(f"wrote {out_path}")
        written.append(out_path)
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="whisper-large-v3-w8a8",
                   help="Registry short-name (e.g. whisper-large-v3-w8a8, "
                        "whisper-large-v3-w4a16, whisper-large-v3), folder "
                        "under models/, or absolute path. Default: w8a8.")
    p.add_argument("--precision", default=None,
                   help="Label for CSV filename (auto-inferred from model name).")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32", "auto"],
                   help="vLLM compute dtype for non-quantized layers "
                        "(MLPerf SUT uses bfloat16; recommended on CPU).")
    p.add_argument("--kv-cache-dtype", default="fp8",
                   help="vLLM kv_cache_dtype. Default 'fp8' matches MLPerf SUT; "
                        "use 'auto' on builds without FP8 KV support.")
    p.add_argument("--datasets", nargs="+", choices=["librispeech", "tedlium"],
                   default=["librispeech", "tedlium"])
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap samples per dataset (default: full split)")
    p.add_argument("--max-new-tokens", type=int, default=200,
                   help="Per-request output cap (MLPerf SUT: 200)")
    p.add_argument("--max-num-seqs", type=int, default=64,
                   help="vLLM scheduler concurrent-request cap (MLPerf SUT: 64)")
    p.add_argument("--max-model-len", type=int, default=448,
                   help="vLLM max context length (MLPerf SUT: 448 = Whisper max)")
    p.add_argument("--max-num-batched-tokens", type=int, default=2048,
                   help="Per-step token budget. Auto-bumped if smaller than "
                        "concurrency × 1500 (Whisper encoder length).")
    p.add_argument("--device", default="cpu",
                   choices=["auto", "cpu", "cuda"],
                   help="vLLM device. Requires a matching vLLM build "
                        "(CPU-built vLLM for --device cpu).")
    p.add_argument("--mode", default="offline",
                   choices=["single", "batch", "offline"],
                   help="single = 1 request at a time (single-stream latency); "
                        "batch  = --batch-size requests at a time (multi-stream); "
                        "offline = submit all upfront, drain once "
                        "(max throughput, default).")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Concurrency for --mode batch (default: 8)")
    p.add_argument("--numa-bind", default=None,
                   help="Optional core range for VLLM_CPU_OMP_THREADS_BIND, "
                        "e.g. '0-31'. Mirrors MLPerf SUT's per-instance pinning. "
                        "Single-socket: pin to that socket's cores to avoid "
                        "cross-socket memory traffic.")
    p.add_argument("--cpu-kvcache-gib", type=int, default=30,
                   help="Per-instance VLLM_CPU_KVCACHE_SPACE in GiB "
                        "(DEFAULT: 30 GiB). vLLM CPU autosizes from total "
                        "available RAM, which on big-memory boxes overshoots "
                        "(e.g. ~1 TB per instance, OOMs in multi-instance "
                        "mode). vLLM enforces a per-request floor — "
                        "whisper-large-v3 needs ~7.5 GiB even at batch=1 "
                        "with max_model_len=448, and the requirement grows "
                        "with batch size and concurrent requests. 30 GiB "
                        "covers batch_size up to ~96.")
    p.add_argument("--num-instances", type=int, default=1,
                   help="Spawn N independent vLLM engines, each pinned to a "
                        "contiguous slice of physical cores, processing 1/N of "
                        "the samples in parallel. Mirrors the MLPerf SUT shape "
                        "(N independent workers per node, loadgen distributes). "
                        "Use N = sockets × instances_per_socket. Default: 1.")
    p.add_argument("--enforce-eager", action="store_true",
                   help="Disable torch.compile and CUDA graphs in vLLM. Set this "
                        "if you hit `Could not find nvcc` errors during EngineCore "
                        "startup — torch.compile JIT-generates Triton kernels that "
                        "need the CUDA Toolkit (nvcc/ptxas). Tradeoff: somewhat "
                        "lower throughput vs the fully-compiled path, but you can "
                        "still run the INT8/INT4 GEMM kernels (those are precompiled "
                        "in the vLLM wheel).")
    p.add_argument("--no-streaming", action="store_true",
                   help="Download datasets fully instead of streaming")
    args = p.parse_args()

    benchmark(
        args.model,
        precision=args.precision,
        dtype=args.dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        datasets=args.datasets,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        device=args.device,
        mode=args.mode,
        batch_size=args.batch_size,
        streaming=not args.no_streaming,
        numa_bind=args.numa_bind,
        enforce_eager=args.enforce_eager,
        num_instances=args.num_instances,
        cpu_kvcache_gib=args.cpu_kvcache_gib,
    )


if __name__ == "__main__":
    main()
