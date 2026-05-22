"""vLLM-based Whisper benchmark — throughput, TTFT, TPOT on LibriSpeech / TED-LIUM.

Mirrors Intel's MLPerf Whisper SUT engine config and dispatch pattern (see
whisper_mlperf/code/workspace/SUT.py and QSL.py). This is the path in the
project that exercises real CPU INT8 / INT4 GEMM kernels:

  - vLLM auto-detects the compressed-tensors `quantization_config` in
    whisper-large-v3-w8a8 / -w4a16 and dispatches Intel oneDNN INT8/INT4
    kernels on CPU. No re-quantization needed; load the existing checkpoint.
  - kv_cache_dtype='fp8' matches the MLPerf SUT.
  - --numa-bind sets VLLM_CPU_OMP_THREADS_BIND, the env var the MLPerf SUT
    uses for per-instance core pinning.

Dispatch model — three scenarios, all backed by the same continuous-batching
loop with a different in-flight cap (matches MLPerf SUT's process_queries):

  --mode single        cap = 1                  (≈ MLPerf SingleStream;
                                                 per-sample latency)
  --mode batch         cap = --batch-size       (steady-state multi-stream;
                                                 measure under fixed load)
  --mode offline       cap = N_samples          (≈ MLPerf Offline; vLLM caps
                                                 actual concurrency at
                                                 max_num_seqs internally)

Per-sample TTFT / TPOT come from driving llm_engine.step() ourselves and
recording arrival → first-token → last-token per request — same pattern as
the MLPerf SUT, so numbers are directly comparable.

Output: one CSV per (model, precision, dataset) under output/vllm/, named
    {model}_{precision}_{dataset}_{mode_tag}_{YYYYMMDD_HHMM}.csv
with HHMM in US-Eastern (matches the other benchmarks in this repo).

Requires a CPU-built vLLM (`pip install vllm` ships the CUDA build by
default; for CPU INT8 you need a vLLM built with VLLM_TARGET_DEVICE=cpu).

Quick-start invocations for the W8A8 model:

    # 1) single-stream latency
    python vllm_speech_benchmark.py \\
        --model whisper-large-v3-w8a8 --device cpu \\
        --mode single --datasets librispeech --max-samples 50

    # 2) fixed-concurrency batched
    python vllm_speech_benchmark.py \\
        --model whisper-large-v3-w8a8 --device cpu \\
        --mode batch --batch-size 8 --datasets librispeech --max-samples 200

    # 3) offline throughput (matches MLPerf Offline shape)
    python vllm_speech_benchmark.py \\
        --model whisper-large-v3-w8a8 --device cpu \\
        --mode offline --datasets librispeech --max-samples 500
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
    p99_ttft_ms: float
    avg_tpot_ms: float
    p50_tpot_ms: float
    p95_tpot_ms: float
    p99_tpot_ms: float


# --- dataset iterators (reused from speech_benchmark.py) -----------------------

# Default location of the MLPerf-published Whisper manifest + wav cache.
_MLPERF_MANIFEST_DEFAULT = (
    _HERE / "whisper_mlperf" / "data" / "dev-all-repack.json"
)
_MLPERF_DATA_DIR_DEFAULT = _HERE / "whisper_mlperf" / "data"


def _iter_mlperf_devallrepack(
    max_samples: Optional[int] = None,
    streaming: bool = True,   # ignored; kept for signature parity
    manifest_path: Optional[str] = None,
    data_dir: Optional[str] = None,
):
    """Iterate the MLPerf Whisper SUT's `dev-all-repack` manifest.

    This is LibriSpeech dev clips that have been pre-concatenated into ~30 s
    super-clips (mean 24 s, 25% ≥ 28 s), with mean ~83 output tokens per
    sample — vs ~26 tokens/sample on `test.clean`. Using this dataset makes
    our tok/s directly comparable to the number MLPerf publishes, because
    encoder cost is amortized over more decode work per sample.

    Mirrors QSL.py `load_sample_from_file` (takes the first wav per entry
    at speed=1.0) and manifest.py path resolution. The manifest stores
    `/data/dev-all-repack/foo.wav` (container path); we rebase to the
    local data dir.
    """
    import json
    import soundfile as sf

    mpath = Path(manifest_path) if manifest_path else _MLPERF_MANIFEST_DEFAULT
    ddir = Path(data_dir) if data_dir else _MLPERF_DATA_DIR_DEFAULT

    if not mpath.exists():
        raise FileNotFoundError(
            f"MLPerf manifest not found at {mpath}. Pass --mlperf-manifest "
            f"to override. Expected the dev-all-repack.json shipped under "
            f"speech_ai/whisper_mlperf/data/."
        )

    with open(mpath) as f:
        entries = json.load(f)

    yielded = 0
    for entry in entries:
        if max_samples is not None and yielded >= max_samples:
            return
        # MLPerf takes first file at speed=1.0 (QSL.py:40). Mirror that.
        candidates = [
            f for f in entry.get("files", []) if f.get("speed", 1) == 1
        ]
        if not candidates:
            continue
        fname = candidates[0]["fname"]
        # Manifest paths are absolute container paths like "/data/foo/x.wav".
        # Rebase: strip a leading "/data/" then join with ddir. Falls back to
        # straight basename if the prefix doesn't match (manifest re-rooted).
        if fname.startswith("/data/"):
            local = ddir / fname[len("/data/"):]
        elif Path(fname).is_absolute():
            local = ddir / Path(fname).name
        else:
            local = ddir / fname
        if not local.exists():
            # Try one more rescue: basename in dev-all-repack/.
            alt = ddir / "dev-all-repack" / Path(fname).name
            if alt.exists():
                local = alt
            else:
                print(f"[mlperf] skip — wav not found: {local}")
                continue
        audio, sr = sf.read(str(local))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        text = entry.get("transcript", "")
        sample_id = local.name
        yielded += 1
        yield sample_id, audio.astype(np.float32, copy=False), int(sr), text


def _iter_dataset(name: str, max_samples: Optional[int], streaming: bool):
    from speech_benchmark import _iter_librispeech, _iter_tedlium
    if name == "librispeech":
        return _iter_librispeech(max_samples=max_samples, streaming=streaming)
    if name == "tedlium":
        return _iter_tedlium(max_samples=max_samples, streaming=streaming)
    if name == "mlperf":
        return _iter_mlperf_devallrepack(max_samples=max_samples,
                                         streaming=streaming)
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


def _max_concurrent_for_mode(mode: str, batch_size: int, n_samples: int) -> int:
    """Translate scenario name → in-flight cap. Matches MLPerf SUT's pattern:
    one add_request per query, continuous batching keeps up to N in flight."""
    if mode == "single":
        return 1
    if mode == "batch":
        return max(1, batch_size)
    if mode == "offline":
        # Submit everything; vLLM enforces its own internal cap via max_num_seqs.
        return max(1, n_samples)
    raise ValueError(f"unknown mode: {mode}")


def _process_prepared_samples(llm, samples: list[dict], *,
                              mode: str, batch_size: int,
                              max_new_tokens: int,
                              log_prefix: str = "") -> list[SampleRow]:
    """Drive the engine over a pre-loaded sample list with continuous batching.

    Keeps `max_concurrent` requests in flight at all times. When one finishes,
    the next is added on the same step — no drain-then-refill stalls. Matches
    MLPerf SUT.Instance.process_queries (SUT.py:167-240): add a query whenever
    `unfinished_requests < cap`, step once, repeat.
    """
    _, SamplingParams = _import_vllm()
    sampling = SamplingParams(temperature=0, top_p=1.0,
                              max_tokens=max_new_tokens)

    max_concurrent = _max_concurrent_for_mode(mode, batch_size, len(samples))
    print(f"{log_prefix}mode={mode} max_concurrent={max_concurrent} → "
          f"{len(samples)} request(s) via continuous batching")

    engine = llm.llm_engine

    arrival: dict[str, float] = {}
    first_tok: dict[str, float] = {}
    last_tok: dict[str, float] = {}
    final_n_tok: dict[str, int] = {}
    sample_for_rid: dict[str, dict] = {}
    rid_order: list[str] = []

    sample_iter = iter(samples)
    next_rid = 0

    def _try_dispatch() -> None:
        nonlocal next_rid
        # get_num_unfinished_requests() is the authoritative live count; using
        # it instead of a Python-side counter avoids drift if vLLM aborts /
        # finalizes a request between our checks.
        while engine.get_num_unfinished_requests() < max_concurrent:
            try:
                s = next(sample_iter)
            except StopIteration:
                return
            rid = f"r{next_rid}"
            next_rid += 1
            engine.add_request(rid, s["prompt"], sampling)
            arrival[rid] = time.perf_counter()
            sample_for_rid[rid] = s
            rid_order.append(rid)

    _try_dispatch()
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
        _try_dispatch()

    rows: list[SampleRow] = []
    for rid in rid_order:
        s = sample_for_rid[rid]
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


# --- CPU autotune (mirrors whisper_mlperf configure_workload.sh) --------------
#
# Sizing constants. KV cache per sequence at INT8:
#   (1500 mel + 448 max decoder) × 32 layers × 1280 d_model × 2 (K+V) × 1 byte
# = 1948 × 32 × 1280 × 2 bytes per batched sequence.
_CACHE_BYTES_PER_BATCH = 1948 * 32 * 1280 * 2
_MEM_MODEL_GIB = 4
_CORES_PER_INST_MIN = 4
_CORES_PER_INST_MAX = 10
_BATCH_SIZE_CANDIDATES = (96, 64, 48, 32)


def _detect_numa_nodes() -> int:
    nodes = list(Path("/sys/devices/system/node").glob("node[0-9]*"))
    return max(1, len(nodes))


def _detect_l3_clusters() -> int:
    """Count unique L3 (level=3) shared_cpu_map values across all CPUs.
    On Granite Rapids and other clustered-L3 parts this exceeds the NUMA
    node count and is a better unit for sizing per-instance core slices."""
    maps = set()
    for index_dir in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cache/index*"):
        level = index_dir / "level"
        shared = index_dir / "shared_cpu_map"
        try:
            if level.read_text().strip() == "3" and shared.exists():
                maps.add(shared.read_text().strip())
        except OSError:
            continue
    return max(1, len(maps))


def _available_mem_gib() -> int:
    """psutil "available" matches `free -g | awk /Mem:/{print $7}` (the value
    MLPerf reads). Floored to int GiB to match the shell arithmetic."""
    try:
        import psutil
        return int(psutil.virtual_memory().available // (1024 ** 3))
    except ImportError:
        return 0


def _autotune_cpu(use_l3_clusters: bool = False) -> dict:
    """Replicate the sizing block of configure_workload.sh (lines 36-112).
    Returns the values that script would have exported, plus the topology
    it inferred — printed by the caller so users can sanity-check before
    a real run.

    Logic in plain terms: target 4-10 cores per vLLM instance, pick the
    largest BATCH_SIZE in {96,64,48,32} whose KV cache (per instance) fits
    in MEM_PER_NODE, force CORES_PER_INST to divide CORES_PER_NODE evenly,
    spawn NUM_INSTS = INSTS_PER_NUMA × NUM_NUMA_NODES.
    """
    num_cores = _physical_cores()
    num_numa = _detect_numa_nodes()
    num_l3 = _detect_l3_clusters()
    # MLPerf's NUM_NODES rule: prefer L3 clusters when they outnumber NUMA
    # nodes AND the user opted in. The instance count still derives from
    # NUMA at the end — the L3 count only sizes CORES_PER_INST.
    num_nodes = num_l3 if (use_l3_clusters and num_l3 > num_numa) else num_numa
    cores_per_node = max(1, num_cores // num_nodes)
    mem_avail = _available_mem_gib()
    mem_per_node = mem_avail // num_nodes if num_nodes else 0

    # CORES_PER_INST_MIN: pick the i in [4,10] that maximizes the cores we
    # can actually use ((cores_per_node // i) * i). Falls back to 4. This is
    # the floor we'll clamp to if the memory-driven pick goes below it.
    max_usable = 0
    cores_per_inst_min = _CORES_PER_INST_MIN
    for i in range(_CORES_PER_INST_MIN, _CORES_PER_INST_MAX + 1):
        usable = (cores_per_node // i) * i
        if usable > max_usable:
            max_usable = usable
            cores_per_inst_min = i

    # Try BATCH_SIZE largest-first; accept the first one whose KV cache fits.
    chosen_bs = _BATCH_SIZE_CANDIDATES[-1]
    chosen_cpi = _CORES_PER_INST_MAX + 1
    chosen_mem_cache = 0
    for bs in _BATCH_SIZE_CANDIDATES:
        mem_cache = (bs * _CACHE_BYTES_PER_BATCH) // (1024 ** 3)
        mem_total = _MEM_MODEL_GIB + mem_cache
        insts_per_node = mem_per_node // mem_total if mem_total > 0 else 0
        cpi = cores_per_node // insts_per_node if insts_per_node > 0 else 999
        if cpi <= _CORES_PER_INST_MAX:
            chosen_bs = bs
            chosen_cpi = cpi
            chosen_mem_cache = mem_cache
            break

    cpi = max(chosen_cpi, cores_per_inst_min)
    # Force divisibility: smallest i ≥ cpi (and ≤ 10) where cores_per_node%i==0.
    for i in range(cpi, _CORES_PER_INST_MAX + 1):
        if cores_per_node % i == 0:
            cpi = i
            break

    # NUM_INSTS uses NUM_NUMA_NODES, not NUM_NODES — matches the MLPerf
    # script: L3-cluster sizing for cache locality, NUMA-node binding for
    # the actual workers.
    insts_per_numa = max(1, num_cores // num_numa // cpi)
    num_instances = insts_per_numa * num_numa

    return {
        "num_instances": max(1, num_instances),
        "batch_size": chosen_bs,
        # Per-instance KV cache budget (GiB). Add a small headroom factor
        # of 1 GiB so vLLM's internal slack doesn't OOM the request floor.
        "cpu_kvcache_gib": max(8, chosen_mem_cache + 1),
        "cores_per_inst": cpi,
        "topology": {
            "num_cores": num_cores,
            "num_numa": num_numa,
            "num_l3_clusters": num_l3,
            "sizing_nodes": num_nodes,
            "cores_per_node": cores_per_node,
            "mem_available_gib": mem_avail,
            "mem_per_node_gib": mem_per_node,
        },
    }


def _print_autotune(tune: dict) -> None:
    t = tune["topology"]
    print("===== --auto-cpu (mirrors whisper_mlperf configure_workload.sh) =====")
    print(f"  cores (physical):        {t['num_cores']}")
    print(f"  NUMA nodes:              {t['num_numa']}")
    print(f"  L3 clusters:             {t['num_l3_clusters']}")
    print(f"  sizing nodes used:       {t['sizing_nodes']}  "
          f"({'L3 clusters' if t['sizing_nodes'] == t['num_l3_clusters'] and t['num_l3_clusters'] != t['num_numa'] else 'NUMA nodes'})")
    print(f"  cores per node:          {t['cores_per_node']}")
    print(f"  available memory (GiB):  {t['mem_available_gib']}")
    print(f"  mem per node (GiB):      {t['mem_per_node_gib']}")
    print("  -- chosen --")
    print(f"  num_instances:           {tune['num_instances']}")
    print(f"  batch_size:              {tune['batch_size']}")
    print(f"  cpu_kvcache_gib:         {tune['cpu_kvcache_gib']}  (per instance)")
    print(f"  cores_per_inst:          {tune['cores_per_inst']}")
    print()


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


# Common tcmalloc paths. The MLPerf SUT exports LD_PRELOAD globally; we
# instead set it inside the worker so the vLLM cpu_worker SUBPROCESS (which
# is what actually does GEMM) inherits the preload. Setting in the running
# process is too late for its own malloc, but env vars propagate to children.
_TCMALLOC_CANDIDATES = (
    "/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4",
    "/usr/lib64/libtcmalloc_minimal.so.4",
    "/usr/local/lib/libtcmalloc_minimal.so.4",
)


def _ensure_tcmalloc_preload() -> Optional[str]:
    """Add libtcmalloc_minimal.so.4 to LD_PRELOAD if available. Returns the
    path that was added (or already present), else None."""
    existing = os.environ.get("LD_PRELOAD", "")
    for p in _TCMALLOC_CANDIDATES:
        if p in existing:
            return p
        if Path(p).exists():
            os.environ["LD_PRELOAD"] = f"{p}:{existing}" if existing else p
            return p
    return None


# Top-level (picklable) worker for multi-instance mode. Must NOT close over
# state from benchmark(); everything comes through args.
def _instance_worker(instance_id: int, core_bind: str,
                     samples_shard: list[dict], llm_kwargs: dict,
                     mode: str, batch_size: int, max_new_tokens: int,
                     cpu_kvcache_gib: int, result_queue,
                     log_path: Optional[str] = None):
    # Redirect this worker's stdout+stderr to log_path BEFORE any vLLM import.
    # Uses os.dup2 on the underlying fd so any child process vLLM spawns
    # (engine-core subprocesses) inherits the redirect — that's where the
    # "engine core initialization failed" root-cause traceback lives.
    if log_path:
        import sys
        log_file = open(log_path, "w", buffering=1)
        os.dup2(log_file.fileno(), sys.stdout.fileno())
        os.dup2(log_file.fileno(), sys.stderr.fileno())
    # Preload tcmalloc BEFORE importing vLLM. The vLLM cpu_worker subprocess
    # (where the actual matmul runs) will inherit this and skip the
    # "libtcmalloc is not found in LD_PRELOAD" warning, gaining ~15-25%.
    tcmalloc = _ensure_tcmalloc_preload()
    if tcmalloc:
        print(f"[inst {instance_id}] LD_PRELOAD includes {tcmalloc}")
    else:
        print(f"[inst {instance_id}] tcmalloc not found at any expected path; "
              f"running with default allocator (slower)")
    # NUMA memory binding — matches whisper_mlperf/code/workspace/SUT.py:36+120.
    # Without it, model weight mmap and KV cache pages can land on whichever
    # NUMA node first touched them, causing cross-socket fetches per layer
    # on every decode step. Binding here, BEFORE vLLM is imported, ensures
    # all subsequent allocations (including the cpu_worker subprocess) are
    # NUMA-local. Derives node index from the worker's first pinned core,
    # assuming contiguous per-socket core layout (the typical Xeon topology).
    try:
        from numa import memory as _numa_memory
        first_core = int(core_bind.split("-")[0])
        cores_per_numa = max(1, _physical_cores() // _detect_numa_nodes())
        node_idx = first_core // cores_per_numa
        _numa_memory.set_membind_nodes(node_idx)
        print(f"[inst {instance_id}] NUMA membind → node {node_idx} "
              f"(first_core={first_core}, cores_per_numa={cores_per_numa})")
    except ImportError:
        print(f"[inst {instance_id}] py-libnuma not installed; "
              f"skipping NUMA membind (run `pip install py-libnuma` "
              f"for ~10-20% more throughput)")
    except Exception as e:
        print(f"[inst {instance_id}] NUMA membind failed: {e!r}; continuing")
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
                          "traceback": tb, "log_path": log_path})


def _balanced_shards(samples: list[dict], n_shards: int,
                     seed: int = 42) -> list[list[dict]]:
    """Deterministically shuffle samples, then round-robin into N shards.

    Why shuffle instead of sort-by-duration: audio length doesn't predict
    worker wall — output-token count does (a 30-s silent clip emits ~20
    tokens, a 5-s fast-speech clip emits ~50). And token count per sample
    isn't known a priori. A shuffle is the right tool: it neutralizes any
    upstream clustering (LibriSpeech, for example, often groups consecutive
    rows by speaker), and with ~200+ samples per shard the law of large
    numbers brings the actual work-per-shard variance to ~5-8%.

    Fixed seed so re-runs are diffable. Pass a different seed if you want
    to vary the partition without changing the input file.
    """
    import random
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    shards: list[list[dict]] = [[] for _ in range(n_shards)]
    for i, s in enumerate(shuffled):
        shards[i % n_shards].append(s)
    return shards


def _run_multi_instance(num_instances: int, samples: list[dict],
                        llm_kwargs: dict, mode: str, batch_size: int,
                        max_new_tokens: int,
                        numa_bind: Optional[str],
                        cpu_kvcache_gib: int,
                        spawn_stagger_s: float = 0.0,
                        ) -> tuple[list[SampleRow], float]:
    """Spawn `num_instances` vLLM workers, one per pinned core range, and
    shard the sample list round-robin across them.

    The returned wall is `max(worker_wall_s)` — the slowest worker's
    processing time from "engine loaded" to "last sample done". Engine load
    is excluded so throughput is comparable to MLPerf, whose loadgen timer
    starts only after all SUT workers signal alive.
    """
    import multiprocessing as mp

    bindings = _compute_core_bindings(num_instances, override_range=numa_bind)
    # Shuffle then round-robin so any upstream clustering in the dataset
    # (LibriSpeech often groups by speaker) gets dissolved. With 200+
    # samples per shard, statistical variance in total work is ~5-8% —
    # which is what actually matters, since output-token count drives
    # worker wall and isn't predictable from audio duration anyway.
    shards = _balanced_shards(samples, num_instances)

    print(f"[multi] num_instances={num_instances} bindings={bindings}")
    print(f"[multi] shard sizes: {[len(s) for s in shards]} "
          f"(shuffled + round-robin)")

    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    procs = []
    spawn_t0 = time.perf_counter()
    if spawn_stagger_s > 0:
        print(f"[multi] spawn_stagger_s={spawn_stagger_s:.1f}s "
              f"(workers start {spawn_stagger_s:.1f}s apart to avoid "
              f"engine-core init contention)")
    for i in range(num_instances):
        # log_path=None: worker doesn't dup2 stdout/stderr — its output
        # goes to the parent process's stdout via mp.spawn pipes. Output
        # from 24+ workers will interleave; pipe stdout through `tee` or
        # `>file` at invocation time if you need a clean transcript.
        p = ctx.Process(
            target=_instance_worker,
            args=(i, bindings[i], shards[i], llm_kwargs,
                  mode, batch_size, max_new_tokens, cpu_kvcache_gib, result_q,
                  None),
        )
        p.start()
        procs.append(p)
        # Stagger only between spawns, not after the last one.
        if spawn_stagger_s > 0 and i < num_instances - 1:
            time.sleep(spawn_stagger_s)

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

    parent_wall = time.perf_counter() - spawn_t0

    if errors:
        for e in errors:
            print(f"\n[inst {e['instance_id']}] FAILED:")
            print(e["traceback"])
        raise RuntimeError(
            f"{len(errors)}/{num_instances} instances failed; see Python "
            f"tracebacks above. For engine-core (subprocess) stderr, re-run "
            f"with stdout/stderr captured to a file via `2>&1 | tee run.log`."
        )

    # MLPerf-comparable wall: longest worker's *processing* time, init excluded.
    # _instance_worker measures worker_wall_s from AFTER _build_llm() returns,
    # so it captures only the dispatch loop — matching MLPerf loadgen's window.
    worker_walls = [r["worker_wall_s"] for r in results]
    bottleneck_wall = max(worker_walls) if worker_walls else parent_wall
    load_overhead = parent_wall - bottleneck_wall
    print(f"[multi] parent wall {parent_wall:.1f}s "
          f"(spawn+load: {load_overhead:.1f}s) "
          f"| bottleneck worker processing wall {bottleneck_wall:.1f}s")
    print(f"[multi] worker walls: min={min(worker_walls):.1f}s "
          f"median={sorted(worker_walls)[len(worker_walls)//2]:.1f}s "
          f"max={bottleneck_wall:.1f}s")

    # Reassemble by sample_id → original input order (stable for CSV diffs).
    sid_to_index = {s["sample_id"]: i for i, s in enumerate(samples)}
    all_rows: list[SampleRow] = []
    for r in results:
        all_rows.extend(r["rows"])
    all_rows.sort(key=lambda row: sid_to_index.get(row.sample_id, 1 << 30))

    return all_rows, bottleneck_wall


def _summarize(rows: list[SampleRow], wall_s: float, *,
               model: str, precision: str, dataset: str,
               mode: str, batch_size: int) -> Summary:
    ts = _dt.datetime.now().isoformat(timespec="seconds")
    if not rows:
        return Summary(ts, model, precision, dataset, mode, batch_size, 0,
                       0.0, wall_s, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0)
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
        p99_ttft_ms=float(np.percentile(ttfts, 99)),
        avg_tpot_ms=float(tpots.mean()),
        p50_tpot_ms=float(np.percentile(tpots, 50)),
        p95_tpot_ms=float(np.percentile(tpots, 95)),
        p99_tpot_ms=float(np.percentile(tpots, 99)),
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


# One canonical cross-run summary file. Columns are Summary's fields in order —
# already match the requested schema:
#   timestamp,model,precision,dataset,mode,batch_size,n_samples,
#   total_audio_s,total_wall_s,throughput_xrt,throughput_tokens_per_s,
#   avg_ttft_ms,p50_ttft_ms,p95_ttft_ms,p99_ttft_ms,
#   avg_tpot_ms,p50_tpot_ms,p95_tpot_ms,p99_tpot_ms
SUMMARY_CSV_PATH = OUTPUT_DIR / "summary.csv"


def _append_summary_row(summary: Summary, path: Path = SUMMARY_CSV_PATH) -> None:
    """Append one row to the cross-run summary CSV. Writes the header on
    first use. Every single/batch/offline run adds exactly one line per
    dataset processed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sfields = [f.name for f in fields(Summary)]
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sfields)
        if write_header:
            w.writeheader()
        w.writerow({n: getattr(summary, n) for n in sfields})


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
        ("TTFT avg / p50 / p95 / p99 (ms)",
         f"{s.avg_ttft_ms:.1f} / {s.p50_ttft_ms:.1f} / "
         f"{s.p95_ttft_ms:.1f} / {s.p99_ttft_ms:.1f}"),
        ("TPOT avg / p50 / p95 / p99 (ms)",
         f"{s.avg_tpot_ms:.2f} / {s.p50_tpot_ms:.2f} / "
         f"{s.p95_tpot_ms:.2f} / {s.p99_tpot_ms:.2f}"),
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
              output_csv: Optional[str] = None,
              num_instances: int,
              cpu_kvcache_gib: int,
              spawn_stagger_s: float = 0.0) -> list[Path]:
    model_path = _resolve_model_path(model_arg)
    precision = (precision or _infer_precision(model_path)
                 or _DTYPE_TO_TAG.get(dtype, dtype))
    model_slug = model_path.name
    resolved_device = _resolve_device(device)

    # max_num_batched_tokens caps the per-step prefill+decode budget. For
    # Whisper (encoder-decoder), vLLM v1 forcibly disables chunked MM input
    # (scheduler.py __post_init__), so the encoder's 1500 mel tokens must fit
    # in ONE step — values below ~1500 fail at engine init with
    # "max_tokens_per_mm_item (1500) is larger than max_num_batched_tokens".
    # MLPerf SUT's 800 worked on an older vLLM that pre-dated this check.
    # Default 2048 = 1500 encoder + decoder headroom.

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
            # If --numa-bind was given, also bind memory to that NUMA node so
            # the single-instance run is genuinely NUMA-local. Without this we
            # bind threads but leave memory first-touch — half the model can
            # land on the wrong node. Mirrors the multi-instance membind.
            try:
                from numa import memory as _numa_memory
                first_core = int(numa_bind.split("-")[0])
                cores_per_numa = max(1, _physical_cores() // _detect_numa_nodes())
                node_idx = first_core // cores_per_numa
                _numa_memory.set_membind_nodes(node_idx)
                print(f"[cpu] NUMA membind → node {node_idx} "
                      f"(first_core={first_core}, "
                      f"cores_per_numa={cores_per_numa})")
            except ImportError:
                print("[cpu] py-libnuma not installed; --numa-bind set "
                      "threads but not memory. `pip install py-libnuma` "
                      "for the missing locality bit.")
            except Exception as e:
                print(f"[cpu] NUMA membind failed: {e!r}; continuing "
                      f"with thread-only binding")
        # Cap CPU KV cache size to avoid vLLM auto-sizing from total RAM
        # and OOMing (see _instance_worker for the multi-instance equivalent).
        if resolved_device == "cpu":
            os.environ["VLLM_CPU_KVCACHE_SPACE"] = str(cpu_kvcache_gib)
            print(f"[cpu] VLLM_CPU_KVCACHE_SPACE={cpu_kvcache_gib} GiB")
            tcmalloc = _ensure_tcmalloc_preload()
            if tcmalloc:
                print(f"[cpu] LD_PRELOAD includes {tcmalloc}")
            else:
                print(f"[cpu] tcmalloc not found; running with default "
                      f"allocator (slower)")
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
                spawn_stagger_s=spawn_stagger_s,
            )

        summary = _summarize(rows, wall, model=model_slug, precision=precision,
                             dataset=ds, mode=mode, batch_size=batch_size)
        _print_summary(summary)

        # Always append one row to the cross-run summary CSV — the canonical
        # comparison file. Per-run detail CSV is written only for offline mode
        # (the full benchmark output) or when --output-csv is explicit.
        _append_summary_row(summary)

        write_detail = (output_csv is not None) or (mode == "offline")
        if write_detail:
            stamp = _dt.datetime.now(EAST).strftime("%Y%m%d_%H%M")
            mode_tag = f"batch{batch_size}" if mode == "batch" else mode
            if num_instances > 1:
                mode_tag = f"{mode_tag}_inst{num_instances}"
            if output_csv is not None:
                base = Path(output_csv)
                if len(datasets) > 1:
                    out_path = base.with_name(f"{base.stem}_{ds}{base.suffix}")
                else:
                    out_path = base
            else:
                out_path = OUTPUT_DIR / (
                    f"{model_slug}_{precision}_{ds}_{mode_tag}_{stamp}.csv"
                )
            _write_csv(rows, summary, out_path)
            print(f"wrote {out_path}")
            written.append(out_path)
        else:
            print(f"summary row appended to {SUMMARY_CSV_PATH} "
                  f"(per-run detail CSV skipped for mode={mode})")
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
    p.add_argument("--datasets", nargs="+",
                   choices=["librispeech", "tedlium", "mlperf"],
                   default=["librispeech", "tedlium"],
                   help="Datasets to run. 'mlperf' = the dev-all-repack "
                        "manifest shipped in whisper_mlperf/data/ (LibriSpeech "
                        "dev clips pre-concatenated to ~30s). Use this for "
                        "apples-to-apples tok/s vs MLPerf's published number.")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap samples per dataset (default: full split)")
    p.add_argument("--max-new-tokens", type=int, default=200,
                   help="Per-request output cap (MLPerf SUT: 200)")
    p.add_argument("--max-num-seqs", type=int, default=64,
                   help="vLLM scheduler concurrent-request cap (MLPerf SUT: 64)")
    p.add_argument("--max-model-len", type=int, default=448,
                   help="vLLM max context length (MLPerf SUT: 448 = Whisper max)")
    p.add_argument("--max-num-batched-tokens", type=int, default=2048,
                   help="Per-step token budget for the vLLM scheduler. Must be "
                        ">= 1500 (Whisper encoder length) on vLLM v1, which "
                        "forcibly disables chunked MM input for encoder-decoder "
                        "models. MLPerf SUT's 800 fails at engine init on v1. "
                        "Default 2048 = 1500 + decoder headroom.")
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
    p.add_argument("--batch-size", type=int, default=None,
                   help="Concurrency for --mode batch (default: 8; --auto-cpu picks "
                        "from {96,64,48,32} based on host memory)")
    p.add_argument("--numa-bind", default=None,
                   help="Optional core range for VLLM_CPU_OMP_THREADS_BIND, "
                        "e.g. '0-31'. Mirrors MLPerf SUT's per-instance pinning. "
                        "Single-socket: pin to that socket's cores to avoid "
                        "cross-socket memory traffic.")
    p.add_argument("--cpu-kvcache-gib", type=int, default=None,
                   help="Per-instance VLLM_CPU_KVCACHE_SPACE in GiB "
                        "(default: 30; --auto-cpu sizes from chosen batch_size). "
                        "vLLM CPU autosizes from total available RAM, which on "
                        "big-memory boxes overshoots (e.g. ~1 TB per instance, "
                        "OOMs in multi-instance mode).")
    p.add_argument("--num-instances", type=int, default=None,
                   help="Spawn N independent vLLM engines, each pinned to a "
                        "contiguous slice of physical cores (default: 1; "
                        "--auto-cpu sizes from host topology, mirroring MLPerf "
                        "configure_workload.sh).")
    p.add_argument("--auto-cpu", action="store_true",
                   help="Auto-pick num_instances, batch_size, and cpu_kvcache_gib "
                        "from host topology + available memory. Replicates "
                        "whisper_mlperf/code/workspace/configure_workload.sh. "
                        "Explicit values for --num-instances / --batch-size / "
                        "--cpu-kvcache-gib still win. Only meaningful on CPU.")
    p.add_argument("--use-l3-clusters", action="store_true",
                   help="With --auto-cpu, count L3 cache clusters instead of "
                        "NUMA nodes when sizing cores-per-instance. Helps on "
                        "Granite Rapids / SNC parts where one NUMA node spans "
                        "multiple L3 tiles. Final worker binding is still "
                        "per-NUMA. Default: NUMA only.")
    p.add_argument("--spawn-stagger-s", type=float, default=None,
                   help="Seconds to sleep between starting each multi-instance "
                        "worker. Avoids engine-core init contention (shm, ZMQ "
                        "handshakes, model mmap) when 10+ workers spawn at "
                        "once. Default: 0 for 1-4 instances, 5s for >4. Set "
                        "explicitly to override.")
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
    p.add_argument("--output-csv", default=None,
                   help="Override CSV output path. With one --datasets entry, "
                        "writes exactly there. With multiple, appends _{ds} "
                        "before the extension. Default: auto-named under "
                        "output/vllm/{model}_{precision}_{dataset}_{mode_tag}_"
                        "{YYYYMMDD_HHMM}.csv")
    args = p.parse_args()

    # Resolve autotunable args. Order: explicit CLI > --auto-cpu pick > built-in
    # default. --auto-cpu only fires on CPU runs; warn and fall back otherwise.
    auto = None
    if args.auto_cpu:
        if args.device != "cpu":
            print(f"[auto-cpu] --device={args.device}, skipping autotune "
                  f"(only meaningful on CPU)")
        else:
            auto = _autotune_cpu(use_l3_clusters=args.use_l3_clusters)
            _print_autotune(auto)
    num_instances = (
        args.num_instances if args.num_instances is not None
        else (auto["num_instances"] if auto else 1)
    )
    batch_size = (
        args.batch_size if args.batch_size is not None
        else (auto["batch_size"] if auto else 8)
    )
    cpu_kvcache_gib = (
        args.cpu_kvcache_gib if args.cpu_kvcache_gib is not None
        else (auto["cpu_kvcache_gib"] if auto else 30)
    )
    # Default spawn stagger: 0 for small N, 5s for many workers. Avoids
    # engine-core init contention when >4 workers spawn simultaneously.
    spawn_stagger_s = (
        args.spawn_stagger_s if args.spawn_stagger_s is not None
        else (5.0 if num_instances > 4 else 0.0)
    )

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
        batch_size=batch_size,
        streaming=not args.no_streaming,
        numa_bind=args.numa_bind,
        enforce_eager=args.enforce_eager,
        num_instances=num_instances,
        cpu_kvcache_gib=cpu_kvcache_gib,
        spawn_stagger_s=spawn_stagger_s,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
