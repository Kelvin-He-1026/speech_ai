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
from zoneinfo import ZoneInfo

import numpy as np
import psutil

_HERE = Path(__file__).resolve().parent
DATA_DIR = _HERE / "data"
OUTPUT_ROOT = _HERE / "output"
PYTORCH_OUT = OUTPUT_ROOT / "pytorch"
OPENVINO_OUT = OUTPUT_ROOT / "openvino"
HF_CACHE_DIR = DATA_DIR / "hf_datasets"

DATA_DIR.mkdir(parents=True, exist_ok=True)
PYTORCH_OUT.mkdir(parents=True, exist_ok=True)
OPENVINO_OUT.mkdir(parents=True, exist_ok=True)
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HOME", str(DATA_DIR / "hf_home"))

EAST = ZoneInfo("America/New_York")  # matches vllm_speech_benchmark.py


# Silence "Both `max_new_tokens` (=440) and `max_length`(=20) seem to have been set."
# from transformers.generation.utils._prepare_generated_length. We own length-control
# on the gen_config in _run_one_dataset (max_length=None, max_new_tokens=440) and do
# not pass max_new_tokens as a generate() kwarg, which removes the underlying conflict.
# The filter is the belt-and-suspenders: Whisper long-form internally calls
# super().generate() per 30-s segment with a freshly-built GenerationConfig() that can
# re-introduce the default max_length=20, re-firing the warning. Filter is installed at
# module-import time so it covers every code path that imports this script — including
# the warmup generate(), batched offline call, and any sub-process spawned for sharding.
import logging as _logging  # noqa: E402
class _MaxLengthWarnFilter(_logging.Filter):  # noqa: E302
    def filter(self, record):
        return "Both `max_new_tokens`" not in record.getMessage()
for _name in ("transformers", "transformers.generation", "transformers.generation.utils"):
    _logging.getLogger(_name).addFilter(_MaxLengthWarnFilter())

_DTYPE_TO_TAG = {
    "bfloat16": "bf16",
    "float16": "fp16",
    "float32": "fp32",
    "auto": "auto",
}


def _infer_precision_from_name(name: str) -> Optional[str]:
    n = name.lower()
    for tag in ("w8a8", "int8", "int4", "fp16", "bf16", "fp32"):
        if tag in n:
            return tag
    return None


def _dtype_arg_to_tag(dtype: Optional[str]) -> Optional[str]:
    if not dtype:
        return None
    return _DTYPE_TO_TAG.get(dtype, dtype)


def _torch_dtype_to_tag(model_dtype) -> Optional[str]:
    if model_dtype is None:
        return None
    s = str(model_dtype).replace("torch.", "")
    return _DTYPE_TO_TAG.get(s, s)

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
    tpot_seconds: float = 0.0
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
    mode: str
    batch_size: int
    device: str
    resolved_device: str
    dtype: str
    n_samples: int
    total_audio_s: float
    total_wall_s: float
    throughput_xrt: float
    total_new_tokens: int
    tok_per_s: float
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
    """Counts decode steps and records TTFT (works for batched generate too).

    For single-stream generate, `n_steps` equals tokens emitted for that one
    sequence. For batched generate, `n_steps` is the number of decode steps
    (= max tokens across the batch); per-sample token counts come from the
    final output tensor.
    """
    from transformers.generation.streamers import BaseStreamer

    class TimingStreamer(BaseStreamer):
        def __init__(self):
            self.first_token_time: Optional[float] = None
            self.n_steps: int = 0
            self.start: float = 0.0

        def begin(self, t0: float):
            self.start = t0
            self.first_token_time = None
            self.n_steps = 0

        def put(self, value):
            if self.first_token_time is None:
                self.first_token_time = time.perf_counter() - self.start
            self.n_steps += 1

        def end(self):
            pass

    return TimingStreamer()


def _chunk_iter(iterable, n: Optional[int]):
    """Yield lists of up to `n` items. n=None → one chunk containing everything."""
    if n is None:
        yield list(iterable)
        return
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= n:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


_WHISPER_PROMPT_LEN = 4  # SOT + lang + task + <|notimestamps|>


def _per_sample_new_tokens(out, prompt_len: int, pad_or_eos_id) -> list[int]:
    """For batched generate, count tokens generated per sample (excluding the
    forced decoder prompt and any trailing pad/eos repetition)."""
    counts = []
    for row in out:
        gen = row[prompt_len:]
        if pad_or_eos_id is not None:
            mask = (gen != pad_or_eos_id)
            n = int(mask.sum().item())
            # If the sequence ended with EOS, include that one EOS token
            if n < gen.size(0):
                n += 1
        else:
            n = int(gen.size(0))
        counts.append(max(1, n))
    return counts


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
    # "chime6": _iter_chime6,
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
    mode: str = "single", batch_size: int = 1,
) -> tuple[BenchReport, list]:
    import jiwer
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    spelling = getattr(processor.tokenizer, "english_spelling_normalizer", {})
    normalizer = EnglishTextNormalizer(spelling)

    gen_kwargs = dict(
        language=language,
        task="transcribe",
        return_timestamps=False,
    )

    # Length control: own it on the gen_config side, with NO max_new_tokens in
    # kwargs. transformers warns "Both `max_new_tokens` and `max_length` seem to
    # have been set" when both paths supply a value; setting max_length=None and
    # max_new_tokens on the gen_config eliminates the conflict. The accompanying
    # logger filter is installed at module-import time (see top of file).
    gc = getattr(model, "generation_config", None)
    if gc is not None:
        gc.max_length = None
        gc.max_new_tokens = 440

    if not is_ov and torch_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    if mode == "single":
        chunk_size = 1
    elif mode == "batch":
        chunk_size = max(1, batch_size)
    elif mode == "offline":
        chunk_size = None  # one big chunk = all samples
    else:
        raise ValueError(f"unknown mode: {mode}")

    pad_id = getattr(model.generation_config, "pad_token_id", None)
    if pad_id is None:
        eos = getattr(model.generation_config, "eos_token_id", None)
        pad_id = eos[0] if isinstance(eos, list) else eos

    streamer = _make_streamer()
    results: list[SampleResult] = []
    total_audio_s = 0.0
    total_wall_s = 0.0

    # One-shot warmup with silence so kernel JIT/autotune, allocator growth,
    # attention backend selection, and tokenizer/normalizer first-touch don't
    # get charged to the first timed chunk (which otherwise looks ~10x slower
    # than steady state). Sized to chunk_size so batched kernels autotune at
    # the right shape; offline mode falls back to 1.
    warmup_n = chunk_size if chunk_size is not None else 1
    warmup_audios = [np.zeros(16000, dtype=np.float32) for _ in range(warmup_n)]
    warmup_inputs = processor(
        warmup_audios, sampling_rate=16000, return_tensors="pt",
        return_attention_mask=True,
    )
    warmup_features = warmup_inputs.input_features
    warmup_mask = warmup_inputs.get("attention_mask")
    if not is_ov:
        warmup_features = warmup_features.to(torch_device)
        if model_dtype is not None and warmup_features.dtype != model_dtype:
            warmup_features = warmup_features.to(model_dtype)
        if warmup_mask is not None:
            warmup_mask = warmup_mask.to(torch_device)
    warmup_kwargs = dict(gen_kwargs)
    if warmup_mask is not None:
        warmup_kwargs["attention_mask"] = warmup_mask
    print(f"[warmup] running 1 untimed generate() at batch={warmup_n}")
    with torch.inference_mode():
        _ = model.generate(input_features=warmup_features, **warmup_kwargs)
    if not is_ov and torch_device.type == "cuda":
        torch.cuda.synchronize()

    with MemoryMonitor() as mon:
        for chunk in _chunk_iter(sample_iter, chunk_size):
            ids = [c[0] for c in chunk]
            audios = [_resample(c[1], c[2], 16000) for c in chunk]
            refs = [c[3] for c in chunk]
            audio_seconds_each = [len(a) / 16000.0 for a in audios]

            # Single-element list still produces a [1, ...] batched tensor.
            # return_attention_mask=True makes the feature extractor emit a mask
            # over the (zero-padded) mel frames — required to silence the
            # "pad_token == eos_token, attention mask not set" warning during
            # decoder generation and to get reliable batched results.
            inputs = processor(
                audios, sampling_rate=16000, return_tensors="pt",
                return_attention_mask=True,
            )
            input_features = inputs.input_features
            attention_mask = inputs.get("attention_mask")
            if not is_ov:
                input_features = input_features.to(torch_device)
                if model_dtype is not None and input_features.dtype != model_dtype:
                    input_features = input_features.to(model_dtype)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(torch_device)

            gen_call_kwargs = dict(gen_kwargs)
            if attention_mask is not None:
                gen_call_kwargs["attention_mask"] = attention_mask

            t0 = time.perf_counter()
            streamer.begin(t0)
            with torch.inference_mode():
                out = model.generate(
                    input_features=input_features,
                    streamer=streamer,
                    **gen_call_kwargs,
                )
            if not is_ov and torch_device.type == "cuda":
                torch.cuda.synchronize()
            total = time.perf_counter() - t0

            hyps = processor.batch_decode(out, skip_special_tokens=True)
            if out.dim() == 2 and out.size(0) > 1:
                per_sample_tokens = _per_sample_new_tokens(out, _WHISPER_PROMPT_LEN, pad_id)
            else:
                per_sample_tokens = [streamer.n_steps]

            ttft = streamer.first_token_time if streamer.first_token_time is not None else total
            for i, sid in enumerate(ids):
                results.append(SampleResult(
                    sample_id=sid,
                    audio_seconds=audio_seconds_each[i],
                    total_seconds=total,  # chunk wall — shared across batch
                    ttft_seconds=ttft,    # shared across batch
                    n_new_tokens=per_sample_tokens[i] if i < len(per_sample_tokens) else streamer.n_steps,
                    reference=refs[i],
                    hypothesis=hyps[i],
                ))
                total_audio_s += audio_seconds_each[i]
            total_wall_s += total

    for r in results:
        # Same formula used for the avg_tpot_ms aggregate. In batch mode the
        # numerator is shared across the chunk, so per-sample TPOT mainly
        # varies with how many tokens each sample actually emitted.
        r.tpot_seconds = (
            (r.total_seconds - r.ttft_seconds) / (r.n_new_tokens - 1)
            if r.n_new_tokens > 1 else 0.0
        )
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

    report = BenchReport(
        timestamp=_dt.datetime.now().isoformat(timespec="seconds"),
        model=model_name,
        dataset=dataset,
        mode=mode,
        batch_size=batch_size,
        device=device,
        resolved_device=resolved_device,
        dtype=effective_dtype,
        n_samples=len(results),
        total_audio_s=total_audio_s,
        total_wall_s=total_wall_s,
        throughput_xrt=total_audio_s / max(1e-9, total_wall_s),
        total_new_tokens=sum(r.n_new_tokens for r in results),
        tok_per_s=sum(r.n_new_tokens for r in results) / max(1e-9, total_wall_s),
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
    return report, results


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
    mode: str = "single",
    batch_size: int = 1,
    sort_by_length: bool = False,
    output_csv: Optional[str] = None,
    num_shards: int = 1,
    shard_idx: int = 0,
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

    # mode_tag stays a single underscore-free token so consolidate_outputs.py's
    # rpartition('_') parser keeps working. Hyphens are used to compose multi-part
    # tags. Batch size is shown explicitly for every mode:
    #   single        → single-b1            (one sample per generate)
    #   batch  -B=N   → batch-bN             (N samples per generate)
    #   offline       → offline-bAll         (all samples in one generate call)
    if mode == "batch":
        mode_tag = f"batch-b{batch_size}"
    elif mode == "single":
        mode_tag = "single-b1"
    elif mode == "offline":
        mode_tag = "offline-bAll"
    else:
        mode_tag = mode
    if sort_by_length:
        mode_tag += "-sorted"
    if num_shards > 1:
        mode_tag += f"-shard{shard_idx}of{num_shards}"
    if mode == "batch":
        print(f"[mode] batch with batch_size={batch_size} "
              f"→ output tag '{mode_tag}'")
    if num_shards > 1:
        print(f"[shard] this process handles every {num_shards}th sample starting at index {shard_idx} "
              f"→ output tag '{mode_tag}'")
    precision = (
        _infer_precision_from_name(model_name)
        or _dtype_arg_to_tag(dtype)
        or _torch_dtype_to_tag(model_dtype)
        or "ov-native"
    )
    out_dir = _output_dir_for(is_ov)

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

        if num_shards > 1:
            # Modulo shard on the GLOBAL stream index. Each shard sees every Nth
            # sample; the union of all shards equals the full split. In streaming
            # mode this still pulls every row over the wire and discards 1-of-N
            # — wasted bandwidth but trivial vs compute. Sharding happens before
            # sort_by_length so each shard sorts only its own subset.
            def _shard(it, ns=num_shards, si=shard_idx):
                for i, ex in enumerate(it):
                    if i % ns == si:
                        yield ex
            sample_iter = _shard(sample_iter)

        if sort_by_length:
            # Materialize the stream and sort by ascending audio duration so
            # each batch holds similarly-sized clips, reducing the "short waits
            # on long" waste in batched generate(). Cost: keeps the decoded
            # waveforms in memory (LibriSpeech test.clean ~75 MB).
            sample_iter = sorted(sample_iter, key=lambda s: len(s[1]) / max(1, s[2]))
            durs = [len(s[1]) / max(1, s[2]) for s in sample_iter]
            print(f"[sort] {len(sample_iter)} samples by ascending duration "
                  f"(min={min(durs):.2f}s, max={max(durs):.2f}s)")

        print(f"\n=== {_slug(model_name)} | {precision} | {ds} | {mode_tag} ===")
        report, samples = _run_one_dataset(
            model=model, processor=processor,
            dataset=ds, sample_iter=sample_iter,
            model_name=model_name, device=device, resolved_device=resolved_device,
            is_ov=is_ov, torch=torch, torch_device=torch_device, model_dtype=model_dtype,
            dtype_arg=dtype, language=language,
            mode=mode, batch_size=batch_size,
        )
        _print_report(report)

        stamp = _dt.datetime.now(EAST).strftime("%Y%m%d_%H%M")
        out_path = (Path(output_csv) if output_csv
                    else _csv_path(out_dir, model_name, precision, ds, mode_tag, stamp))
        _write_combined_csv(report, samples, out_path)
        print(f"wrote {out_path}")
        reports.append(report)

        # Also append a one-row entry to output/{backend}/box_summary.csv for
        # mode in {single, batch} non-sharded runs. Sharded runs are aggregated
        # via aggregate_shards.py; offline runs are excluded by request — their
        # per-sample latency stats are meaningless (one giant call shares wall
        # time across the whole batch) and they distort the cross-run table.
        if mode in ("single", "batch") and num_shards == 1:
            _append_box_summary_row(
                out_dir / "box_summary.csv",
                _box_row_from_report(report, samples, batch_size),
            )

    return reports


def _slug(s: str) -> str:
    return s.replace("/", "__").replace(" ", "_")


def _output_dir_for(is_ov: bool) -> Path:
    """PyTorch runs → output/pytorch/, OpenVINO runs → output/openvino/."""
    return OPENVINO_OUT if is_ov else PYTORCH_OUT


def _csv_path(out_dir: Path, model_name: str, precision: str,
              dataset: str, mode_tag: str, stamp: str) -> Path:
    """Match vllm_speech_benchmark.py naming:
        {model}_{precision}_{dataset}_{mode_tag}_{YYYYMMDD_HHMM}.csv"""
    return out_dir / f"{_slug(model_name)}_{precision}_{dataset}_{mode_tag}_{stamp}.csv"


def _write_combined_csv(report: BenchReport, samples: list, path: Path) -> None:
    """One CSV per (model, precision, dataset, mode): summary block on top,
    per-sample rows below. Same layout as the vLLM benchmark output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    summary_fields = [f.name for f in fields(BenchReport)]
    sample_fields = [f.name for f in fields(SampleResult)]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# summary"])
        w.writerow(summary_fields)
        w.writerow([getattr(report, h) for h in summary_fields])
        w.writerow([])
        w.writerow(["# per-sample"])
        w.writerow(sample_fields)
        for r in samples:
            w.writerow([getattr(r, h) for h in sample_fields])


# Field order is locked to aggregate_shards.py so single/batch runs and sharded
# runs share one box_summary.csv schema. Keep these two lists in sync.
_BOX_SUMMARY_FIELDS = [
    "model", "dtype", "dataset", "base_mode", "batch_size", "n_shards",
    "n_samples", "total_audio_s", "max_wall_s", "box_xrt", "box_tok_per_s",
    "wer", "ref_words", "substitutions", "deletions", "insertions",
    "p50_total_ms", "p95_total_ms", "p50_ttft_ms", "p95_ttft_ms",
]


def _percentile(xs, p):
    if not xs:
        return 0.0
    xs_sorted = sorted(xs)
    idx = max(0, min(len(xs_sorted) - 1, int(p * len(xs_sorted)) - 1))
    return xs_sorted[idx]


def _box_row_from_report(report: BenchReport, samples: list, batch_size: int) -> dict:
    """Build a one-row dict for box_summary.csv from a single non-sharded run."""
    import statistics
    totals = [s.total_seconds for s in samples if s.total_seconds > 0]
    ttfts  = [s.ttft_seconds  for s in samples if s.ttft_seconds  > 0]
    p50_total = statistics.median(totals) if totals else 0.0
    p95_total = _percentile(totals, 0.95)
    p50_ttft  = statistics.median(ttfts)  if ttfts  else 0.0
    p95_ttft  = _percentile(ttfts, 0.95)

    if report.mode == "single":
        batch_size_eff = "1"
    elif report.mode == "batch":
        batch_size_eff = str(batch_size)
    else:  # offline path is excluded at the call site, but stay safe
        batch_size_eff = "all"

    return {
        "model":          report.model,
        "dtype":          report.dtype,
        "dataset":        report.dataset,
        "base_mode":      report.mode,
        "batch_size":     batch_size_eff,
        "n_shards":       1,
        "n_samples":      report.n_samples,
        "total_audio_s":  round(report.total_audio_s, 3),
        "max_wall_s":     round(report.total_wall_s, 3),
        "box_xrt":        round(report.throughput_xrt, 4),
        "box_tok_per_s":  round(report.tok_per_s, 3),
        "wer":            round(report.wer, 6),
        "ref_words":      sum(s.n_ref_words   for s in samples),
        "substitutions":  sum(s.substitutions for s in samples),
        "deletions":      sum(s.deletions     for s in samples),
        "insertions":     sum(s.insertions    for s in samples),
        "p50_total_ms":   round(p50_total * 1000, 2),
        "p95_total_ms":   round(p95_total * 1000, 2),
        "p50_ttft_ms":    round(p50_ttft  * 1000, 2),
        "p95_ttft_ms":    round(p95_ttft  * 1000, 2),
    }


def _append_box_summary_row(box_csv: Path, row: dict) -> None:
    """Append one row, writing the header on first use. Field set is fixed to
    _BOX_SUMMARY_FIELDS (the same as aggregate_shards.py) so both code paths
    produce a consistent file."""
    box_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not box_csv.exists() or box_csv.stat().st_size == 0
    with box_csv.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_BOX_SUMMARY_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)


def _print_report(r: BenchReport) -> None:
    rows = [
        ("model", r.model),
        ("dataset", r.dataset),
        ("mode", f"{r.mode} (batch_size={r.batch_size})" if r.mode == "batch" else r.mode),
        ("device", f"{r.device} ({r.resolved_device})"),
        ("dtype", r.dtype),
        ("samples", str(r.n_samples)),
        ("audio (s)", f"{r.total_audio_s:.1f}"),
        ("wall (s)", f"{r.total_wall_s:.1f}"),
        ("throughput (xRT)", f"{r.throughput_xrt:.2f}"),
        ("generated tokens", str(r.total_new_tokens)),
        ("throughput (tok/s)", f"{r.tok_per_s:.2f}"),
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
    p.add_argument("--mode", default="single",
                   choices=["single", "batch", "offline"],
                   help="single = 1 audio at a time through generate (default); "
                        "batch  = --batch-size audios per fused generate call; "
                        "offline = stack ALL samples into one generate call "
                        "(can OOM — use with --max-samples).")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Audios per fused generate call when --mode batch (default: 8)")
    p.add_argument("--sort-by-length", action="store_true",
                   help="Materialize the dataset and sort by ascending audio "
                        "duration before batching so each batch holds "
                        "similarly-sized clips (reduces 'short waits on long' "
                        "waste in batch mode). Tag '_sorted' is appended to "
                        "the output filename.")
    p.add_argument("--output-csv", default=None,
                   help="Override CSV path for ALL datasets (default per-dataset: "
                        "output/{pytorch|openvino}/"
                        "{model}_{precision}_{dataset}_{mode_tag}_{YYYYMMDD_HHMM}.csv)")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Total number of shards. Pair with --shard-idx and launch one "
                        "process per socket under numactl to scale throughput across NUMA "
                        "nodes. Each shard takes every Nth sample. After both finish, "
                        "use aggregate_shards.py to compute box-level metrics.")
    p.add_argument("--shard-idx", type=int, default=0,
                   help="This process's shard index in [0, num_shards). Ignored when num-shards=1.")
    args = p.parse_args()

    if args.num_shards < 1 or not (0 <= args.shard_idx < args.num_shards):
        raise SystemExit(
            f"--shard-idx ({args.shard_idx}) must be in [0, --num-shards={args.num_shards})"
        )

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
        mode=args.mode,
        batch_size=args.batch_size,
        sort_by_length=args.sort_by_length,
        output_csv=args.output_csv,
        num_shards=args.num_shards,
        shard_idx=args.shard_idx,
    )


if __name__ == "__main__":
    main()
