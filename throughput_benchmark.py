"""Throughput benchmark for Whisper on LibriSpeech test-clean (CPU).

Measures xRT (audio-seconds / wall-seconds) and tok/s.

Threading env vars (OMP_NUM_THREADS / MKL_NUM_THREADS) must be set BEFORE
torch is imported to take effect, so this script parses args first and only
then imports torch / transformers / datasets.

Recommended on a dual Xeon 8568Y+ (2x 48-core, AMX-capable):

    # single-socket, one NUMA node — best xRT per stream:
    numactl --cpunodebind=0 --membind=0 \\
        python throughput_benchmark.py --dtype bf16 --threads 48

    # MAX THROUGHPUT — two processes, one per socket, dataset sharded in half:
    numactl --cpunodebind=0 --membind=0 \\
        python throughput_benchmark.py --dtype bf16 --threads 48 --streaming \\
        --num-shards 2 --shard-idx 0 &
    numactl --cpunodebind=1 --membind=1 \\
        python throughput_benchmark.py --dtype bf16 --threads 48 --streaming \\
        --num-shards 2 --shard-idx 1 &
    wait
    # Box xRT = (audio_shard0 + audio_shard1) / max(wall_shard0, wall_shard1)

    # never use 192 — HT siblings contend on the same vector/AMX units and hurt FP throughput.

Results are written to output/throughput/:
  - summary.csv         (one appended row per run)
  - <run_id>_samples.csv (per-sample timings)
"""

import argparse
import csv
import datetime as _dt
import gc
import os
import platform
import statistics
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
DATA_DIR = _HERE / "data"
HF_DATASETS_CACHE = DATA_DIR / "hf_datasets"
HF_HOME = DATA_DIR / "hf_home"
OUTPUT_DIR = _HERE / "output" / "throughput"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="whisper-large-v3",
                   help="Alias from model_loader.MODEL_REGISTRY, or any HF repo ID.")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"],
                   help="Compute dtype. bf16 enables AMX on Sapphire/Emerald Rapids.")
    p.add_argument("--dataset", default="openslr/librispeech_asr")
    p.add_argument("--config", default="clean")
    p.add_argument("--split", default="test",
                   help="LibriSpeech test split. With config=clean this is the ~2620-sample test-clean.")
    p.add_argument("--num-samples", type=int, default=None,
                   help="Cap on benchmark samples after warmup. Default: all (~2620 for test-clean).")
    p.add_argument("--warmup-samples", type=int, default=3)
    p.add_argument("--threads", type=int, default=None,
                   help="OMP_NUM_THREADS / MKL_NUM_THREADS. If unset, leave to system default.")
    p.add_argument("--language", default="en")
    p.add_argument("--task", default="transcribe")
    p.add_argument("--max-new-tokens", type=int, default=440)
    p.add_argument("--num-beams", type=int, default=1)
    p.add_argument("--streaming", action="store_true",
                   help="Stream the dataset over HTTP instead of using the on-disk cache.")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--gc-every", type=int, default=50,
                   help="Run gc.collect() and drop tensor caches every N samples.")
    p.add_argument("--run-id", default=None,
                   help="Override run identifier (default: <model>_<dtype>_<threads>_<UTC timestamp>).")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Total number of shards. Pair with --shard-idx for one-process-per-socket throughput. "
                        "Each shard processes every Nth sample (e.g. shard 0 of 2 gets samples 0,2,4,...).")
    p.add_argument("--shard-idx", type=int, default=0,
                   help="This process's shard index in [0, num_shards). Ignored when --num-shards=1.")
    return p.parse_args()


def configure_threads_and_cache(num_threads):
    """Set env BEFORE importing torch / datasets so the values actually apply."""
    if num_threads is not None:
        n = str(num_threads)
        os.environ["OMP_NUM_THREADS"] = n
        os.environ["MKL_NUM_THREADS"] = n
        os.environ["OPENBLAS_NUM_THREADS"] = n
        os.environ["NUMEXPR_NUM_THREADS"] = n
        os.environ["VECLIB_MAXIMUM_THREADS"] = n

    HF_DATASETS_CACHE.mkdir(parents=True, exist_ok=True)
    HF_HOME.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    os.environ.setdefault("HF_DATASETS_CACHE", str(HF_DATASETS_CACHE))


_WHISPER_PROMPT_LEN = 4  # <|startoftranscript|> <|lang|> <|task|> <|notimestamps|>


def run_one(model, processor, torch, audio_array, sampling_rate, model_dtype,
            language, task, num_beams):
    """`max_new_tokens` is set once on `model.generation_config` in main() so
    we don't pass it per call — see comment there for why."""
    inputs = processor(
        audio_array,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        return_attention_mask=True,
    )
    input_features = inputs.input_features.to(model_dtype)
    attention_mask = inputs.get("attention_mask")

    gen_kwargs = dict(
        input_features=input_features,
        language=language,
        task=task,
        num_beams=num_beams,
        do_sample=False,
        return_timestamps=False,
    )
    if attention_mask is not None:
        gen_kwargs["attention_mask"] = attention_mask

    t0 = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**gen_kwargs)
    elapsed = time.perf_counter() - t0

    new_tokens = max(0, int(output_ids.shape[-1]) - _WHISPER_PROMPT_LEN)
    # Drop refs immediately so the caller's gc.collect() can reclaim.
    del inputs, input_features, attention_mask, gen_kwargs, output_ids
    return elapsed, new_tokens


def free_caches(torch):
    """Drop Python garbage and any allocator caches between samples."""
    gc.collect()
    # Inductor / oneDNN keep some scratch around; CPU has no torch.cuda.empty_cache
    # equivalent, but malloc trim helps return freed pages to the OS.
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def main():
    args = parse_args()
    configure_threads_and_cache(args.threads)

    # Lazy imports — env vars above must be set first to take effect.
    import io
    import numpy as np
    import soundfile as sf
    import torch
    import transformers
    import datasets
    from datasets import Audio, load_dataset

    def _decode_audio(cell):
        """Return (np.float32 mono, sr). Accepts either a {bytes, path} cell
        (Audio(decode=False)) or an already-decoded {array, sampling_rate} cell."""
        if isinstance(cell, dict) and "array" in cell and cell["array"] is not None:
            return np.asarray(cell["array"], dtype=np.float32), int(cell["sampling_rate"])
        if cell.get("bytes"):
            audio, sr = sf.read(io.BytesIO(cell["bytes"]))
        else:
            audio, sr = sf.read(cell["path"])
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32, copy=False), int(sr)

    def _resample_if_needed(arr, sr, target_sr=16000):
        if sr == target_sr:
            return arr, sr
        import librosa
        return librosa.resample(arr, orig_sr=sr, target_sr=target_sr), target_sr

    try:
        from .model_loader import load_model
    except ImportError:
        sys.path.insert(0, str(_HERE))
        from model_loader import load_model

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    if args.threads is not None:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(max(1, min(4, args.threads // 8)))

    if args.num_shards < 1 or not (0 <= args.shard_idx < args.num_shards):
        raise SystemExit(f"--shard-idx ({args.shard_idx}) must be in [0, --num-shards={args.num_shards})")

    shard_tag = f"_shard{args.shard_idx}of{args.num_shards}" if args.num_shards > 1 else ""
    run_id = args.run_id or (
        f"{args.model.replace('/', '__')}_{args.dtype}"
        f"_t{args.threads if args.threads is not None else 'sys'}"
        f"{shard_tag}"
        f"_{_dt.datetime.now(_dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    per_sample_csv = OUTPUT_DIR / f"{run_id}_samples.csv"
    summary_csv = OUTPUT_DIR / "summary.csv"

    print("===== Environment =====")
    print(f"platform:              {platform.platform()}")
    print(f"processor:             {platform.processor()}")
    print(f"torch:                 {torch.__version__}")
    print(f"transformers:          {transformers.__version__}")
    print(f"datasets:              {datasets.__version__}")
    print(f"OMP_NUM_THREADS:       {os.environ.get('OMP_NUM_THREADS', '<unset>')}")
    print(f"MKL_NUM_THREADS:       {os.environ.get('MKL_NUM_THREADS', '<unset>')}")
    print(f"torch num_threads:     {torch.get_num_threads()}")
    print(f"torch interop_threads: {torch.get_num_interop_threads()}")
    print(f"mkldnn:                {torch.backends.mkldnn.is_available()}")
    print(f"dtype:                 {torch_dtype}")
    print(f"run_id:                {run_id}")
    print()

    print(f"===== Loading model: {args.model} =====")
    t_load = time.perf_counter()
    model, processor = load_model(args.model, device="cpu", dtype=args.dtype)
    model.eval()
    print(f"load wall:             {time.perf_counter() - t_load:.2f}s")

    # Length control: Whisper ships with generation_config.max_length=448. If we
    # ALSO pass max_new_tokens as a generate() kwarg, transformers emits
    #   "Both `max_new_tokens` (=440) and `max_length`(=...) seem to have been set."
    # on every call (logger.warning in transformers.generation.utils._prepare_generated_length).
    # The robust fix: own length-control on the gen_config side, with no kwarg
    # for max_new_tokens. Clear max_length, set max_new_tokens. Belt-and-suspenders:
    # also filter the warning at the logger level in case any inner segment-pass
    # rebuilds a default GenerationConfig().
    g_cfg = getattr(model, "generation_config", None)
    if g_cfg is not None:
        g_cfg.max_length = None
        g_cfg.max_new_tokens = args.max_new_tokens

    import logging as _logging
    class _MaxLengthWarnFilter(_logging.Filter):
        def filter(self, record):
            return "Both `max_new_tokens`" not in record.getMessage()
    _logging.getLogger("transformers.generation.utils").addFilter(_MaxLengthWarnFilter())

    print(f"===== Loading dataset: {args.dataset} [{args.config}] {args.split} =====")
    ds = load_dataset(
        args.dataset,
        args.config,
        split=args.split,
        streaming=args.streaming,
        cache_dir=str(HF_DATASETS_CACHE),
    )
    # decode=False keeps the audio cell as {bytes, path} so we can decode with
    # soundfile ourselves. datasets >=4.8 routes Audio decoding through torchcodec
    # which is an avoidable extra dep — soundfile handles LibriSpeech FLAC fine.
    ds = ds.cast_column("audio", Audio(decode=False))

    # Sharding is applied in the iterator below as a modulo filter on the global
    # sample index. IterableDataset.shard() can't help here because openslr's
    # streaming source is a single file (so it offers exactly 1 shard).
    total_available = None if args.streaming else len(ds)
    if total_available is not None and args.num_shards > 1:
        expected = (total_available - args.shard_idx + args.num_shards - 1) // args.num_shards
        print(f"split size:            {total_available}  (shard {args.shard_idx} of {args.num_shards} → ~{expected} samples)")
    else:
        suffix = f"  (shard {args.shard_idx} of {args.num_shards})" if args.num_shards > 1 else ""
        print(f"split size:            {total_available if total_available is not None else 'streaming'}{suffix}")
    print(f"warmup samples:        {args.warmup_samples}")
    print(f"benchmark samples:     {'all remaining' if args.num_samples is None else args.num_samples}")
    print()

    common = dict(
        model=model, processor=processor, torch=torch, model_dtype=torch_dtype,
        language=args.language, task=args.task,
        num_beams=args.num_beams,
    )

    # Iterate the dataset lazily — never materialize 2620 audio arrays into RAM.
    base_iter = iter(ds) if args.streaming else (ds[i] for i in range(len(ds)))
    if args.num_shards > 1:
        # Modulo shard: process i belongs to me iff (i % num_shards) == shard_idx.
        # In streaming mode each process still pulls every row over the wire and
        # discards 1-of-N — wasted bandwidth (~346 MB extra for test-clean × 2
        # shards) but trivial compared to compute. For non-streaming this is free.
        def _shard(it):
            for i, ex in enumerate(it):
                if i % args.num_shards == args.shard_idx:
                    yield ex
        sample_iter = _shard(base_iter)
    else:
        sample_iter = base_iter

    print("===== Warmup =====")
    for i in range(1, args.warmup_samples + 1):
        try:
            row = next(sample_iter)
        except StopIteration:
            break
        arr, sr = _decode_audio(row["audio"])
        arr, sr = _resample_if_needed(arr, sr)
        e, t = run_one(**common, audio_array=arr, sampling_rate=sr)
        print(f"  warmup {i}/{args.warmup_samples}: {e:.2f}s, {t} tok")
        del row, arr
    free_caches(torch)
    print()

    print("===== Benchmark =====")
    total_audio = 0.0
    total_wall = 0.0
    total_tokens = 0
    per_elapsed = []
    per_xrt = []
    n_done = 0

    with per_sample_csv.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["idx", "sample_id", "audio_seconds", "wall_seconds",
                         "n_new_tokens", "xrt", "tok_per_s"])

        for i, row in enumerate(sample_iter, 1):
            if args.num_samples is not None and i > args.num_samples:
                break

            a = row["audio"]
            arr, sr = _decode_audio(a)
            arr, sr = _resample_if_needed(arr, sr)
            audio_sec = len(arr) / sr
            sid = row.get("id") or row.get("audio_id") or a.get("path") or f"sample_{i}"

            elapsed, tokens = run_one(**common, audio_array=arr, sampling_rate=sr)

            total_audio += audio_sec
            total_wall += elapsed
            total_tokens += tokens
            per_elapsed.append(elapsed)
            sample_xrt = audio_sec / elapsed if elapsed > 0 else 0.0
            per_xrt.append(sample_xrt)
            n_done = i

            writer.writerow([
                i, sid, f"{audio_sec:.4f}", f"{elapsed:.4f}",
                tokens, f"{sample_xrt:.4f}",
                f"{tokens / elapsed if elapsed > 0 else 0.0:.4f}",
            ])

            # Drop the heavy refs from this iteration before the next.
            del row, a, arr

            if i % args.gc_every == 0:
                free_caches(torch)

            if i % args.log_every == 0:
                fp.flush()
                print(
                    f"  {i:5d}  "
                    f"audio={total_audio:8.2f}s  "
                    f"wall={total_wall:8.2f}s  "
                    f"xRT={total_audio / total_wall:6.3f}  "
                    f"tok/s={total_tokens / total_wall:6.2f}"
                )

    if n_done == 0:
        print("No samples processed.")
        return

    xrt = total_audio / total_wall
    rtf = total_wall / total_audio
    toks_per_sec = total_tokens / total_wall
    p50 = statistics.median(per_elapsed)
    p95 = sorted(per_elapsed)[max(0, int(0.95 * len(per_elapsed)) - 1)] if per_elapsed else 0.0
    xrt_p50 = statistics.median(per_xrt)

    print()
    print("===== Final =====")
    print(f"run_id:                {run_id}")
    print(f"model:                 {args.model}")
    print(f"dtype:                 {args.dtype}")
    print(f"threads:               {args.threads}")
    print(f"dataset:               {args.dataset} / {args.config} / {args.split}")
    print(f"samples:               {n_done}")
    print(f"total audio:           {total_audio:.2f} s")
    print(f"total wall:            {total_wall:.2f} s")
    print(f"xRT (aggregate):       {xrt:.4f}x")
    print(f"RTF (aggregate):       {rtf:.4f}")
    print(f"xRT (per-sample p50):  {xrt_p50:.4f}x")
    print(f"latency p50:           {p50*1000:.1f} ms")
    print(f"latency p95:           {p95*1000:.1f} ms")
    print(f"generated tokens:      {total_tokens}")
    print(f"tok/s:                 {toks_per_sec:.2f}")
    print()
    print(f"per-sample CSV:        {per_sample_csv}")
    print(f"summary CSV:           {summary_csv}")

    summary_row = {
        "run_id": run_id,
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "model": args.model,
        "dtype": args.dtype,
        "threads": args.threads if args.threads is not None else "",
        "torch_num_threads": torch.get_num_threads(),
        "backend": "pytorch+mkldnn",
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "n_samples": n_done,
        "total_audio_s": round(total_audio, 3),
        "total_wall_s": round(total_wall, 3),
        "xrt_aggregate": round(xrt, 4),
        "rtf_aggregate": round(rtf, 4),
        "xrt_p50": round(xrt_p50, 4),
        "latency_p50_ms": round(p50 * 1000, 2),
        "latency_p95_ms": round(p95 * 1000, 2),
        "generated_tokens": total_tokens,
        "tok_per_s": round(toks_per_sec, 3),
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "shard_idx": args.shard_idx,
        "num_shards": args.num_shards,
    }
    write_header = not summary_csv.exists()
    with summary_csv.open("a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(summary_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(summary_row)


if __name__ == "__main__":
    main()
