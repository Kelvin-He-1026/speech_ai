"""Whisper benchmark on NVIDIA GPU.

GPU-focused counterpart to speech_benchmark.py:
  * dtype defaults to fp16 (H100/H200 throughput peak; same as bf16 on Hopper).
  * batch_size defaults to 32 (GPU shines with bigger batches; HBM has headroom).
  * Sharding is per-GPU instead of per-NUMA-node — launch one process per GPU.
  * Outputs land in output/gpu/ so they don't mix with CPU runs.
  * Per-GPU pinning is done via torch.cuda.set_device — the model lives on
    cuda:0 inside this process regardless of which physical GPU was selected.

Single-GPU run:

    python gpu_speech_benchmark.py --gpu 0 --datasets librispeech \\
        --mode batch --batch-size 32

Multi-GPU throughput sweep (one process per GPU, 4× H200 here):

    cd /home/lenovoai/kelvin/speech_ai
    mkdir -p output/gpu
    for i in 0 1 2 3; do
        python gpu_speech_benchmark.py --gpu $i \\
            --datasets librispeech --mode batch --batch-size 32 \\
            --num-shards 4 --shard-idx $i \\
            > output/gpu/gpu${i}.log 2>&1 &
    done
    wait

    python aggregate_shards.py \\
        output/gpu/*librispeech_batch-b32-shard*of4_*.csv \\
        --write-csv output/gpu/box_summary.csv
"""

import argparse
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# Import speech_benchmark.py without tripping its CPU-default argv sniff.
# That sniff sets CUDA_VISIBLE_DEVICES="" when it doesn't see --device on
# the command line, which would defeat the whole point of this script.
# We temporarily replace sys.argv with one that explicitly declares cuda,
# import, then restore. argparse will run later on the real argv.
_real_argv = sys.argv
sys.argv = [sys.argv[0], "--device", "cuda"]
sys.path.insert(0, str(_HERE))
import speech_benchmark as sb  # noqa: E402
sys.argv = _real_argv


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", default="whisper-large-v3",
                   help=f"Registry key or HF repo. Known: {list(sb.MODEL_REGISTRY)}")
    p.add_argument("--datasets", nargs="+",
                   choices=list(sb.DATASET_LOADERS),
                   default=["librispeech"])
    p.add_argument("--gpu", type=int, default=0,
                   help="CUDA device index passed to torch.cuda.set_device(). "
                        "If CUDA_VISIBLE_DEVICES is set, this is the index into "
                        "the visible list, not the physical GPU.")
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"],
                   help="fp16 = throughput peak on H100/H200. bf16 = same speed, better "
                        "numeric range. fp32 only for accuracy debugging.")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap samples per dataset (default: full split).")
    p.add_argument("--mode", default="batch",
                   choices=["single", "batch", "offline"])
    p.add_argument("--batch-size", type=int, default=32,
                   help="Samples per fused generate() in batch mode. H200 fits batch>=256 "
                        "for whisper-large in fp16; tune up if you have HBM headroom.")
    p.add_argument("--librispeech-split", default="test.clean")
    p.add_argument("--tedlium-split", default="test")
    p.add_argument("--chime6-manifest", default=None)
    p.add_argument("--no-streaming", action="store_true",
                   help="Use the on-disk dataset cache instead of streaming.")
    p.add_argument("--language", default="english")
    p.add_argument("--sort-by-length", action="store_true",
                   help="Sort within each shard by ascending duration. Cuts padding "
                        "waste in batched mode.")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Total shards. Pair with --shard-idx, one process per GPU.")
    p.add_argument("--shard-idx", type=int, default=0)
    p.add_argument("--output-csv", default=None,
                   help="Override CSV path. Default: output/gpu/<auto-name>.csv")
    return p.parse_args()


def _check_cuda_and_pin(gpu_idx: int):
    """Validate CUDA visibility and pin this process to one GPU."""
    import torch
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available in this Python.\n"
            "  Likely cause: torch is the CPU-only wheel (e.g. 2.x.x+cpu).\n"
            "  Fix:  pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
            "  Then re-run."
        )

    n_visible = torch.cuda.device_count()
    if not (0 <= gpu_idx < n_visible):
        raise SystemExit(
            f"--gpu {gpu_idx} is out of range. torch sees {n_visible} CUDA device(s) "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')})."
        )

    torch.cuda.set_device(gpu_idx)

    props = torch.cuda.get_device_properties(gpu_idx)
    print("===== GPU =====")
    print(f"  CUDA device index:  {gpu_idx}")
    print(f"  Device name:        {props.name}")
    print(f"  HBM total (GiB):    {props.total_memory / (1024**3):.1f}")
    print(f"  Compute capability: {props.major}.{props.minor}")
    print(f"  Multi-processors:   {props.multi_processor_count}")
    print(f"  CUDA visible count: {n_visible}")
    print(f"  torch version:      {torch.__version__}")
    print()


def main():
    args = parse_args()
    if args.num_shards < 1 or not (0 <= args.shard_idx < args.num_shards):
        raise SystemExit(
            f"--shard-idx ({args.shard_idx}) must be in [0, --num-shards={args.num_shards})"
        )

    _check_cuda_and_pin(args.gpu)

    # Route all outputs into output/gpu/ instead of output/pytorch/, regardless
    # of whether the model happens to be torch- or OV-backed. We do this by
    # monkey-patching speech_benchmark._output_dir_for; benchmark_model() and
    # _append_box_summary_row() both call it.
    gpu_out_dir = sb.OUTPUT_ROOT / "gpu"
    gpu_out_dir.mkdir(parents=True, exist_ok=True)
    sb._output_dir_for = lambda is_ov: gpu_out_dir

    reports = sb.benchmark_model(
        args.model,
        device="cuda",
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

    # Surface peak HBM for each run so it's easy to gauge headroom for tuning
    # batch_size upward.
    print()
    print("===== Peak HBM per run =====")
    for r in reports:
        if r.peak_cuda_mb is not None:
            print(f"  {r.dataset:14s} {r.mode:8s} bs={r.batch_size:<4d}  "
                  f"peak HBM: {r.peak_cuda_mb:8.1f} MiB "
                  f"({r.peak_cuda_mb / 1024:.2f} GiB)")


if __name__ == "__main__":
    main()
