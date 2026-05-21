"""Quantize a Whisper checkpoint with auto-round (RTN).

Defaults to W8A8 INT in `llm_compressor` format (the scheme used by Intel's
MLPerf Whisper SUT). For other configs (W4A16, W4A4, etc.) the output format
is auto-switched to `auto_round`, since `llm_compressor` only supports a fixed
set of schemes: MXFP4, MXFP8, NVFP4, FPW8A16, FP8_STATIC, INT8_W8A8, FP8_BLOCK.

Mirrors whisper_mlperf/code/workspace/quantize_model.py with current auto-round
API: AutoRound (not AutoRoundMLLM), quantize_and_save(), dtype= (not torch_dtype=).
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer

from auto_round import AutoRound

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "models" / "openai__whisper-large-v3"
DEFAULT_OUTPUT = HERE / "models" / "openai__whisper-large-v3-w8a8"


def _default_format(bits: int, act_bits: int) -> str:
    """llm_compressor only supports INT8_W8A8 among INT schemes; fall back to
    auto_round for everything else (e.g. W4A16, W4A4)."""
    if bits == 8 and act_bits == 8:
        return "llm_compressor"
    return "auto_round"


def quantize(input_path: Path, output_path: Path, *,
             bits: int = 8, act_bits: int = 8,
             group_size: int = -1, sym: bool = True,
             fmt: str | None = None) -> None:
    fmt = fmt or _default_format(bits, act_bits)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        input_path, dtype=torch.float32, low_cpu_mem_usage=True, use_safetensors=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(input_path)
    processor = AutoProcessor.from_pretrained(input_path)

    autoround = AutoRound(
        model, tokenizer, processor=processor,
        bits=bits, group_size=group_size, sym=sym, act_bits=act_bits,
        iters=0, disable_opt_rtn=True,
    )
    autoround.quantize_and_save(str(output_path), format=fmt, inplace=True)
    print(f"Saved W{bits}A{act_bits} model ({fmt}) to {output_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", "-i", type=Path, default=DEFAULT_INPUT,
                   help=f"Source HuggingFace model directory (default: {DEFAULT_INPUT})")
    p.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT,
                   help=f"Destination directory for quantized weights (default: {DEFAULT_OUTPUT})")
    p.add_argument("--bits", type=int, default=8, help="Weight bits (default: 8)")
    p.add_argument("--act-bits", type=int, default=8, help="Activation bits (default: 8)")
    p.add_argument("--group-size", type=int, default=-1,
                   help="Group size; -1 = per-channel (default: -1)")
    p.add_argument("--asym", action="store_true",
                   help="Use asymmetric quantization (default: symmetric)")
    p.add_argument("--format", dest="fmt", default=None,
                   help="Output format (default: llm_compressor for W8A8, auto_round otherwise). "
                        "Other options: auto_gptq, auto_awq, fake, fp8, gguf:q4_k_m, ...")
    args = p.parse_args()

    quantize(args.input, args.output,
             bits=args.bits, act_bits=args.act_bits,
             group_size=args.group_size, sym=not args.asym,
             fmt=args.fmt)


if __name__ == "__main__":
    main()
