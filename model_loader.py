from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoProcessor

MODELS_DIR = Path(__file__).resolve().parent / "models"

MODEL_REGISTRY = {
    "whisper-large-v3":          "openai/whisper-large-v3",
    "whisper-large-v3-turbo":    "openai/whisper-large-v3-turbo",
    # Locally-quantized checkpoints produced by quantization.py
    # (no HF repo — must exist under models/ before use).
    "whisper-large-v3-w8a8":     "openai/whisper-large-v3-w8a8",
    "whisper-large-v3-w8a8_v2":  "openai/whisper-large-v3-w8a8_v2",
    "whisper-large-v3-w4a16":    "openai/whisper-large-v3-w4a16",
}


def _resolve_repo_id(name: str) -> str:
    return MODEL_REGISTRY.get(name, name)


def _local_path(repo_id: str) -> Path:
    return MODELS_DIR / repo_id.replace("/", "__")


_COMMON_PATTERNS = [
    "*.json", "*.txt", "*.model",
    "tokenizer*", "vocab*", "merges*",
    "normalizer.json", "preprocessor_config.json",
    "added_tokens.json", "special_tokens_map.json",
    "generation_config.json",
]
_TRANSFORMERS_PATTERNS = _COMMON_PATTERNS + ["*.safetensors", "*.safetensors.index.json"]
_OPENVINO_PATTERNS = _COMMON_PATTERNS + ["*.xml", "*.bin"]

# Weight files transformers / optimum will look for at load time.
_TRANSFORMERS_WEIGHT_FILES = (
    "model.safetensors", "model.safetensors.index.json",
    "pytorch_model.bin", "pytorch_model.bin.index.json",
)
_OPENVINO_WEIGHT_FILES = ("openvino_model.xml",)


def _is_openvino(repo_id: str) -> bool:
    return "-ov" in repo_id.lower() or "openvino" in repo_id.lower()


def _is_local_only(repo_id: str) -> bool:
    """Quantized outputs from quantization.py — no matching HF repo exists."""
    n = repo_id.lower()
    return any(tag in n for tag in ("-w8a8", "-w4a16", "-w4a4"))


def _has_weights(target: Path, is_ov: bool) -> bool:
    """True iff `target` already contains a usable weight file. Guards against
    treating an interrupted snapshot_download (configs present, weights missing)
    as a complete checkpoint."""
    files = _OPENVINO_WEIGHT_FILES if is_ov else _TRANSFORMERS_WEIGHT_FILES
    return any((target / f).exists() for f in files)


def ensure_model(name: str) -> Path:
    repo_id = _resolve_repo_id(name)
    target = _local_path(repo_id)
    is_ov = _is_openvino(repo_id)
    if target.exists() and _has_weights(target, is_ov):
        return target
    if _is_local_only(repo_id):
        raise FileNotFoundError(
            f"{target} does not exist (or has no weights). '{name}' is a "
            f"locally-quantized checkpoint with no HF repo — produce it first, "
            f"e.g.:\n  python quantization.py -o {target}"
        )
    target.mkdir(parents=True, exist_ok=True)
    patterns = _OPENVINO_PATTERNS if is_ov else _TRANSFORMERS_PATTERNS
    print(f"[ensure_model] fetching {repo_id} -> {target}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        allow_patterns=patterns,
    )
    if not _has_weights(target, is_ov):
        raise FileNotFoundError(
            f"download of {repo_id} into {target} finished but no weight file "
            f"(expected one of: {_OPENVINO_WEIGHT_FILES if is_ov else _TRANSFORMERS_WEIGHT_FILES}) "
            f"is present. Check network/HF auth and re-run."
        )
    return target


_DTYPE_ALIASES = {
    "fp32": "float32", "float32": "float32", "f32": "float32",
    "fp16": "float16", "float16": "float16", "f16": "float16", "half": "float16",
    "bf16": "bfloat16", "bfloat16": "bfloat16",
}


def _resolve_torch_dtype(dtype, device):
    import torch
    if dtype is None or (isinstance(dtype, str) and dtype.lower() == "auto"):
        return torch.float16 if device == "cuda" else torch.float32
    if isinstance(dtype, torch.dtype):
        return dtype
    key = _DTYPE_ALIASES.get(str(dtype).lower())
    if key is None:
        raise ValueError(f"Unknown dtype: {dtype!r}")
    return getattr(torch, key)


def _has_quant_config(path: Path) -> bool:
    """Detect quantized checkpoints by inspecting config.json (compressed-tensors,
    auto_round, gptq, awq, etc. all write a `quantization_config` block)."""
    import json
    cfg = path / "config.json"
    if not cfg.exists():
        return False
    try:
        with open(cfg) as f:
            return "quantization_config" in json.load(f)
    except Exception:
        return False


def _dequantize_int_linears_inplace(model, target_dtype) -> int:
    """Walk the model; for every Linear whose `weight` is still INT8/UINT8 with
    a `weight_scale` buffer, fold scale (and optional zero-point) into the
    weight so plain bf16/fp16 matmul works.

    compressed-tensors in transformers loads the packed int weights but doesn't
    install a forward-hook to dequantize on CPU — bare nn.Linear then tries
    `bf16 @ int8` and crashes. This converts the model to W8A16-equivalent.
    """
    import torch
    n = 0
    for module in model.modules():
        w_scale = getattr(module, "weight_scale", None)
        w = getattr(module, "weight", None)
        if w_scale is None or w is None:
            continue
        if w.dtype not in (torch.int8, torch.uint8):
            continue
        scale = w_scale.data.to(target_dtype)
        zp = getattr(module, "weight_zero_point", None)
        wf = w.data.to(target_dtype)
        if zp is not None:
            wf = wf - zp.data.to(target_dtype)
        w_dq = (wf * scale).contiguous()
        module.weight = torch.nn.Parameter(w_dq, requires_grad=False)
        for attr in ("weight_scale", "weight_zero_point",
                     "input_scale", "input_zero_point"):
            if hasattr(module, attr):
                try:
                    delattr(module, attr)
                except AttributeError:
                    pass
        n += 1
    return n


def load_model(name: str, device: str = "AUTO", dtype=None):
    repo_id = _resolve_repo_id(name)
    path = ensure_model(name)
    processor = AutoProcessor.from_pretrained(str(path))

    if "-ov" in repo_id.lower() or "openvino" in repo_id.lower():
        from optimum.intel.openvino import OVModelForSpeechSeq2Seq
        model = OVModelForSpeechSeq2Seq.from_pretrained(str(path), device=device)
    else:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq
        torch_device = device.lower()
        if torch_device == "auto":
            torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = _resolve_torch_dtype(dtype, torch_device)

        if _has_quant_config(path):
            # Quantized model: must use device_map so the quantizer hooks fire
            # during load. NEVER call .to() afterwards — it can strip wrappers.
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                str(path), dtype=torch_dtype, device_map=torch_device,
            )
            # CPU fallback: compressed-tensors loads INT8 weights but doesn't
            # install a dequant-on-forward hook on CPU. Fold scale into the
            # weight ourselves so plain bf16 matmul works.
            n = _dequantize_int_linears_inplace(model, torch_dtype)
            if n > 0:
                print(f"[load_model] dequantized {n} int-weight Linears "
                      f"to {torch_dtype} for CPU inference")
        else:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                str(path), dtype=torch_dtype, low_cpu_mem_usage=True
            ).to(torch_device)

    return model, processor


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "whisper-large-v3-int8-ov"
    m, p = load_model(target)
    print(f"Loaded {target}: {type(m).__name__}, processor {type(p).__name__}")
