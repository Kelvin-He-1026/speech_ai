from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoProcessor

MODELS_DIR = Path(__file__).resolve().parent / "models"

MODEL_REGISTRY = {
    "whisper-large-v3":          "openai/whisper-large-v3",
    "whisper-large-v3-int8-ov":  "OpenVINO/whisper-large-v3-int8-ov",
    "whisper-large-v3-int4-ov":  "OpenVINO/whisper-large-v3-int4-ov",
    "whisper-large-v3-fp16-ov":  "OpenVINO/whisper-large-v3-fp16-ov",
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


def _is_openvino(repo_id: str) -> bool:
    return "-ov" in repo_id.lower() or "openvino" in repo_id.lower()


def ensure_model(name: str) -> Path:
    repo_id = _resolve_repo_id(name)
    target = _local_path(repo_id)
    if not target.exists() or not any(target.iterdir()):
        target.mkdir(parents=True, exist_ok=True)
        patterns = _OPENVINO_PATTERNS if _is_openvino(repo_id) else _TRANSFORMERS_PATTERNS
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target),
            local_dir_use_symlinks=False,
            allow_patterns=patterns,
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
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            str(path), torch_dtype=torch_dtype, low_cpu_mem_usage=True
        ).to(torch_device)

    return model, processor


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "whisper-large-v3-int8-ov"
    m, p = load_model(target)
    print(f"Loaded {target}: {type(m).__name__}, processor {type(p).__name__}")
