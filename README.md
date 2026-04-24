# speech_ai

Whisper ASR benchmarking on LibriSpeech, TED-LIUM, and CHiME-6 across PyTorch and OpenVINO model variants.

## Layout

```
speech_ai/
├── model_loader.py          # Downloads (if missing) and loads a Whisper model + processor
├── speech_benchmark.py      # Runs a model across all datasets, writes per-sample + summary CSVs
├── consolidate_outputs.py   # Merges per-sample CSVs into one combined_<date>.csv
├── models/                  # Auto-populated model weights (gitignored)
├── data/                    # HuggingFace datasets cache (gitignored)
└── output/                  # Benchmark CSVs (gitignored by default)
```

## Setup

```bash
pip install transformers huggingface_hub psutil numpy soundfile librosa jiwer datasets
pip install "optimum[openvino]"        # for the *-ov model variants
pip install torch                      # for the vanilla openai/whisper-large-v3
```

For gated datasets (e.g. `chengan/tedlium_small`):

```bash
huggingface-cli login                  # paste a Read token from huggingface.co/settings/tokens
```

…and click "Agree and access repository" on the dataset's Hub page first.

## Models

`model_loader.py` exposes a registry of short aliases:

| Alias                          | HF repo                                  |
| ------------------------------ | ---------------------------------------- |
| `whisper-large-v3`             | `openai/whisper-large-v3`                |
| `whisper-large-v3-fp16-ov`     | `OpenVINO/whisper-large-v3-fp16-ov`      |
| `whisper-large-v3-int8-ov`     | `OpenVINO/whisper-large-v3-int8-ov`      |
| `whisper-large-v3-int4-ov`     | `OpenVINO/whisper-large-v3-int4-ov`      |

Any other HF repo ID also works as `--model <repo>`.

The first call downloads weights into `models/<safe_repo_id>/` (only safetensors / OpenVINO IR files — framework duplicates like `pytorch_model.bin`, `tf_model.h5`, `flax_model.msgpack` are skipped).

```python
from model_loader import load_model

model, proc = load_model("whisper-large-v3", device="cuda:0", dtype="fp16")
model, proc = load_model("whisper-large-v3-int8-ov", device="CPU")
```

## Datasets

| Dataset      | Source                                            | Notes                                      |
| ------------ | ------------------------------------------------- | ------------------------------------------ |
| LibriSpeech  | HF `openslr/librispeech_asr` (split `test.clean`) | public                                     |
| TED-LIUM     | HF `chengan/tedlium_small` (split `test`)         | gated — accept terms + `huggingface-cli login` |
| CHiME-6      | Local JSONL manifest, register at chimechallenge.org | not on HF; see manifest format below    |

CHiME-6 manifest (one JSON per line):

```json
{"id": "S02_P05_0001234-0001789", "audio": "/abs/path/seg.wav", "text": "yeah I think so"}
```

## Run

Single model, all three datasets, default output paths:

```bash
# OpenVINO INT8 on CPU
python -m speech_ai.speech_benchmark --model whisper-large-v3-int8-ov \
    --device cpu --chime6-manifest /data/chime6/eval.jsonl

# Vanilla fp16 on the second NVIDIA GPU
python -m speech_ai.speech_benchmark --model whisper-large-v3 \
    --device cuda --gpu 1 --dtype fp16 \
    --chime6-manifest /data/chime6/eval.jsonl
```

Other useful flags:

| Flag                  | Default                  | Purpose                                              |
| --------------------- | ------------------------ | ---------------------------------------------------- |
| `--datasets`          | `librispeech tedlium chime6` | Subset of datasets to run                        |
| `--max-samples N`     | full split               | Cap utterances per dataset (smoke testing)          |
| `--no-streaming`      | off                      | Download datasets fully into `data/`                |
| `--librispeech-split` | `test.clean`             | LibriSpeech split                                    |
| `--tedlium-split`     | `test`                   | TED-LIUM split                                       |
| `--language`          | `english`                | Whisper generation language hint                     |
| `--output-csv`        | `output/<model>_<date>.csv` | Override summary CSV path                          |

CHiME-6 is auto-skipped (with a `[skip]` message) if `--chime6-manifest` isn't supplied.

## Outputs

Each run produces multiple CSVs in `output/`:

| File                               | Rows         | Contents                                   |
| ---------------------------------- | ------------ | ------------------------------------------ |
| `<model>_<date>.csv`               | 1 per dataset | Aggregate metrics                         |
| `<model>_<dataset>_<date>.csv`     | 1 per utterance | Per-sample diagnostics                  |

### Summary columns
`timestamp, model, dataset, device, resolved_device, dtype, n_samples, total_audio_s, total_wall_s, throughput_xrt, avg_ttft_ms, p50_ttft_ms, p95_ttft_ms, avg_tpot_ms, wer, sub_rate, del_rate, ins_rate, peak_rss_mb, peak_cuda_mb`

### Per-sample columns
`sample_id, audio_seconds, total_seconds, ttft_seconds, n_new_tokens, wer, n_ref_words, substitutions, deletions, insertions, reference, hypothesis, reference_normalized, hypothesis_normalized`

The `*_normalized` columns are what WER is computed on (Whisper's `EnglishTextNormalizer`: lowercase, strip punctuation, expand contractions, etc.).

## Metrics

| Metric              | How it's measured                                                                       |
| ------------------- | --------------------------------------------------------------------------------------- |
| WER, sub/del/ins    | `jiwer.process_words` on normalized refs/hyps                                           |
| Throughput (xRT)    | Σ audio_seconds / Σ wall_seconds                                                        |
| TTFT                | Custom `BaseStreamer` records first `put()` time after `model.generate(streamer=…)`     |
| TPOT                | `(total_decode − ttft) / (n_new_tokens − 1)` per sample, then averaged                  |
| Peak RSS            | Background thread polling `psutil.Process().memory_info().rss` every 50 ms              |
| Peak CUDA           | `torch.cuda.max_memory_allocated()` (CPU runs hide all GPUs via `CUDA_VISIBLE_DEVICES=""`) |

## Consolidate per-sample CSVs

Merge every `<model>_<dataset>_<date>.csv` into one file with `model` + `dataset` columns prepended:

```bash
python -m speech_ai.consolidate_outputs
# → output/combined_<today>.csv
```

Drop into pandas / Excel and pivot by `model` × `dataset` to compare runs.

## Notes

- CPU runs set `CUDA_VISIBLE_DEVICES=""` before importing torch, so no CUDA context is created and `nvidia-smi` reports 0 MiB for the process.
- For the vanilla torch path on CUDA, `torch.backends.cuda.enable_cudnn_sdp(False)` is set to avoid a cuDNN-frontend crash when `libnvrtc` isn't installed. Install `nvidia-cuda-nvrtc-cu12` (or `-cu11`) if you want the cuDNN attention backend back.
- HF `datasets >= 4.0` defaults to `torchcodec` for audio decoding; we bypass it via `cast_column("audio", Audio(decode=False))` and decode with `soundfile`, so you don't need FFmpeg/torchcodec installed.
