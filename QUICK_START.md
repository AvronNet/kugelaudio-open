# KugelAudio Quick Start

Quick reference for running KugelAudio locally. Defaults: CUDA GPU, 4-bit quantized language model (fits an 8 GB card, ~6.5 GB used).

## Run

No build step. Run from the project folder with `uv run`.

```powershell
uv run python start.py                                  # web UI at http://127.0.0.1:7860
uv run python start.py generate "Hello!" -o out.wav     # text -> wav
uv run python start.py verify out.wav                   # check for the KugelAudio watermark
```

The first run downloads the model from Hugging Face. Loading takes ~30 s. In the UI the model loads on the first request.

## Options

### `generate`

| Option | Default | Notes |
|---|---|---|
| `text` (positional) | required | Text to speak |
| `-o`, `--output` | `output.wav` | Output file |
| `-v`, `--voice` | `default` | See voices below |
| `--quantize` | `4bit` | `4bit`, `8bit` or `none` |
| `--cfg-scale` | `3.0` | 1.0-10.0. Higher sticks closer to the text |
| `--max-tokens` | `4096` | Caps audio length. Lower it for quick tests |
| `--model` | `kugelaudio/kugelaudio-0-open` | Model ID |

### `ui`

| Option | Default | Notes |
|---|---|---|
| `--quantize` | `4bit` | `4bit`, `8bit` or `none` |
| `--host` | `127.0.0.1` | Use `0.0.0.0` to allow other devices on your network |
| `--port` | `7860` | Server port |
| `--share` | off | Public Gradio link (anyone with the link can use it) |

## Quantization

| Mode | VRAM | Use when |
|---|---|---|
| `4bit` (default) | ~6.5 GB | 8 GB GPUs. Fastest to load, small quality trade-off |
| `8bit` | ~9-10 GB (estimate, not tested here) | 12 GB+ GPUs |
| `none` | ~16 GB | 16 GB+ GPUs, full bf16 quality |

Quantization needs CUDA. Without it the model loads unquantized on the CPU with a warning, and that is slow.

## Voices

Pass with `--voice NAME` or pick in the UI.

| Name | Language | Description |
|---|---|---|
| `default` | German | Calm female narrator |
| `clear` | German | Clear, young female conversational voice |
| `english_female` | English (UK) | Friendly female teacher |
| `english_male` | English (UK) | Conversational male voice |

## Examples

```powershell
# English male voice
uv run python start.py generate "Good evening, everyone." --voice english_male -o evening.wav

# Short quick test
uv run python start.py generate "Testing." --max-tokens 300 -o test.wav

# Stronger text adherence
uv run python start.py generate "Read this exactly." --cfg-scale 5 -o exact.wav

# UI reachable from other devices on the network
uv run python start.py ui --host 0.0.0.0 --port 8080
```

## Check CUDA

```powershell
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Startup also prints `Quantization: 4bit` (CLI) or `Loading model ... on cuda (quantization: 4bit)` (UI).

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'torch'` | Use `uv run ...` (or activate `.venv`), not plain `python` |
| `is_available()` is `False` | Update the NVIDIA driver. Don't `pip install torch` over the environment. Re-run `uv sync` |
| `UnicodeEncodeError` when piping output | `$env:PYTHONUTF8=1` |
| Very slow generation (~3 s per token) | GPU memory is full. Close other GPU-heavy apps, keep `--quantize 4bit` |
| Out of memory with `--quantize none` | Needs ~16 GB VRAM. Use `4bit` |

## Also available

- Installed command: `uv run kugelaudio ui` / `uv run kugelaudio generate "Hello!" --quantize 4bit` (same options as above).
- Python API: see [README.md](README.md#python-api). Use `kugelaudio_open.utils.load_model_and_processor(quantization="4bit")` to get the same quantized loading in your own code.
