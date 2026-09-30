"""High-level generation utilities for KugelAudio."""

import importlib.util
import warnings
from typing import Optional, Union

import torch

QUANTIZATION_CHOICES = ("4bit", "8bit", "none")
DEFAULT_QUANTIZATION = "4bit"

# Only the Qwen2 language model backbone is quantized. The LM head (tied to the
# embeddings), acoustic decoder, connector and diffusion head are small and
# quality-sensitive, so they stay in bf16.
_QUANTIZATION_SKIP_MODULES = [
    "lm_head",
    "acoustic_tokenizer",
    "acoustic_connector",
    "prediction_head",
]


def _quantization_config(quantization: str, compute_dtype: torch.dtype):
    from transformers import BitsAndBytesConfig

    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
            llm_int8_skip_modules=_QUANTIZATION_SKIP_MODULES,
        )
    return BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_skip_modules=_QUANTIZATION_SKIP_MODULES,
    )


def load_model_and_processor(
    model_name_or_path: str = "kugelaudio/kugelaudio-0-open",
    device: Optional[Union[str, torch.device]] = None,
    torch_dtype: Optional[torch.dtype] = None,
    use_flash_attention: bool = True,
    quantization: str = DEFAULT_QUANTIZATION,
):
    """Load KugelAudio model and processor.

    Args:
        model_name_or_path: HuggingFace model ID or local path
        device: Device to load model on (auto-detected if None)
        torch_dtype: Data type for model weights
        use_flash_attention: Whether to use flash attention if installed
        quantization: "4bit" (default), "8bit" or "none". Quantization uses
            bitsandbytes and needs CUDA; without CUDA it falls back to "none".

    Returns:
        Tuple of (model, processor)

    Example:
        >>> model, processor = load_model_and_processor("kugelaudio/kugelaudio-0-open")
    """
    from kugelaudio_open.models import KugelAudioForConditionalGenerationInference
    from kugelaudio_open.processors import KugelAudioProcessor

    if quantization not in QUANTIZATION_CHOICES:
        raise ValueError(f"quantization must be one of {QUANTIZATION_CHOICES}, got {quantization!r}")

    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Auto-detect dtype
    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    if quantization != "none" and device.type != "cuda":
        warnings.warn(
            f"{quantization} quantization needs CUDA but device is '{device.type}'; "
            "loading unquantized weights instead."
        )
        quantization = "none"

    use_flash = use_flash_attention and importlib.util.find_spec("flash_attn") is not None
    load_kwargs = {
        "torch_dtype": torch_dtype,
        "attn_implementation": "flash_attention_2" if use_flash else "sdpa",
    }

    # Load model
    if quantization == "none":
        model = KugelAudioForConditionalGenerationInference.from_pretrained(
            model_name_or_path, **load_kwargs
        ).to(device)
    else:
        # Quantized weights are placed on the GPU while loading and can't be moved with .to()
        model = KugelAudioForConditionalGenerationInference.from_pretrained(
            model_name_or_path,
            quantization_config=_quantization_config(quantization, torch_dtype),
            device_map={"": device.index if device.index is not None else 0},
            **load_kwargs,
        )

    model.eval()
    if quantization != "none":
        # Frees ~1 GB so a 4-bit 7B model plus KV cache fits on an 8 GB GPU
        model.restrict_lm_head()

    # Strip encoders to save VRAM (only decoders needed for inference)
    model.model.strip_encoders()

    # Load processor
    processor = KugelAudioProcessor.from_pretrained(model_name_or_path)

    return model, processor


def generate_speech(
    model,
    processor,
    text: str,
    voice: Optional[str] = None,
    cfg_scale: float = 3.0,
    max_new_tokens: int = 4096,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Generate speech from text using an optional pre-encoded voice.

    Voice cloning from raw audio is no longer supported. Use a pre-encoded
    voice name from the voices.json registry instead.

    All generated audio is automatically watermarked for identification.

    Args:
        model: KugelAudio model
        processor: KugelAudio processor
        text: Text to synthesize
        voice: Name of a pre-encoded voice (from voices.json registry)
        cfg_scale: Classifier-free guidance scale
        max_new_tokens: Maximum number of tokens to generate
        device: Device for generation

    Returns:
        Generated audio tensor (watermarked)

    Example:
        >>> audio = generate_speech(model, processor, "Hello world!", voice="default")
        >>> processor.save_audio(audio, "output.wav")
    """
    if device is None:
        device = next(model.parameters()).device

    # Process inputs with optional pre-encoded voice
    inputs = processor(text=text, voice=voice, return_tensors="pt")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Generate (watermark is automatically applied by the model)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            cfg_scale=cfg_scale,
            max_new_tokens=max_new_tokens,
        )

    audio = outputs.speech_outputs[0] if outputs.speech_outputs else None

    if audio is None:
        raise RuntimeError("Generation failed - no audio output")

    return audio
