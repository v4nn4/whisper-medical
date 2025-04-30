from pathlib import Path
from transformers.models.whisper import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


def download_hf_model(model_id: str, models_dir: Path):
    model_path = models_dir / model_id / "pytorch_model.bin"
    if model_path.exists():
        print(f"Model {model_id} already exists at {model_path}.")
        return
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, trust_remote_code=True
    )
    model_dir = models_dir / model_id
    model.save_pretrained(model_dir, safe_serialization=False)
    processor = WhisperProcessor.from_pretrained(model_id)
    processor.save_pretrained(model_dir)
    print(f"Model downloaded to {model_dir}")
