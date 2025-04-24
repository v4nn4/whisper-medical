from transformers.models.whisper import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


def download_hf_model(model_id: str):
    model = WhisperForConditionalGeneration.from_pretrained(
        "qanastek-whisper-small-fr", trust_remote_code=True
    )
    model.save_pretrained("qanastek-whisper-small-fr", safe_serialization=False)
    processor = WhisperProcessor.from_pretrained(model_id)
    assert processor is WhisperProcessor
    processor.save_pretrained("qanastek-whisper-small-fr")
