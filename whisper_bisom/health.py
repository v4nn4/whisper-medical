import json
from pathlib import Path


def health_check(wav_dir: Path, reference_transcripts_path: Path):
    errors = []

    if not wav_dir.exists() or not wav_dir.is_dir():
        errors.append(f"Missing or invalid directory: {wav_dir}")

    elif not any(wav_dir.glob("*.wav")):
        errors.append(f"No .wav files found in: {wav_dir}")

    if not reference_transcripts_path.exists():
        errors.append(f"Missing file: {reference_transcripts_path}")

    else:
        try:
            with reference_transcripts_path.open() as f:
                json.load(f)
        except json.JSONDecodeError:
            errors.append(f"Invalid JSON format in: {reference_transcripts_path}")

    if errors:
        print("❌ Health check failed:")
        for error in errors:
            print("-", error)
    else:
        print("✅ Health check passed: all required files are present and valid.")
