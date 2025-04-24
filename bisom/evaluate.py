from pathlib import Path
import json
from jiwer import wer, cer
import re
import unicodedata
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess


def full_transcribe(
    whisper_bin: Path = Path("../whisper.cpp/build/bin/Release/whisper-cli"),
    model: Path = Path("models/ggml-model.bin"),
    language: str = "fr",
    input_dir: Path = Path("data/samples/wav"),
    output_path: Path = Path("data/samples/transcripts/transcriptions.json"),
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    transcriptions = {}

    for i in range(1, 32):
        wav_file = input_dir / f"New Recording {i}.wav"
        if not wav_file.exists():
            print(f"⚠️  File not found: {wav_file}")
            continue

        result = subprocess.run(
            [
                str(whisper_bin),
                "-m",
                str(model),
                "-l",
                language,
                "-f",
                str(wav_file),
                "-bs",
                "5",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",  # <-- this is the fix
            errors="replace",  # replaces undecodable bytes instead of crashing
        )

        lines = [
            line
            for line in result.stdout.splitlines()
            if line.strip().startswith("[") and "]" in line
        ]

        transcriptions[wav_file.name] = lines

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(transcriptions, f, indent=2, ensure_ascii=False)

    print(f"✅ Transcriptions saved to {output_path}")


def compute_metrics():
    gt_path = Path("data/ground_truth.json")
    transcripts_dir = Path("data/samples/transcripts")

    with gt_path.open(encoding="utf-8") as f:
        ground_truth = json.load(f)

    def clean_text(text):
        # Remove timestamps (just in case)
        text = re.sub(r"\[.*?\]", "", text)

        # Normalize unicode accents (é → e, optional)
        text = unicodedata.normalize("NFKC", text)

        # Remove ellipses or artifacts
        text = re.sub(r"[…]", "", text)

        # Remove multiple spaces
        text = re.sub(r"\s+", " ", text)

        text = re.sub(r"[^\w\s]", "", text)  # remove punctuation

        # Strip leading/trailing whitespace
        return text.strip()

    def extract_text(transcription_lines):
        return " ".join(
            re.sub(r"\[.*?\]", "", line).strip() for line in transcription_lines
        )

    # Prepare data
    records = []

    # Evaluate WER and CER for each transcript
    for transcript_file in transcripts_dir.glob("transcriptions_*.json"):
        model_name = transcript_file.stem.replace("transcriptions_", "")
        with transcript_file.open(encoding="utf-8") as f:
            predictions = json.load(f)

        for file_name, ref_text in ground_truth.items():
            if file_name not in predictions:
                continue
            hyp_text = extract_text(predictions[file_name])
            ref_clean = clean_text(ref_text.lower())
            hyp_clean = clean_text(hyp_text.lower())
            records.append(
                {
                    "Model": model_name,
                    "Sample": file_name,
                    "Ref text": ref_clean,
                    "Hyp text": hyp_clean,
                    "WER": wer(ref_clean, hyp_clean),
                    "CER": cer(ref_clean, hyp_clean),
                }
            )

    df = pd.DataFrame(records)
    df.to_csv("results.csv")

    # Compute averages
    df_avg = df.groupby("Model")[["WER", "CER"]].mean().reset_index()

    # Melt for seaborn
    df_melted = df_avg.melt(id_vars="Model", var_name="Metric", value_name="Score")

    # Sort models by WER
    model_order = df_avg.sort_values("WER")["Model"]

    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(data=df_melted, y="Model", x="Score", hue="Metric", order=model_order)
    plt.xlabel("Error Rate (%)")
    plt.title("WER and CER per Model")
    plt.grid(True, axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
