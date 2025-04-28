from pathlib import Path
import json
from jiwer import wer, cer
import re
import unicodedata
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess


def whisper_full_transcribe(
    model_path: Path,
    whisper_build_path: Path,
    input_dir: Path,
    output_dir: Path,
):
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    predicted_transcripts = {}

    for i in range(1, 32):
        wav_file = input_dir / f"New Recording {i}.wav"
        if not wav_file.exists():
            print(f"⚠️  File not found: {wav_file}")
            continue

        result = subprocess.run(
            [
                str(whisper_build_path.resolve()),
                "-m",
                str(model_path.resolve()),
                "-l",
                "fr",
                "-f",
                str(wav_file.resolve()),
                "-bs",
                "5",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",  # replaces undecodable bytes instead of crashing
            cwd=str(
                Path(__file__).parent.parent.resolve()
            ),  # root directory of the project
        )

        lines = [
            line
            for line in result.stdout.splitlines()
            if line.strip().startswith("[") and "]" in line
        ]

        predicted_transcripts[wav_file.name] = lines

    predicted_transcripts_path = (
        output_dir / f"predicted_transcripts_{model_path.stem}.json"
    )
    with predicted_transcripts_path.open("w", encoding="utf-8") as f:
        json.dump(predicted_transcripts, f, indent=2, ensure_ascii=False)

    print(f"✅ Transcriptions saved to {str(predicted_transcripts_path)}")


def compute_metrics(
    reference_transcripts_path: Path, transcripts_dir: Path, metrics_dir: Path
):
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with reference_transcripts_path.open(encoding="utf-8") as f:
        reference_transcripts = json.load(f)

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

    # Evaluate WER and CER for each transcript file
    for transcript_file in transcripts_dir.glob("predicted_transcripts_*.json"):
        model_name = transcript_file.stem.replace("predicted_transcripts_", "")
        with transcript_file.open(encoding="utf-8") as f:
            predictions = json.load(f)

        for file_name, ref_text in reference_transcripts.items():
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
    df.to_csv(metrics_dir / "results.csv")

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
