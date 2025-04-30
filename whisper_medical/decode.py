import json
from pathlib import Path
from rapidfuzz import process
from unidecode import unidecode


def normalize(text: str) -> str:
    return unidecode(text.lower().strip())


def flatten_predicted(predicted: dict) -> dict:
    flat = {}
    for fname, lines in predicted.items():
        content = " ".join(line.split("]", 1)[-1].strip() for line in lines)
        flat[fname] = content
    return flat


def project_to_vocab(text: str, vocab: list, threshold: float = 85.0) -> str:
    words = text.split()
    projected = []
    for word in words:
        normalized = normalize(word)
        match, score, _ = process.extractOne(normalized, vocab)
        if score >= threshold:
            projected.append(match)
        else:
            projected.append(word)
    return " ".join(projected)


def correct_transcripts_with_vocab(
    predicted_path: Path,
    vocab_path: Path,
    output_path: Path,
):
    with open(predicted_path) as f:
        predicted_raw = json.load(f)

    with open(vocab_path) as f:
        vocab = json.load(f)

    flat_predicted = flatten_predicted(predicted_raw)

    corrected = {}
    for fname, pred_text in flat_predicted.items():
        corrected[fname] = project_to_vocab(pred_text, vocab)

    with open(output_path, "w") as f:
        json.dump(corrected, f, indent=2, ensure_ascii=False)
