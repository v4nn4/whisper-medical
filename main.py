from pathlib import Path
import fire

from whisper_medical.convert import convert_pytorch_to_ggml
from whisper_medical.decode import correct_transcripts_with_vocab
from whisper_medical.health import health_check
from whisper_medical.prepare import convert_m4a_to_wav_16khz_mono
from whisper_medical.evaluate import whisper_full_transcribe, compute_metrics
from whisper_medical.download import download_hf_model
from whisper_medical.train import augment_dataset, finetune_hf_model


class Paths:
    BASE = Path("data")
    DATASET = BASE / "romain-30"
    SAMPLES = DATASET / "samples"
    M4A = SAMPLES / "m4a"
    WAV = SAMPLES / "wav"
    AUGMENTED = WAV / "augmented"

    TRANSCRIPTS = DATASET / "transcripts"
    REFERENCE_TRANSCRIPTS = TRANSCRIPTS / "reference_transcripts.json"
    PREDICTED_TRANSCRIPTS = TRANSCRIPTS / "predicted_transcripts.json"

    MODELS = Path("models")
    WHISPER_CLI = Path("../../Repos/whisper.cpp/build/bin/whisper-cli")
    OPENAI_WHISPER_REPO = Path("../../Repos/whisper")

    METRICS = Path("metrics")


class Cli:
    def health(
        self,
        wav_path: Path = Paths.WAV,
        reference_transcripts_path: Path = Paths.REFERENCE_TRANSCRIPTS,
    ):
        health_check(wav_path, reference_transcripts_path)

    def prepare(
        self,
        input_dir: Path = Paths.M4A,
        output_dir: Path = Paths.WAV,
    ):
        convert_m4a_to_wav_16khz_mono(input_dir, output_dir)

    def augment(
        self,
        input_dir: Path = Paths.WAV,
        output_dir: Path = Paths.AUGMENTED,
        reference_transcripts_path: Path = Paths.REFERENCE_TRANSCRIPTS,
        output_transcripts_path: Path = Paths.TRANSCRIPTS
        / "reference_transcripts_augmented.json",
        num_augmentations: int = 5,
    ):
        augment_dataset(
            input_dir,
            output_dir,
            reference_transcripts_path,
            output_transcripts_path,
            num_augmentations,
        )

    def download(
        self,
        model_id: str = "qanastek/whisper-small-french-uncased",
        models_dir: Path = Paths.MODELS,
    ):
        download_hf_model(model_id, models_dir)
        model_dir = models_dir / model_id
        convert_pytorch_to_ggml(model_dir, Paths.OPENAI_WHISPER_REPO, model_dir)

    def convert(
        self,
        model_id: str,
        models_dir: Path = Paths.MODELS,
        openai_whisper_path: Path = Paths.OPENAI_WHISPER_REPO,
    ):
        model_dir = models_dir / model_id
        convert_pytorch_to_ggml(model_dir, openai_whisper_path, model_dir)

    def train(
        self,
        model_id: str,
        base_model_id: str = "openai/whisper-tiny",
        data_dir: Path = Paths.WAV,
        reference_transcripts_path: Path = Paths.REFERENCE_TRANSCRIPTS,
        models_dir: Path = Paths.MODELS,
    ):
        model_output_path = finetune_hf_model(
            model_id, base_model_id, data_dir, reference_transcripts_path, models_dir
        )
        print(f"Model saved to {model_output_path}")

    def evaluate(
        self,
        ggml_model_path: Path,
        whisper_build_path: Path = Paths.WHISPER_CLI,
        wav_dir: Path = Paths.WAV,
        output_dir: Path = Paths.TRANSCRIPTS,
    ):
        # Transcribe the audio files using local whisper.cpp
        whisper_full_transcribe(
            model_path=Path(ggml_model_path),
            whisper_build_path=whisper_build_path,
            input_dir=wav_dir,
            output_dir=output_dir,
        )
        # Compute the metrics
        compute_metrics(
            reference_transcripts_path=Paths.REFERENCE_TRANSCRIPTS,
            transcripts_dir=Paths.TRANSCRIPTS,
            metrics_dir=Paths.METRICS,
        )

    def decode(
        self,
        predicted_path: Path = Paths.PREDICTED_TRANSCRIPTS,
        vocab_path: Path = Paths.TRANSCRIPTS / "expected_vocab.json",
        output_path: Path = Paths.TRANSCRIPTS / "corrected_transcripts.json",
    ):
        correct_transcripts_with_vocab(predicted_path, vocab_path, output_path)
        print(f"Corrected transcripts written to {output_path}")

    def compute_metrics(
        self,
    ):
        compute_metrics(
            reference_transcripts_path=Paths.REFERENCE_TRANSCRIPTS,
            transcripts_dir=Paths.TRANSCRIPTS,
            metrics_dir=Paths.METRICS,
        )


if __name__ == "__main__":
    fire.Fire(Cli)
