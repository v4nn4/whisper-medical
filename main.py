from pathlib import Path
import fire

from whisper_bisom.convert import convert_pytorch_to_ggml
from whisper_bisom.health import health_check
from whisper_bisom.prepare import convert_m4a_to_wav_16khz_mono
from whisper_bisom.evaluate import whisper_full_transcribe, compute_metrics
from whisper_bisom.download import download_hf_model
from whisper_bisom.train import finetune_hf_model


class Paths:
    BASE = Path("data")
    SAMPLES = BASE / "samples"
    M4A = SAMPLES / "m4a"
    WAV = SAMPLES / "wav"

    TRANSCRIPTS = BASE / "transcripts"
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

    def download(
        self,
        model_id: str = "qanastek/whisper-small-french-uncased",
        models_dir: Path = Paths.MODELS,
    ):
        download_hf_model(model_id, models_dir)
        model_dir = models_dir / model_id
        convert_pytorch_to_ggml(model_dir, Paths.OPENAI_WHISPER_REPO, model_dir)

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

    def export(self):
        pass


if __name__ == "__main__":
    fire.Fire(Cli)
