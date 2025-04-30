from pathlib import Path
import subprocess


def convert_pytorch_to_ggml(
    model_dir: Path, openai_whisper_path: Path, output_dir: Path
):
    ggml_model_path = model_dir / "ggml-model.bin"
    if ggml_model_path.exists():
        print(f"GGML model already exists at {ggml_model_path}.")
        return
    subprocess.run(
        [
            "poetry",
            "run",
            "python",
            "whisper_medical/convert-h5-to-ggml.py",  # convert a Hugging Face fine-tuned model (not OpenAI .pt) to GGML format
            str(model_dir.resolve()),
            str(openai_whisper_path.resolve()),
            str(output_dir.resolve()),
        ],
        check=True,
        cwd=str(
            Path(__file__).parent.parent.resolve()
        ),  # root directory of the project
    )
    print(f"Model converted to GGML format and saved to {ggml_model_path}")
