from pathlib import Path
import librosa
import soundfile as sf


def convert_m4a_to_wav(
    input_dir: Path = Path("data/samples/m4a"),
    output_dir: Path = Path("data/samples/wav"),
):
    output_dir.mkdir(parents=True, exist_ok=True)
    for m4a_file in input_dir.rglob("*.m4a"):
        y, sr = librosa.load(m4a_file, sr=16000, mono=True)

        relative_path = m4a_file.relative_to(input_dir).with_suffix(".wav")
        target_path = output_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        sf.write(target_path, y, 16000)
