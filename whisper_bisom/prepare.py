from pathlib import Path
import librosa
import soundfile as sf
import wave
import contextlib


def total_wav_duration_minutes(folder_path: Path) -> float:
    total_seconds = 0.0
    for wav_file in folder_path.rglob("*.wav"):
        with contextlib.closing(wave.open(str(wav_file), "r")) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            total_seconds += frames / float(rate)
    return total_seconds / 60


def convert_m4a_to_wav_16khz_mono(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    converted_files = 0

    for m4a_file in input_dir.rglob("*.m4a"):
        relative_path = m4a_file.relative_to(input_dir).with_suffix(".wav")
        target_path = output_dir / relative_path
        if target_path.exists():
            continue

        y, _ = librosa.load(m4a_file, sr=16000, mono=True)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(target_path, y, 16000)
        converted_files += 1

    if converted_files:
        print(f"Successfully converted {converted_files} file(s) to WAV format.")
    else:
        print("All files were already converted; no action taken.")

    minutes = total_wav_duration_minutes(output_dir)
    print(f"Total audio length: {minutes:.2f} minutes.")
