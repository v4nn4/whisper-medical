# whisper-medical

Fine-tuning of OpenAI's Whisper model on French audio, with a focus on medical use cases in urgent care.

## Fine-tuning Results
### Proof of Concept
A first fine-tuning experiment was conducted to demonstrate the potential of domain adaptation:

- **Base model**: openai/whisper-tiny
- **Dataset**: contains 1200 audio samples (~45 min) of synthetic audio recordings of phrases commonly used in urgent care scenarios
- **Training time**: ~10 minutes on Google Colab with T4 TPU

Final training results (after 100 steps):

| Metric                   | Value  |
|---------------------------|--------|
| Training Loss             | 0.002600 |
| Word Error Rate (WER)     | 0.193496 |
| Character Error Rate (CER)| 0.247611 |

The colab notebook used for fine-tuning can be found [here](https://gist.github.com/v4nn4/29d7b5231718250f73daf702f3c65fbc).

### Advanced Fine-Tuning
To be completed. Once more data is available, we'll extend the training and run proper evaluations on held-out data.

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

### Requirements

- Python 3.11
- `git`, `ffmpeg`
- Local clones of:
    - [openai/whisper](https://github.com/openai/whisper) (for conversion to GGML)
    - [ggerganov/whisper.cpp](https://github.com/ggerganov/whisper.cpp) (for lightweight CPU inference)

Make sure to link the cloned repositories by adjusting their path in the `main.py` file.

### Steps

```bash
poetry config virtualenvs.in-project true
poetry install
poetry run python main.py health
```


## Usage
All features are accessible from the command line using Python Fire.

```bash
poetry run python main.py [command]
```

### Available Commands

#### `health`
Check that WAV files and reference transcripts are available.

```bash
poetry run python main.py health
```

#### `prepare`
Convert `.m4a` files to `.wav` (mono, 16kHz).

```bash
poetry run python main.py prepare
```

#### `augment`
Generate augmented WAV files and update the transcripts.

```bash
poetry run python main.py augment
```

#### `download`
Download a Whisper model from Hugging Face and convert it to GGML.

```bash
poetry run python main.py download --model_id qanastek/whisper-small-french-uncased
```

#### `convert`
Convert a Hugging Face Whisper model to GGML manually.

```bash
poetry run python main.py convert --model_id my-model
```

#### `train`
Fine-tune a Whisper model using the local dataset and transcripts.

```bash
poetry run python main.py train --model_id my-finetuned-model
```

Optional:
- `--base_model_id`
- `--data_dir`, `--reference_transcripts_path`, `--models_dir`

#### `evaluate`
Run inference using `whisper.cpp` and compute WER/CER.

```bash
poetry run python main.py evaluate --ggml_model_path models/my-model/ggml-model.bin
```

#### `decode`
Correct transcripts using a known vocabulary file.

```bash
poetry run python main.py decode
```

Optional:
- `--predicted_path`
- `--vocab_path`
- `--output_path`

#### `compute_metrics`
Manually recompute metrics based on transcripts.

```bash
poetry run python main.py compute_metrics
```

## License
MIT