# whisper-bisom

Fine-tuning of OpenAI's Whisper model on French audio, with a focus on medical use cases in urgent care.

## Fine-tuning Results
### Proof of Concept
A first fine-tuning experiment was conducted to demonstrate the potential of domain adaptation:

- **Base model**: openai/whisper-tiny
- **Dataset**: contains 30 audio samples (approximately 3 minutes total) featuring clear pronunciations of phrases commonly used in urgent care scenarios
- **Training time**: ~10 minutes on Google Colab

Final training results (after 100 steps):

| Metric                   | Value  |
|---------------------------|--------|
| Training Loss             | 0.0029 |
| Word Error Rate (WER)     | 0.1033 |
| Character Error Rate (CER)| 0.3946 |

> **Note**: This proof-of-concept used a very small dataset without a train/test split. Metrics were computed on the training set only.

The colab notebook used for fine-tuning can be found [here](https://colab.research.google.com/gist/v4nn4/195444fd959db1bef4e09cd08ee3a85d/whisper-bisom-fine-tuning-phase-1.ipynb).

### Advanced Fine-Tuning
To be completed. Once more data is available, we'll extend the training and run proper evaluations on held-out data.

## Installation
This project uses [Poetry](https://python-poetry.org/), a modern tool for dependency management and packaging in Python. It helps ensure that all dependencies are locked and reproducible across machines.

Make sure you are using Python 3.11. If you're unfamiliar with virtual environments, Poetry handles this for you automatically.

### Steps

```bash
# Install Poetry (if you don't have it)
curl -sSL https://install.python-poetry.org | python3 -

# Clone the repository
git clone https://github.com/.../whisper-bisom.git
cd whisper-bisom

# Install dependencies
poetry install

# Run a first test command (e.g., health check)
poetry run python main.py health
```

This project also depends on both:

- [whisper](https://github.com/openai/whisper) (for conversion to GGML)
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) (for lightweight CPU inference)

You must clone those repositories locally and modify the respective paths in the `main.py` file.

## Usage
All features are accessible from the command line using Python Fire.

```bash
poetry run python main.py [command]
```

### Available Commands

- `health`: Quick check to ensure required folders and sample files are in place.
- `prepare`: Converts .m4a files to .wav format (16kHz, mono), ready for training.
- `download [model_id]`: Downloads a Whisper model from Hugging Face (default: qanastek/whisper-small-french-uncased) and prepares it for use.
- `train [model_id]`: Fine-tunes a Whisper model on your local dataset (default base: openai/whisper-tiny).
- `evaluate`: Transcribes the dataset and computes evaluation metrics (WER and CER).
- `export`: Placeholder for exporting the fine-tuned model (e.g., to whisper.cpp) — coming soon.

## Notes
This is a quick proof-of-concept. There are no unit tests, as fine-tuning requires large files and long runtimes that are unsuitable for test environments.

The goal is to validate that domain-specific fine-tuning improves transcription quality — and our first results are promising.


## License
MIT