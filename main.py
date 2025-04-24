import fire

from bisom.prepare import convert_m4a_to_wav
from bisom.evaluate import full_transcribe, compute_metrics
from bisom.download import download_hf_model
from bisom.train import finetune_hf_model


class Cli:
    def prepare(self):
        convert_m4a_to_wav()

    def download(self, model_id: str = "qanastek/whisper-small-french-uncased"):
        download_hf_model(model_id)

    def train(self, model_id: str = "openai/whisper-tiny"):
        finetune_hf_model(model_id)

    def evaluate(self):
        full_transcribe()
        compute_metrics()

    def export(self):
        pass


if __name__ == "__main__":
    fire.Fire(Cli)
