from pathlib import Path
import json
from typing import Any, Dict, List
import torch
from datasets import Dataset, Audio, disable_caching
from transformers.models.whisper import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from transformers.trainer_seq2seq import Seq2SeqTrainer
from jiwer import wer, cer


def compute_metrics(pred: Any, processor: WhisperProcessor) -> Dict[str, float]:
    logits = (
        pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    )
    pred_ids = torch.argmax(torch.tensor(logits), dim=-1)
    label_ids = pred.label_ids
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    return {
        "wer": wer(label_str, pred_str),
        "cer": cer(label_str, pred_str),
    }


def whisper_data_collator(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    input_features = torch.stack([torch.tensor(x["input_features"]) for x in batch])
    label_list = [torch.tensor(x["labels"], dtype=torch.long) for x in batch]
    labels = torch.nn.utils.rnn.pad_sequence(
        label_list, batch_first=True, padding_value=-100
    )
    return {"input_features": input_features, "labels": labels}


def finetune_hf_model(
    model_id: str,
    data_dir: Path = Path(R"data/samples/wav"),
    gt_path: Path = Path(R"data/ground_truth.json"),
    output_dir: Path = Path("./output"),
):
    disable_caching()

    with gt_path.open(encoding="utf-8") as f:
        transcripts = json.load(f)

    samples = [
        {"audio": str(p), "sentence": transcripts[p.name]}
        for p in data_dir.glob("*.wav")
        if p.name in transcripts
    ]

    dataset = Dataset.from_list(samples).cast_column(
        "audio", Audio(sampling_rate=16000)
    )

    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    assert processor is WhisperProcessor

    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=32,
        per_device_eval_batch_size=16,
        learning_rate=1e-5,
        max_steps=100,
        warmup_steps=5,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        seed=42,
        lr_scheduler_type="linear",
        report_to="none",
        fp16=torch.cuda.is_available(),
        optim="adamw_torch",
        eval_strategy="steps",
        eval_steps=10,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=whisper_data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor),
    )

    trainer.train()

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
