from dataclasses import dataclass

import evaluate
import numpy as np
from transformers import (
    AutoProcessor,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

import wandb
from caption_quality import CaptionQualityConfig, CaptionQualityModel
from utils import get_dataset, get_device

wandb.require("core")

device = get_device()

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
roc_auc_metric = evaluate.load("roc_auc")


@dataclass
class CaptionQualityModelArguments:
    clip_model_name: str = "google/siglip-so400m-patch14-384"
    freeze_clip: bool = True
    dropout_rate: float = 0.05


@dataclass
class CaptionQualityDataArguments:
    dataset_path: str = "files/project-6-at-2024-07-13-14-32-d556e0fe.json"
    val_size: float = 0.2
    max_train_samples: int | None = None
    max_val_samples: int | None = None


@dataclass
class CaptionQualityTrainingArguments(TrainingArguments):
    auto_find_batch_size: str = "power2"
    eval_steps: int = 50
    eval_strategy: str = "steps"
    learning_rate: float = 5e-5
    logging_steps: int = 10
    metric_for_best_model: str = "roc_auc"
    num_train_epochs: int = 3
    output_dir: str = "mtg-card-art-caption-quality"
    overwrite_output_dir: bool = True
    report_to: str = "wandb"
    run_name: str = "mtg-card-art-caption-quality-run"
    save_strategy: str = "no"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, references = eval_pred
    predictions = (logits > 0).astype(int).flatten()

    # sigmoid to get probabilities
    probs = 1 / (1 + np.exp(-logits.flatten()))

    accuracy = accuracy_metric.compute(predictions=predictions, references=references)[
        "accuracy"
    ]

    f1 = f1_metric.compute(
        predictions=predictions, references=references, average="binary"
    )["f1"]

    precision = precision_metric.compute(
        predictions=predictions, references=references, average="binary"
    )["precision"]

    recall = recall_metric.compute(
        predictions=predictions, references=references, average="binary"
    )["recall"]

    roc_auc = roc_auc_metric.compute(prediction_scores=probs, references=references)[
        "roc_auc"
    ]

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
    }


def main(
    model_args: CaptionQualityModelArguments,
    data_args: CaptionQualityDataArguments,
    training_args: CaptionQualityTrainingArguments,
) -> None:
    print("Loading dataset")
    dataset = get_dataset(data_args.dataset_path)
    print("Loaded dataset with", len(dataset), "examples")

    print(f"Encoding dataset using {model_args.clip_model_name} processor")
    processor = AutoProcessor.from_pretrained(model_args.clip_model_name)

    def preprocess(examples):
        inputs = processor(
            text=examples["caption"],
            images=examples["image"],
            return_tensors="pt",
            max_length=64,
            truncation=True,
            padding="max_length",
        )

        inputs["labels"] = examples["quality"]

        return inputs

    dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Preprocessing dataset",
    )

    dataset = dataset.train_test_split(test_size=data_args.val_size)

    dataset["validation"] = dataset["test"]
    print("Training dataset has", len(dataset["train"]), "examples")
    print("Validation dataset has", len(dataset["validation"]), "examples")

    if data_args.max_train_samples:
        print(f"Limiting training dataset to {data_args.max_train_samples} samples")
        dataset["train"] = dataset["train"].select(range(data_args.max_train_samples))

    if data_args.max_val_samples:
        print(f"Limiting validation dataset to {data_args.max_val_samples} samples")
        dataset["validation"] = dataset["validation"].select(
            range(data_args.max_val_samples)
        )

    print("Creating model")
    config = CaptionQualityConfig(
        clip_model_name=model_args.clip_model_name,
        freeze_clip=model_args.freeze_clip,
        dropout_rate=model_args.dropout_rate,
    )

    model = CaptionQualityModel(config)
    model.to(device)

    print("Creating trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    print("Training model")
    trainer.train()

    print("Saving model")
    trainer.save_model()
    model.config.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    parser = HfArgumentParser(
        (
            CaptionQualityModelArguments,
            CaptionQualityDataArguments,
            CaptionQualityTrainingArguments,
        )
    )

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    main(model_args, data_args, training_args)
