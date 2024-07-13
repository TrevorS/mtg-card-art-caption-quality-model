from dataclasses import dataclass

import evaluate
import numpy as np
from transformers import (
    AutoProcessor,
    HfArgumentParser,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
)

from caption_quality import CaptionQualityConfig, CaptionQualityModel
from utils import get_dataset, get_device

device = get_device()

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
roc_auc_metric = evaluate.load("roc_auc")


@dataclass
class CaptionQualityModelArguments:
    clip_model_name: str = "google/siglip-so400m-patch14-384"
    freeze_clip: bool = False
    dropout_rate: float = 0.05


@dataclass
class CaptionQualityDataArguments:
    dataset_path: str = "files/project-6-at-2024-07-13-14-32-d556e0fe.json"
    val_size: float = 0.2
    max_train_samples: int | None = None
    max_val_samples: int | None = None


@dataclass
class CaptionQualityTrainingArguments(TrainingArguments):
    logging_dir: str = "logs"
    output_dir: str = "mtg-card-art-caption-quality"
    overwrite_output_dir: bool = True
    eval_strategy: str = "steps"
    logging_strategy: str = "steps"
    save_strategy: str = "steps"
    eval_steps: int = 50
    logging_steps: int = 10
    save_steps: int = 50
    metric_for_best_model: str = "roc_auc"
    learning_rate: float = 1e-5
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = 0.1
    auto_find_batch_size: str = "power2"
    num_train_epochs: int = 1
    weight_decay: float = 0.001
    report_to: str = "wandb"


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


class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)

        if state.is_local_process_zero:
            print(f"Step: {state.global_step}")
            print(logs)


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
        callbacks=[LoggingCallback()],
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
