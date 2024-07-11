import evaluate
import torch
import numpy as np
from transformers import (
    AutoProcessor,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from caption_quality import CaptionQualityConfig, CaptionQualityModel
from utils import get_dataset, get_device

accuracy_metric = evaluate.load("accuracy")
device = get_device()


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    num_classes = logits.shape[1] // 2

    accuracy_logits, creativity_logits = (
        logits[:, :num_classes],
        logits[:, num_classes:],
    )
    accuracy_labels, creativity_labels = labels[:, 0], labels[:, 1]

    accuracy_preds = np.argmax(accuracy_logits, axis=-1)
    creativity_preds = np.argmax(creativity_logits, axis=-1)

    accuracy_correct = (accuracy_preds == accuracy_labels).mean()
    creativity_correct = (creativity_preds == creativity_labels).mean()

    return {
        "accuracy": accuracy_correct,
        "creativity": creativity_correct,
        "mean_accuracy": (accuracy_correct + creativity_correct) / 2,
    }


def main(
    dataset_path: str,
    val_size: float = 0.2,
    output_dir: str = "mtg-card-art-caption-quality",
    logs_dir: str = "logs",
) -> None:
    print("Creating model")
    config = CaptionQualityConfig()
    model = CaptionQualityModel(config)
    model.to(device)

    print("Loading dataset")
    dataset = get_dataset(dataset_path)
    print("Loaded dataset with", len(dataset), "examples")

    print("Encoding dataset")
    processor = AutoProcessor.from_pretrained(config.clip_model_name)

    def encode(example):
        inputs = processor(
            text=example["caption"],
            images=example["image"],
            return_tensors="pt",
            max_length=64,
            truncation=True,
            padding="max_length",
        )

        return {
            "pixel_values": inputs.pixel_values.squeeze(0).to(torch.float32),
            "input_ids": inputs.input_ids.squeeze(0).to(torch.long),
            "labels": torch.tensor(
                [example["accuracy"], example["creativity"]], dtype=torch.long
            ),
        }

    dataset = dataset.map(encode)
    dataset = dataset.train_test_split(test_size=val_size)

    dataset["validation"] = dataset["test"]
    print("Training dataset has", len(dataset["train"]), "examples")
    print("Validation dataset has", len(dataset["validation"]), "examples")

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="mean_accuracy",
        eval_steps=10,
        logging_dir=logs_dir,
        logging_steps=10,
        save_steps=10,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        report_to="wandb",
    )

    print("Creating trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    print("Initial evaluation")
    initial_metrics = trainer.evaluate()
    print(f"Initial metrics: {initial_metrics}")

    print("Training model")
    trainer.train()

    print("Final evaluation")
    final_metrics = trainer.evaluate()
    print(f"Final metrics: {final_metrics}")

    print("Saving model")
    trainer.save_model()
    model.config.save_pretrained(output_dir)


if __name__ == "__main__":
    main(
        dataset_path="project-4-at-2024-07-09-18-44-77422fbd.json",
    )
