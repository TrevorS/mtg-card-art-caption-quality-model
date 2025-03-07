import json
from dataclasses import dataclass

import torch
from transformers import AutoProcessor, HfArgumentParser

from caption_quality import CaptionQualityConfig, CaptionQualityModel
from utils import get_dataset, get_device

device = get_device()


def main(
    weights_path: str,
    dataset_path: str,
    output_path: str,
) -> None:
    config = CaptionQualityConfig()

    print(f"Loading {config.clip_model_name} model from {weights_path}")
    model = CaptionQualityModel.from_pretrained(
        weights_path,
        config=CaptionQualityConfig(),
    )

    model.eval()
    model.to(device)
    print(model)

    processor = AutoProcessor.from_pretrained(config.clip_model_name)

    print("Loading dataset")
    dataset = get_dataset(dataset_path)

    print("Loaded dataset with", len(dataset), "examples")
    results = []

    for example in dataset:
        inputs = processor(
            text=example["caption"],
            images=example["image"],
            return_tensors="pt",
            padding="max_length",
            max_length=64,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits.squeeze()
        probability = torch.sigmoid(logits).item()

        result = {
            "id": example["id"],
            "caption": example["caption"],
            "probability": probability,
        }

        print(result)
        results.append(result)

    print(f"Saving {len(results)} results to {output_path}")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)


@dataclass
class CaptionQualityCliArgs:
    dataset_path: str
    output_path: str
    weights_path: str = "./mtg-card-art-caption-quality"


if __name__ == "__main__":
    parser = HfArgumentParser(CaptionQualityCliArgs)

    main(**vars(parser.parse_args()))
