import requests
import torch
from datasets import Dataset
from PIL import Image


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_dataset(dataset_path: str, filter_no_labels=False) -> Dataset:
    dataset = Dataset.from_json(dataset_path)

    def preprocess(example):
        annotations = example.get("annotations", [])

        if "data" in example:
            example = example["data"]

        scryfall_id = example["id"]
        caption = example["florence_more_detailed_caption"]

        image = example["card_art_uri"]
        image = Image.open(requests.get(image, stream=True).raw).convert("RGB")

        if len(annotations) == 0:
            quality = None

        else:
            quality = annotations[0]["result"][0]["value"]["choices"][0]
            quality = 1 if quality == "High Quality" else 0

        return {
            "id": scryfall_id,
            "image": image,
            "caption": caption,
            "quality": quality,
        }

    dataset = dataset.map(preprocess)

    if filter_no_labels:
        dataset = dataset.filter(lambda x: x["quality"] is not None)

    cols_to_drop = list(
        set(dataset.column_names) - {"id", "image", "caption", "quality"}
    )

    dataset = dataset.remove_columns(cols_to_drop)

    return dataset
