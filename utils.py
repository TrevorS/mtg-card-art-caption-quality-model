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
        if "data" in example:
            example = example["data"]

        scryfall_id = example["id"]

        image = example["card_art_uri"]
        image = Image.open(requests.get(image, stream=True).raw).convert("RGB")

        caption = example["florence_more_detailed_caption"]
        annotations = example.get("annotations", [])

        if len(annotations) == 0:
            accuracy = None
            creativity = None

        else:
            annotations = annotations[0]["result"]

            accuracy = [
                annotation
                for annotation in annotations
                if annotation["from_name"] == "visual_accuracy"
            ][0]["value"]["rating"]

            creativity = [
                annotation
                for annotation in annotations
                if annotation["from_name"] == "creativity"
            ][0]["value"]["rating"]

        return {
            "id": scryfall_id,
            "image": image,
            "caption": caption,
            "accuracy": accuracy,
            "creativity": creativity,
        }

    dataset = dataset.map(preprocess)

    if filter_no_labels:
        dataset = dataset.filter(
            lambda x: x["accuracy"] is not None and x["creativity"] is not None
        )

    cols_to_drop = list(
        set(dataset.column_names) - {"id", "image", "caption", "accuracy", "creativity"}
    )

    dataset = dataset.remove_columns(cols_to_drop)

    return dataset
