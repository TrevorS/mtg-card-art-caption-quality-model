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


def get_dataset(dataset_path: str) -> Dataset:
    dataset = Dataset.from_json(dataset_path)

    def preprocess(example):
        scryfall_id = example["data"]["id"]

        image = example["data"]["card_art_uri"]
        image = Image.open(requests.get(image, stream=True).raw).convert("RGB")

        caption = example["data"]["florence_more_detailed_caption"]
        annotations = example["annotations"][0]["result"]

        if len(annotations) != 2:
            accuracy = None
            creativity = None

        else:
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

    dataset = dataset.map(preprocess).filter(
        lambda x: x["accuracy"] is not None and x["creativity"] is not None
    )

    cols_to_drop = list(
        set(dataset.column_names) - {"id", "image", "caption", "accuracy", "creativity"}
    )

    dataset = dataset.remove_columns(cols_to_drop)

    return dataset
