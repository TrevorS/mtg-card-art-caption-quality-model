from collections import Counter
from dataclasses import dataclass

import pandas as pd
from transformers import HfArgumentParser

from utils import get_dataset


@dataclass
class CaptionQualityMetricsArguments:
    annotations_path: str = "files/project-6-at-2024-07-15-19-33-e45248ab.json"


def main(args: CaptionQualityMetricsArguments):
    print("Loading dataset")
    dataset = get_dataset(args.annotations_path)
    print("Loaded dataset with", len(dataset), "examples")

    df = dataset.to_pandas()
    analyze(df)


def analyze(annotations: pd.DataFrame) -> pd.DataFrame:
    quality_dist = annotations["quality"].value_counts(normalize=True)
    print("Quality distribution")
    print(quality_dist)

    annotations["caption_length"] = annotations["caption"].apply(len)
    print("Caption length statistics")
    print(annotations["caption_length"].describe())

    all_words = " ".join(annotations["caption"]).lower().split()
    word_counts = Counter(all_words)

    print("Top 20 most common words")
    print(pd.DataFrame(word_counts.most_common(20), columns=["Word", "Frequency"]))

    print("Top 20 least common words")
    print(
        pd.DataFrame(word_counts.most_common()[:-21:-1], columns=["Word", "Frequency"])
    )

    print("Number of unique words:", len(word_counts))
    print("Number of total words:", len(all_words))
    print("Average word frequency:", len(all_words) / len(word_counts))

    corr = annotations["caption_length"].corr(annotations["quality"])
    print(f"Correlation between caption length and quality: {corr:.2f}")

    print("Sample captions with high quality")
    print(
        annotations.sort_values("quality", ascending=False).head(5)[
            ["caption", "quality"]
        ]
    )

    print("Sample captions with low quality")
    print(annotations.sort_values("quality").head(5)[["caption", "quality"]])

    return annotations


if __name__ == "__main__":
    args = HfArgumentParser((CaptionQualityMetricsArguments)).parse_args()

    main(args)
