"""HotpotQA official answer metrics, with standard-library dependencies only.

Algorithm: https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py
The tests compare against the official scorer shipped with the experiment record.
"""

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    text = "".join(
        character for character in text.lower() if character not in string.punctuation
    )
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def f1_score(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    prediction, ground_truth = (
        normalize_answer(prediction),
        normalize_answer(ground_truth),
    )
    if prediction != ground_truth and (
        prediction in {"yes", "no", "noanswer"}
        or ground_truth in {"yes", "no", "noanswer"}
    ):
        return 0.0, 0.0, 0.0
    predicted, gold = prediction.split(), ground_truth.split()
    overlap = sum((Counter(predicted) & Counter(gold)).values())
    if not overlap:
        return 0.0, 0.0, 0.0
    precision, recall = overlap / len(predicted), overlap / len(gold)
    return 2 * precision * recall / (precision + recall), precision, recall


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)
