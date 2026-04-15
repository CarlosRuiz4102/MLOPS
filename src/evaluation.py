import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def build_results_table(results: list[dict], split: str = "val") -> pd.DataFrame:
    return (
        pd.DataFrame(results)
        .query("split == @split")
        .sort_values(["macro_f1", "balanced_accuracy", "accuracy"], ascending=False)
        .reset_index(drop=True)
    )
