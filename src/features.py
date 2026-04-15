import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


def extract_time_features(X: np.ndarray) -> pd.DataFrame:
    X = np.asarray(X, dtype=np.float32)
    features = pd.DataFrame(
        {
            "mean": X.mean(axis=1),
            "std": X.std(axis=1),
            "min": X.min(axis=1),
            "max": X.max(axis=1),
            "median": np.median(X, axis=1),
            "q25": np.quantile(X, 0.25, axis=1),
            "q75": np.quantile(X, 0.75, axis=1),
            "ptp": np.ptp(X, axis=1),
            "energy": np.sum(X**2, axis=1),
            "mean_abs": np.mean(np.abs(X), axis=1),
            "argmax": np.argmax(X, axis=1),
            "argmin": np.argmin(X, axis=1),
            "signal_length": np.sum(np.abs(np.diff(X, axis=1)), axis=1),
            "zero_crossings": np.sum(np.diff(np.signbit(X), axis=1), axis=1),
            "skew": skew(X, axis=1, bias=False, nan_policy="omit"),
            "kurtosis": kurtosis(X, axis=1, fisher=True, bias=False, nan_policy="omit"),
        }
    )
    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_feature_views(data: dict) -> dict:
    data = dict(data)
    data["X_train_feat"] = extract_time_features(data["X_train"])
    data["X_val_feat"] = extract_time_features(data["X_val"])
    data["X_test_feat"] = extract_time_features(data["X_test"])
    return data
