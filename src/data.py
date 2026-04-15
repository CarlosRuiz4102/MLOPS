from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedShuffleSplit


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_data_paths() -> tuple[Path, Path]:
    root = get_project_root()
    return root / "data" / "mitbih_train.csv", root / "data" / "mitbih_test.csv"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_path, test_path = get_data_paths()
    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    feature_cols = [f"t_{i}" for i in range(train_df.shape[1] - 1)]
    columns = feature_cols + ["label"]
    train_df.columns = columns
    test_df.columns = columns

    return train_df, test_df, feature_cols


def prepare_splits(test_size: float = 0.15, random_state: int = 42, fast_run: bool = False) -> dict:
    train_df, test_df, feature_cols = load_data()

    X_train_full = train_df[feature_cols].values.astype(np.float32)
    y_train_full = train_df["label"].values.astype(int)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["label"].values.astype(int)

    if fast_run:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=3000, random_state=random_state)
        idx_sub, _ = next(sss.split(X_train_full, y_train_full))
        X_train_full = X_train_full[idx_sub]
        y_train_full = y_train_full[idx_sub]

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=test_size,
        random_state=random_state,
        stratify=y_train_full,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}

    return {
        "feature_cols": feature_cols,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,
        "X_train_cnn": X_train[..., np.newaxis],
        "X_val_cnn": X_val[..., np.newaxis],
        "X_test_cnn": X_test[..., np.newaxis],
        "class_weight_dict": class_weight_dict,
        "scaler": scaler,
        "train_shape": train_df.shape,
        "test_shape": test_df.shape,
    }
