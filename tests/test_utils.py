import numpy as np

from src.features import extract_time_features


def test_extract_time_features_shape():
    X = np.array(
        [
            [0.0, 1.0, 0.5, 0.0],
            [1.0, 0.5, 0.25, 0.0],
        ],
        dtype=np.float32,
    )

    features = extract_time_features(X)

    assert features.shape[0] == 2
    assert "mean" in features.columns
    assert "energy" in features.columns
