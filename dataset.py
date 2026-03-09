import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def create_sliding_windows(
    metric: np.ndarray,
    labels: np.ndarray,
    window_size: int = 30,
    horizon: int = 5,
):
    n = len(metric)
    X, y = [], []
    for i in range(window_size, n - horizon):
        window = metric[i - window_size : i]
        # label = 1 if there is at least one incident in the next H steps
        future_labels = labels[i : i + horizon]
        target = int(future_labels.any())
        X.append(window)
        y.append(target)

    return np.array(X), np.array(y)


def add_handcrafted_features(X_windows: np.ndarray) -> np.ndarray:
    feats = []
    for w in X_windows:
        feat = [
            w.mean(),
            w.std(),
            w.max(),
            w.min(),
            w[-1] - w[0],                  # slope over window
            (w[-5:].mean() - w[:5].mean()), # recent vs. old mean
            np.percentile(w, 90),
        ]
        feats.append(feat)
    return np.hstack([X_windows, np.array(feats)])


def prepare_data(
    df: pd.DataFrame,
    window_size: int = 30,
    horizon: int = 5,
    test_ratio: float = 0.2,
    add_features: bool = True,
    seed: int = 42,
):
    metric = df["metric_value"].values
    labels = df["is_incident"].values

    X, y = create_sliding_windows(metric, labels, window_size, horizon)

    if add_features:
        X = add_handcrafted_features(X)

    # chronological split – last 20 % is the test set
    split_idx = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Window size W={window_size}, Horizon H={horizon}")
    print(f"Train samples: {len(X_train)}  |  Test samples: {len(X_test)}")
    print(f"Train incident rate: {y_train.mean():.2%}")
    print(f"Test  incident rate: {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test
