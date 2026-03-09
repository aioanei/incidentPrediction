import os
import numpy as np
import pandas as pd

SEED = 42
N_POINTS = 10000          # total time steps
INCIDENT_PROB = 0.03       # probability of starting an incident at any step
INCIDENT_LEN_RANGE = (5, 15)  # how long an incident lasts (steps)
SPIKE_MAGNITUDE = (2.0, 4.0)  # how much the metric jumps during incident


def make_normal_signal(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n)
    seasonal = 0.5 * np.sin(2 * np.pi * t / 200)   # slow cycle
    noise = rng.normal(0, 0.15, size=n)
    trend = 0.0001 * t                               # tiny upward drift
    return seasonal + noise + trend


def inject_incidents(signal: np.ndarray, rng: np.random.Generator):
    n = len(signal)
    labels = np.zeros(n, dtype=int)
    i = 0
    while i < n:
        if rng.random() < INCIDENT_PROB and labels[i] == 0:
            length = rng.integers(*INCIDENT_LEN_RANGE)
            mag = rng.uniform(*SPIKE_MAGNITUDE)
            end = min(i + length, n)
            signal[i:end] += mag + rng.normal(0, 0.3, size=end - i)
            labels[i:end] = 1
            i = end + 10       # cooldown so incidents don't overlap
        else:
            i += 1
    return signal, labels


def generate_dataset(n_points: int = N_POINTS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    signal = make_normal_signal(n_points, rng)
    signal, labels = inject_incidents(signal, rng)

    df = pd.DataFrame({
        "timestamp": np.arange(n_points),
        "metric_value": signal,
        "is_incident": labels,
    })
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_dataset()
    df.to_csv("data/timeseries.csv", index=False)

    inc_pct = df["is_incident"].mean() * 100
    print(f"Generated {len(df)} data points  |  incidents: {inc_pct:.1f}%")
    print(f"Saved to data/timeseries.csv")
