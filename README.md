# Incident Prediction from Time-Series Metrics

Predicting whether an incident (anomaly/spike) will occur in the next **H** time steps, given the previous **W** steps of a monitoring metric.

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

This generates the synthetic dataset, trains two models, prints evaluation metrics, and saves plots to `results/`.

There's also a Jupyter notebook (`notebook.ipynb`) that walks through everything step by step with inline plots.

## Problem Formulation

Given a univariate time series with binary incident labels:

- **Input (X):** a sliding window of the last `W = 30` metric values (+ 7 hand-crafted statistical features)
- **Output (y):** binary label — `1` if **any** of the next `H = 5` time steps is an incident, `0` otherwise

This converts the time-series forecasting problem into a standard tabular binary classification task.

### Why this formulation?
- Simple to implement and reason about
- Compatible with any classifier (no need for sequence models)
- The horizon H gives advance notice before the incident actually starts

## Dataset

Synthetic time series (10,000 points) that simulates a server metric (e.g., CPU usage):
- **Normal behavior:** noisy sine wave with slight upward trend
- **Incidents:** random spikes of magnitude 2–4x above normal, lasting 5–15 steps
- **Incident rate:** ~6–8% of all time steps

I chose synthetic data because it lets me control the signal-to-noise ratio and incident frequency. The concepts transfer directly to real monitoring data.

## Models

| Model | Why I chose it |
|---|---|
| **Random Forest** | Strong baseline for tabular data, handles class imbalance with `class_weight='balanced'`, fast to train |
| **Gradient Boosting** | Usually outperforms RF on structured data, good with imbalanced classes |

Both are from scikit-learn. I also add a `StandardScaler` in the pipeline (not strictly needed for trees, but good practice).

### Hand-crafted features
On top of the 30 raw window values, I add:
- mean, std, max, min of the window
- slope (last value − first value)
- recent vs. old mean (last 5 values vs. first 5)
- 90th percentile

These help the tree models pick up on distributional changes without needing deep feature learning.

## Evaluation

- **Chronological train/test split** (80/20) — no shuffling, to avoid data leakage
- **Metrics:** Precision, Recall, F1-score, ROC-AUC
- **Confusion matrix** — to see false positives vs. missed incidents
- **Threshold sweep** — shows how precision/recall trade off at different alert thresholds

### Why these metrics?
In an alerting system:
- **High precision** = fewer false alarms (important for on-call fatigue)
- **High recall** = fewer missed incidents (important for reliability)
- The **threshold sweep** is especially relevant because in production you'd tune the threshold based on operational constraints, not just use 0.5

## Project Structure

```
├── generate_data.py    # synthetic dataset generation
├── dataset.py          # sliding-window creation + feature engineering
├── model.py            # model definitions (RF, Gradient Boosting)
├── evaluate.py         # metrics, plots, threshold analysis
├── main.py             # run everything end-to-end
├── notebook.ipynb      # interactive walkthrough with plots
├── requirements.txt
└── results/            # saved plots (created by main.py)
```

## Limitations

- Synthetic data has a very clear signal — real incidents are subtler
- Single metric only — production systems have many correlated metrics
- No temporal modeling (we flatten the window) — an LSTM or Transformer could capture order dependencies
- Class imbalance would be much worse in real data (~0.1% incident rate)
- Only one train/test split — walk-forward cross-validation would be more rigorous

## Possible Extensions for Production

1. **Multi-metric input:** concatenate windows from CPU, memory, latency, error rate, etc.
2. **Online retraining:** periodically update the model as the system evolves
3. **Alert cooldown:** suppress repeated alerts within a configurable window
4. **Threshold tuning from SLOs:** set the alert threshold to meet a target false-alarm rate
5. **Explainability:** use SHAP values to tell on-call engineers *why* the model is alerting
