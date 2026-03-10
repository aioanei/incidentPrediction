import os
import numpy as np
from generate_data import generate_dataset
from dataset import prepare_data
from model import build_random_forest, build_gradient_boosting, train_model
from evaluate import (
    print_report,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall,
    threshold_analysis,
)

WINDOW_SIZE = 30      # W: look-back window
HORIZON     = 5       # H: prediction horizon
SEED        = 42


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # generate or load data
    print("Generating synthetic dataset")
    df = generate_dataset(seed=SEED)
    df.to_csv("data/timeseries.csv", index=False)
    print(f"    {len(df)} points, incident rate: {df['is_incident'].mean():.2%}\n")

    # build sliding-window features
    print("Creating sliding windows")
    X_train, X_test, y_train, y_test = prepare_data(
        df,
        window_size=WINDOW_SIZE,
        horizon=HORIZON,
        seed=SEED,
    )

    # train models
    models = {
        "RandomForest": build_random_forest(seed=SEED),
        "GradientBoosting": build_gradient_boosting(seed=SEED),
    }

    for name, pipeline in models.items():
        print(f"\nTraining {name}")
        train_model(pipeline, X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        # evaluate
        print_report(y_test, y_pred, y_prob, model_name=name)

        plot_confusion_matrix(
            y_test, y_pred, name,
            save_path=f"results/{name}_confusion_matrix.png",
        )
        plot_roc_curve(
            y_test, y_prob, name,
            save_path=f"results/{name}_roc.png",
        )
        plot_precision_recall(
            y_test, y_prob, name,
            save_path=f"results/{name}_pr_curve.png",
        )
        threshold_analysis(
            y_test, y_prob, name,
            save_path=f"results/{name}_threshold.png",
        )

    print("\n all done")
    print("Check the results/ folder for plots.")


if __name__ == "__main__":
    main()
