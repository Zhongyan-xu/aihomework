"""
Train several classifiers on the UCI/Kaggle Sonar dataset and report accuracy,
precision, recall, and confusion matrices. Outputs plots into outputs/.
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv"
OUTPUT_DIR = pathlib.Path("outputs")


def load_sonar() -> pd.DataFrame:
    """
    Load the Sonar dataset from a public mirror.
    The original dataset is the same as on Kaggle and UCI.
    """
    df = pd.read_csv(DATA_URL, header=None)
    feature_cols = [f"f{i:02d}" for i in range(df.shape[1] - 1)]
    df.columns = feature_cols + ["target"]
    return df


@dataclass
class ModelResult:
    name: str
    accuracy: float
    precision: float
    recall: float
    y_true: np.ndarray
    y_pred: np.ndarray


def build_models() -> Dict[str, Pipeline]:
    """
    Create the models with light tuning. All models share the same preprocessing:
    standardization of the 60 numeric features.
    """
    scaler = StandardScaler()
    models: Dict[str, Pipeline] = {
        "knn": Pipeline(
            steps=[
                ("scaler", scaler),
                ("model", KNeighborsClassifier(n_neighbors=7, weights="distance", p=2)),
            ]
        ),
        "gaussian_nb": Pipeline(
            steps=[
                ("scaler", scaler),
                ("model", GaussianNB()),
            ]
        ),
        "decision_tree": Pipeline(
            steps=[
                ("scaler", scaler),
                ("model", DecisionTreeClassifier(max_depth=6, min_samples_leaf=2, random_state=42)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("scaler", scaler),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=350,
                        max_depth=12,
                        min_samples_leaf=1,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            steps=[
                ("scaler", scaler),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=240,
                        max_depth=None,
                        max_features="sqrt",
                        min_samples_leaf=1,
                        random_state=26,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "log_reg": Pipeline(
            steps=[
                ("scaler", scaler),
                (
                    "model",
                    LogisticRegression(
                        penalty="l2",
                        C=3.0,
                        solver="liblinear",
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "mlp": Pipeline(
            steps=[
                ("scaler", scaler),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        alpha=1e-3,
                        learning_rate_init=5e-3,
                        max_iter=1200,
                        random_state=42,
                        early_stopping=True,
                        n_iter_no_change=25,
                    ),
                ),
            ]
        ),
    }
    return models


def evaluate_models(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 26
) -> List[ModelResult]:
    X = df.drop(columns=["target"])
    y = df["target"].map({"M": 1, "R": 0}).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    results: List[ModelResult] = []
    models = build_models()

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        results.append(
            ModelResult(
                name=name,
                accuracy=acc,
                precision=prec,
                recall=rec,
                y_true=y_test.to_numpy(),
                y_pred=y_pred,
            )
        )
        print(f"{name:15s}  acc={acc:.4f}  precision={prec:.4f}  recall={rec:.4f}")
        print(classification_report(y_test, y_pred, digits=4))
    return results


def save_confusion_matrices(results: List[ModelResult]) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    labels = ["Rock (0)", "Mine (1)"]

    for res in results:
        cm = confusion_matrix(res.y_true, res.y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(f"Confusion Matrix - {res.name}")
        plt.tight_layout()
        out_path = OUTPUT_DIR / f"sonar_{res.name}_cm.png"
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")


def main() -> None:
    df = load_sonar()
    results = evaluate_models(df)
    save_confusion_matrices(results)
    best = max(results, key=lambda r: r.accuracy)
    print(f"Best model: {best.name} with accuracy={best.accuracy:.4f}")


if __name__ == "__main__":
    main()
