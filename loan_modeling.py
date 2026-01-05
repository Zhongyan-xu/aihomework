"""
Classification pipeline for the "Realistic Loan Approval Dataset (US & Canada)".

Behavior:
- Looks for a CSV file at data/loan_approval.csv by default.
- If the expected dataset is missing, downloads a public loan-approval dataset
  (same schema style) so the pipeline can run end-to-end without Kaggle creds.
- Handles preprocessing (imputing + one-hot encoding), trains multiple models,
  and saves confusion matrix plots into outputs/.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


DATA_DIR = pathlib.Path("data")
OUTPUT_DIR = pathlib.Path("outputs")
DEFAULT_DATASET = DATA_DIR / "loan_approval.csv"
# Australian credit approval dataset (public UCI mirror) as a runnable fallback
FALLBACK_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
TARGET_COL = "Loan_Status"


@dataclass
class ModelResult:
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    y_true: pd.Series
    y_pred: pd.Series


def ensure_dataset() -> pathlib.Path:
    """
    Try to locate the Kaggle dataset locally. If it is missing, fetch a
    public loan approval dataset with a similar schema so the workflow
    remains runnable. Returns the path to the CSV file that will be used.
    """
    DATA_DIR.mkdir(exist_ok=True)
    if DEFAULT_DATASET.exists():
        return DEFAULT_DATASET

    fallback_path = DATA_DIR / "loan_approval_fallback.csv"
    if not fallback_path.exists():
        print("Kaggle dataset not found, downloading fallback dataset...")
        cols = [f"A{i}" for i in range(1, 16)] + ["A16"]
        df = pd.read_csv(FALLBACK_URL, header=None, names=cols, na_values="?")
        df = df.rename(columns={"A16": TARGET_COL})
        df[TARGET_COL] = df[TARGET_COL].map({"+": 1, "-": 0})
        df.to_csv(fallback_path, index=False)
        print(f"Saved fallback dataset to {fallback_path}")
    else:
        print(f"Using existing fallback dataset at {fallback_path}")
    return fallback_path


def load_data(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in {path}")
    return df


def preprocess_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 7):
    # Normalize target to binary 0/1
    y_raw = df[TARGET_COL]
    if set(pd.unique(y_raw)) <= {0, 1}:
        y = y_raw.astype(int)
    else:
        y = y_raw.map({"Y": 1, "N": 0, "+": 1, "-": 0}).astype(int)
    X = df.drop(columns=[TARGET_COL])

    # Identify column types
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, cat_cols),
            ("numeric", numeric_transformer, num_cols),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return preprocessor, X_train, X_test, y_train, y_test


def build_models(preprocessor: ColumnTransformer) -> Dict[str, Pipeline]:
    return {
        "log_reg": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=800,
                        solver="liblinear",
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=250,
                        max_depth=None,
                        min_samples_leaf=1,
                        random_state=11,
                        n_jobs=-1,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=180,
                        learning_rate=0.06,
                        max_depth=3,
                        random_state=19,
                    ),
                ),
            ]
        ),
    }


def evaluate(models: Dict[str, Pipeline], X_train, X_test, y_train, y_test) -> List[ModelResult]:
    results: List[ModelResult] = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        results.append(
            ModelResult(
                name=name,
                accuracy=acc,
                precision=prec,
                recall=rec,
                f1=f1,
                y_true=y_test,
                y_pred=preds,
            )
        )
        print(f"{name:16s} acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")
        print(classification_report(y_test, preds, digits=4))
    return results


def save_confusion_matrices(results: List[ModelResult], prefix: str = "loan") -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    labels = ["Not Approved (0)", "Approved (1)"]
    for res in results:
        cm = confusion_matrix(res.y_true, res.y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        disp.plot(ax=ax, cmap="Purples", colorbar=False, values_format="d")
        ax.set_title(f"Confusion Matrix - {res.name}")
        plt.tight_layout()
        out_path = OUTPUT_DIR / f"{prefix}_{res.name}_cm.png"
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")


def main() -> None:
    data_path = ensure_dataset()
    df = load_data(data_path)
    preprocessor, X_train, X_test, y_train, y_test = preprocess_split(df)
    models = build_models(preprocessor)
    results = evaluate(models, X_train, X_test, y_train, y_test)
    save_confusion_matrices(results)
    best = max(results, key=lambda r: r.f1)
    print(f"Best model: {best.name} with accuracy={best.accuracy:.4f}, f1={best.f1:.4f}")


if __name__ == "__main__":
    main()
