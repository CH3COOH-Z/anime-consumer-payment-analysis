from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


PAY_INDEX = 42

CONTROL_INDEXES = [6, 7, 8, 9, 10, 21, 22, 65, 66]

# Intentionally excludes:
# - spending type (43:49)
# - spending amount (49)
# - purchase decision block (50:57)
# because these are too close to the target and/or conditionally skipped.
MODULE_RANGES = {
    "access": list(range(23, 29)),
    "content_type": list(range(29, 36)),
    "ip_type": list(range(36, 42)),
    "offline_motive": list(range(67, 73)),
    "problem": list(range(73, 80)),
    "expect": list(range(80, 86)),
}


def load_data(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("Input must be .xlsx, .xls, or .csv")


def clean_binary_block(X):
    """Convert multi-select block to 0/1 numeric values.

    Questionnaire skip codes or unexpected values are mapped to 0.
    """
    if isinstance(X, pd.DataFrame):
        out = X.apply(pd.to_numeric, errors="coerce")
        out = out.where(out.isin([0, 1]), 0).astype(float)
        return out

    out = pd.DataFrame(X).apply(pd.to_numeric, errors="coerce")
    out = out.where(out.isin([0, 1]), 0).astype(float)
    return out.to_numpy()


def build_target(df: pd.DataFrame) -> pd.Series:
    s = pd.to_numeric(df.iloc[:, PAY_INDEX], errors="coerce")
    if s.isna().any():
        raise ValueError("Target column contains non-numeric or missing values.")
    return (s > 1).astype(int).rename("Y_pay")


def build_pipeline() -> Pipeline:
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )

    transformers = [
        ("controls", categorical_pipe, CONTROL_INDEXES),
    ]

    for block_name, indexes in MODULE_RANGES.items():
        block_pipe = Pipeline(
            steps=[
                (
                    "clean",
                    FunctionTransformer(
                        clean_binary_block,
                        validate=False,
                        feature_names_out="one-to-one",
                    ),
                ),
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=0.80, svd_solver="full")),
            ]
        )
        transformers.append((block_name, block_pipe, indexes))

    preprocess = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        solver="liblinear",
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )


def evaluate_holdout(pipe, X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "n_total": int(len(y)),
        "positive_rate": float(y.mean()),
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "average_precision": float(average_precision_score(y_test, proba)),
    }

    return metrics, confusion_matrix(y_test, pred), pipe


def evaluate_cv(pipe, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "balanced_accuracy": "balanced_accuracy",
        "roc_auc": "roc_auc",
        "f1": "f1",
        "average_precision": "average_precision",
    }

    result = cross_validate(pipe, X, y, cv=cv, scoring=scoring)

    rows = []
    for metric in scoring:
        values = result[f"test_{metric}"]
        rows.append(
            {
                "metric": metric,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
        )

    return pd.DataFrame(rows)


def evaluate_baseline(X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    dummy_X = pd.DataFrame({"constant": np.ones(len(y))})

    model = DummyClassifier(strategy="prior")

    scoring = {
        "balanced_accuracy": "balanced_accuracy",
        "roc_auc": "roc_auc",
        "f1": "f1",
        "average_precision": "average_precision",
    }

    result = cross_validate(model, dummy_X, y, cv=cv, scoring=scoring)

    rows = []
    for metric in scoring:
        values = result[f"test_{metric}"]
        rows.append(
            {
                "metric": metric,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
        )

    return pd.DataFrame(rows)


def save_figures(y, cm, fitted_pipe, figures_dir: Path):
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Target distribution
    counts = y.value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    plt.bar([str(i) for i in counts.index], counts.values)
    plt.title("Payment-intention target distribution")
    plt.xlabel("Y_pay")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(figures_dir / "target_distribution.png", dpi=200)
    plt.close()

    # Confusion matrix
    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title("Holdout confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    # PCA variance summaries from fitted training pipeline
    pre = fitted_pipe.named_steps["preprocess"]
    rows = []
    for block_name in MODULE_RANGES:
        block_pipe = pre.named_transformers_[block_name]
        pca = block_pipe.named_steps["pca"]
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        for component, value in enumerate(cumulative, start=1):
            rows.append((block_name, component, value))

    plt.figure(figsize=(8, 5))
    pca_df = pd.DataFrame(rows, columns=["block", "component", "cumulative"])
    for block_name, group in pca_df.groupby("block"):
        plt.plot(
            group["component"],
            group["cumulative"],
            marker="o",
            label=block_name,
        )
    plt.axhline(0.80, linestyle="--", linewidth=1)
    plt.ylim(0, 1.05)
    plt.xlabel("Principal component")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA variance retained by survey block")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(figures_dir / "pca_variance.png", dpi=200)
    plt.close()

    # Logistic coefficients
    try:
        feature_names = pre.get_feature_names_out()
        coefficients = fitted_pipe.named_steps["model"].coef_[0]
        coef_df = pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": coefficients,
            }
        )
        coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
        top = coef_df.nlargest(15, "abs_coefficient").sort_values("coefficient")

        plt.figure(figsize=(10, 6))
        plt.barh(top["feature"], top["coefficient"])
        plt.axvline(0, linewidth=1)
        plt.title("Largest logistic-regression coefficients")
        plt.xlabel("Coefficient")
        plt.tight_layout()
        plt.savefig(figures_dir / "top_coefficients.png", dpi=200)
        plt.close()

        return coef_df.sort_values("abs_coefficient", ascending=False)
    except Exception:
        return pd.DataFrame()


def main(data_path: Path, output_dir: Path):
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)

    if df.shape[1] < 86:
        raise ValueError(
            f"Expected the coded questionnaire format with at least 86 columns; got {df.shape[1]}."
        )

    y = build_target(df)
    X = df.copy()

    pipe = build_pipeline()

    holdout_metrics, cm, fitted_pipe = evaluate_holdout(pipe, X, y)
    cv_metrics = evaluate_cv(build_pipeline(), X, y)
    baseline_metrics = evaluate_baseline(X, y)

    coefficient_table = save_figures(y, cm, fitted_pipe, figures_dir)

    with open(
        tables_dir / "holdout_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(holdout_metrics, f, ensure_ascii=False, indent=2)

    cv_metrics.to_csv(
        tables_dir / "cross_validation_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    baseline_metrics.to_csv(
        tables_dir / "baseline_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not coefficient_table.empty:
        coefficient_table.to_csv(
            tables_dir / "logistic_coefficients.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print("Holdout metrics")
    print(json.dumps(holdout_metrics, ensure_ascii=False, indent=2))

    print("\n5-fold cross-validation")
    print(cv_metrics.to_string(index=False))

    print("\nMajority baseline")
    print(baseline_metrics.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", default=Path("reports"), type=Path)
    args = parser.parse_args()

    main(args.data, args.out)
