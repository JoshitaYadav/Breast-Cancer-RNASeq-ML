
from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
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
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42


def label_from_sample(sample: str) -> int:
    """Return 1 for cancer/tumor and 0 for non-cancer/normal using GSE183947 column names."""
    s = str(sample).lower().strip()
    if s.startswith("cap."):
        return 0
    if s.startswith("ca."):
        return 1
    if any(x in s for x in ["normal", "control", "non_cancer", "noncancer", "healthy", "adjacent"]):
        return 0
    if any(x in s for x in ["tumor", "tumour", "cancer", "carcinoma", "malignant"]):
        return 1
    raise ValueError(f"Could not infer cancer/non-cancer label from sample name: {sample}")


def group_from_sample(sample: str) -> str:
    """Use the identifier after CA./CAP. as the patient/group ID."""
    sample = str(sample).strip()
    return sample.split(".", 1)[1] if "." in sample else sample


def read_expression_matrix(path: str | Path) -> pd.DataFrame:
    """Read genes x samples FPKM CSV/CSV.GZ."""
    path = Path(path)
    if not path.exists() and not path.is_absolute():
        candidate = Path("/mnt/data") / path
        if candidate.exists():
            path = candidate
    if not path.exists():
        raise FileNotFoundError(f"Expression CSV not found: {path}")

    expr = pd.read_csv(path, index_col=0)
    expr.index = expr.index.astype(str).str.strip().str.strip('"')
    expr.columns = expr.columns.astype(str).str.strip().str.strip('"')
    expr = expr.apply(pd.to_numeric, errors="coerce")
    if expr.index.duplicated().any():
        expr = expr.groupby(expr.index).mean()
    return expr


def make_pipeline(n_features: int, k_best: int, n_estimators: int, random_state: int, oob: bool) -> Pipeline:
    """Build the classifier pipeline. Feature selection is inside the pipeline to avoid leakage."""
    k = "all" if k_best <= 0 else min(k_best, n_features)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("feature_selection", SelectKBest(score_func=f_classif, k=k)),
        ("classifier", RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
            max_features="sqrt",
            bootstrap=True,
            oob_score=oob,
        )),
    ])


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float | int | list]:
    """Compute binary classification metrics with cancer as the positive class."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_cancer": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity_cancer": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity_non_cancer": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "f1_cancer": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan"),
        "average_precision": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan"),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "confusion_matrix_labels_0_non_cancer_1_cancer": cm.tolist(),
    }


def summarize_cv(scores: Dict[str, np.ndarray], scoring_names: list[str]) -> Dict[str, object]:
    """Summarize cross_validate output."""
    out: Dict[str, object] = {}
    for name in scoring_names:
        vals = np.asarray(scores[f"test_{name}"], dtype=float)
        out[name] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "fold_values": [float(v) for v in vals],
        }
    if "train_accuracy" in scores:
        train = np.asarray(scores["train_accuracy"], dtype=float)
        test = np.asarray(scores["test_accuracy"], dtype=float)
        out["train_accuracy"] = {
            "mean": float(train.mean()),
            "std": float(train.std(ddof=1)) if len(train) > 1 else 0.0,
            "fold_values": [float(v) for v in train],
        }
        out["generalization_gap_train_minus_test_accuracy"] = float(train.mean() - test.mean())
    return out


def run_cv(name: str, model: Pipeline, X: pd.DataFrame, y: np.ndarray, cv, groups: Optional[np.ndarray]) -> Dict[str, object]:
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision_cancer": "precision",
        "recall_sensitivity_cancer": "recall",
        "f1_cancer": "f1",
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
    }
    kwargs = {"groups": groups} if groups is not None else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=True,
            **kwargs,
        )
    summary = summarize_cv(scores, list(scoring.keys()))
    summary["validation_name"] = name
    return summary


def plot_validation_summary(results: Dict[str, object], outpath: Path) -> None:
    holdout = results["holdout_corrected_patient_ID_group_split"]
    skf = results["cv_stratified_5fold_sample_level"]
    sgkf = results["cv_stratified_group_5fold_patient_ID"]
    rows = [
        ["Corrected holdout", holdout["accuracy"], holdout["recall_sensitivity_cancer"], holdout["roc_auc"]],
        ["Stratified 5-fold", skf["accuracy"]["mean"], skf["recall_sensitivity_cancer"]["mean"], skf["roc_auc"]["mean"]],
        ["Stratified-group 5-fold", sgkf["accuracy"]["mean"], sgkf["recall_sensitivity_cancer"]["mean"], sgkf["roc_auc"]["mean"]],
    ]
    if "permutation_sanity_check" in results:
        p = results["permutation_sanity_check"]
        rows.append(["Permutation null", p["accuracy_mean"], np.nan, p["roc_auc_mean"]])
    df = pd.DataFrame(rows, columns=["Validation", "Accuracy", "Cancer recall", "ROC AUC"])
    df.to_csv(outpath.with_suffix(".csv"), index=False)

    x = np.arange(len(df))
    width = 0.24
    plt.figure(figsize=(9.5, 5))
    plt.bar(x - width, df["Accuracy"], width=width, label="Accuracy")
    plt.bar(x, df["Cancer recall"], width=width, label="Cancer recall")
    plt.bar(x + width, df["ROC AUC"], width=width, label="ROC AUC")
    plt.axhline(0.5, linestyle="--", linewidth=1)
    plt.ylim(0, 1.08)
    plt.ylabel("Metric value")
    plt.title("RandomForestClassifier validation re-check")
    plt.xticks(x, df["Validation"], rotation=25, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def run_permutation_check(X: pd.DataFrame, y: np.ndarray, n_permutations: int, k_best: int, random_state: int) -> Dict[str, object]:
    """Run a lightweight shuffled-label sanity check."""
    if n_permutations <= 0:
        return {}
    rng = np.random.default_rng(random_state)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    acc, auc = [], []
    for i in range(n_permutations):
        y_perm = rng.permutation(y)
        model = make_pipeline(X.shape[1], k_best=k_best, n_estimators=10, random_state=random_state + i, oob=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_validate(
                model,
                X,
                y_perm,
                cv=cv,
                scoring={"accuracy": "accuracy", "roc_auc": "roc_auc"},
                n_jobs=1,
                return_train_score=False,
            )
        acc.append(float(np.mean(scores["test_accuracy"])))
        auc.append(float(np.mean(scores["test_roc_auc"])))
    return {
        "purpose": "Labels shuffled; performance should be close to chance if leakage is absent.",
        "n_permutations": n_permutations,
        "n_estimators_for_permutation_model": 10,
        "accuracy_mean": float(np.mean(acc)),
        "accuracy_std": float(np.std(acc, ddof=1)) if len(acc) > 1 else 0.0,
        "accuracy_values": acc,
        "roc_auc_mean": float(np.mean(auc)),
        "roc_auc_std": float(np.std(auc, ddof=1)) if len(auc) > 1 else 0.0,
        "roc_auc_values": auc,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Corrected RandomForestClassifier validation for GSE183947.")
    parser.add_argument("--expression_csv", default="/mnt/data/GSE183947_fpkm.csv")
    parser.add_argument("--output_dir", default="rechecked_rf_classifier_outputs")
    parser.add_argument("--test_size", type=float, default=0.25)
    parser.add_argument("--k_best", type=int, default=500)
    parser.add_argument("--n_estimators", type=int, default=1000)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--n_permutations", type=int, default=10)
    parser.add_argument("--random_state", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    expr = read_expression_matrix(args.expression_csv)
    X = np.log1p(expr.clip(lower=0)).T
    X.index.name = "sample"
    y = np.array([label_from_sample(s) for s in X.index], dtype=int)
    groups = np.array([group_from_sample(s) for s in X.index])

    sample_table = pd.DataFrame({
        "sample": X.index,
        "class_label": np.where(y == 1, "cancer", "non_cancer"),
        "target": y,
        "patient_group_by_id": groups,
    })
    sample_table.to_csv(outdir / "sample_table_rechecked.csv", index=False)
    group_table = sample_table.groupby("patient_group_by_id").agg(
        n_samples=("sample", "count"),
        labels=("class_label", lambda z: ",".join(z)),
        samples=("sample", lambda z: ",".join(z)),
    ).reset_index()
    group_table.to_csv(outdir / "corrected_patient_grouping_by_sample_id.csv", index=False)

    group_counts = Counter(groups)
    dataset_checks = {
        "n_samples": int(X.shape[0]),
        "n_genes": int(X.shape[1]),
        "n_cancer": int((y == 1).sum()),
        "n_non_cancer": int((y == 0).sum()),
        "class_balance": "balanced: 30 cancer and 30 non-cancer",
        "missing_expression_values": int(pd.isna(X).sum().sum()),
        "patient_groups_by_ID": int(len(np.unique(groups))),
        "complete_ID_matched_pairs": int(sum(1 for v in group_counts.values() if v == 2)),
        "singleton_ID_groups": int(sum(1 for v in group_counts.values() if v == 1)),
        "singleton_samples": sample_table[sample_table["patient_group_by_id"].map(group_counts) == 1]["sample"].tolist(),
    }

   .
    chosen = None
    splitter = GroupShuffleSplit(n_splits=200, test_size=args.test_size, random_state=args.random_state)
    for train_idx, test_idx in splitter.split(X, y, groups=groups):
        if len(np.unique(y[test_idx])) == 2 and len(np.unique(y[train_idx])) == 2:
            if (y[test_idx] == 1).sum() == (y[test_idx] == 0).sum():
                chosen = (train_idx, test_idx)
                break
            if chosen is None:
                chosen = (train_idx, test_idx)
    if chosen is None:
        raise RuntimeError("Could not create a valid group-aware holdout split.")

    train_idx, test_idx = chosen
    holdout_model = make_pipeline(X.shape[1], args.k_best, args.n_estimators, args.random_state, oob=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        holdout_model.fit(X.iloc[train_idx], y[train_idx])
    y_pred = holdout_model.predict(X.iloc[test_idx])
    y_prob = holdout_model.predict_proba(X.iloc[test_idx])[:, 1]
    holdout_metrics = binary_metrics(y[test_idx], y_pred, y_prob)
    holdout_metrics.update({
        "split_method": "GroupShuffleSplit using corrected patient ID groups",
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "train_cancer": int((y[train_idx] == 1).sum()),
        "train_non_cancer": int((y[train_idx] == 0).sum()),
        "test_cancer": int((y[test_idx] == 1).sum()),
        "test_non_cancer": int((y[test_idx] == 0).sum()),
        "oob_training_accuracy": float(holdout_model.named_steps["classifier"].oob_score_),
    })

    pd.DataFrame({
        "sample": X.index[test_idx],
        "patient_group_by_id": groups[test_idx],
        "true_label": np.where(y[test_idx] == 1, "cancer", "non_cancer"),
        "predicted_label": np.where(y_pred == 1, "cancer", "non_cancer"),
        "predicted_probability_cancer": y_prob,
    }).to_csv(outdir / "corrected_holdout_predictions.csv", index=False)

    # CV checks. Both use a full Pipeline, so imputation and feature selection happen inside each fold.
    cv_model = make_pipeline(X.shape[1], args.k_best, args.n_estimators, args.random_state, oob=False)
    skf = run_cv(
        "StratifiedKFold 5-fold sample-level CV",
        cv_model,
        X,
        y,
        StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state),
        groups=None,
    )
    sgkf = run_cv(
        "StratifiedGroupKFold 5-fold corrected patient-ID CV",
        cv_model,
        X,
        y,
        StratifiedGroupKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state),
        groups=groups,
    )

    results: Dict[str, object] = {
        "dataset_checks": dataset_checks,
        "holdout_corrected_patient_ID_group_split": holdout_metrics,
        "cv_stratified_5fold_sample_level": skf,
        "cv_stratified_group_5fold_patient_ID": sgkf,
    }
    if args.n_permutations > 0:
        results["permutation_sanity_check"] = run_permutation_check(X, y, args.n_permutations, args.k_best, args.random_state)

    with open(outdir / "rechecked_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    plot_validation_summary(results, outdir / "validation_recheck_metrics.png")
    joblib.dump(holdout_model, outdir / "random_forest_classifier_rechecked.joblib")

    print("\nDataset checks")
    print(json.dumps(dataset_checks, indent=2))
    print("\nCorrected holdout metrics")
    print(json.dumps(holdout_metrics, indent=2))
    print("\nStratifiedKFold accuracy:", skf["accuracy"])
    print("StratifiedGroupKFold accuracy:", sgkf["accuracy"])
    if "permutation_sanity_check" in results:
        p = results["permutation_sanity_check"]
        print("\nPermutation null accuracy mean +/- SD:", f"{p['accuracy_mean']:.4f} +/- {p['accuracy_std']:.4f}")
    print(f"\nSaved outputs to: {outdir}")


if __name__ == "__main__":
    main()
