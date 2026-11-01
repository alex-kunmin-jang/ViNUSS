"""
Evaluation of per-epoch predictions (the metrics reported in the paper).

Reads a prediction CSV produced by either
    python train_stage2.py --config ... --test        -> experiments/stage2/<name>/test_results.csv
    python inference.py ...                            -> results/predictions.csv
and reports, for the 5-class (Wake/N1/N2/N3/REM) and the 4-class
(Wake/Light/Deep/REM, N1+N2 -> Light, N3 -> Deep) settings:
    accuracy, macro-F1 (MF1), Cohen's kappa, per-class precision/recall/F1,
    and the confusion matrix.
Optionally a subject-level nonparametric bootstrap 95% CI is computed for
accuracy / MF1 / kappa (resample subjects with replacement, keep each
subject's full night, recompute the pooled metric).

The CSV must contain the columns
    subject, epoch, prediction, label
with integer class indices 0..4 (5-class model outputs).

Usage:
    python evaluate.py --predictions results/predictions.csv
    python evaluate.py --predictions experiments/stage2/ViNUSS_stage2_test/test_results.csv \
        --output results/metrics --bootstrap 1000
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, cohen_kappa_score,
                             confusion_matrix, f1_score)

CLASS_NAMES_5 = ["Wake", "N1", "N2", "N3", "REM"]
CLASS_NAMES_4 = ["Wake", "Light", "Deep", "REM"]
# 4-class mapping used in the paper: N1+N2 -> Light, N3 -> Deep
LABEL_MAP_4CLASS = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3}


def load_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {}
    for cand in ("prediction", "preds", "pred"):
        if cand in df.columns:
            rename[cand] = "pred"
            break
    for cand in ("label", "labels"):
        if cand in df.columns:
            rename[cand] = "label"
            break
    df = df.rename(columns=rename)
    missing = {"subject", "pred", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    df["pred"] = df["pred"].astype(int)
    df["label"] = df["label"].astype(int)
    return df


def to_4class(x: np.ndarray) -> np.ndarray:
    return np.vectorize(LABEL_MAP_4CLASS.get)(x)


def pooled_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }


def bootstrap_ci(df: pd.DataFrame, n_boot: int, seed: int = 42) -> dict:
    """Subject-level bootstrap 95% CI of the pooled metrics."""
    rng = np.random.default_rng(seed)
    subjects = df["subject"].unique()
    arrays = {s: (g["label"].to_numpy(), g["pred"].to_numpy()) for s, g in df.groupby("subject")}
    samples = {k: [] for k in ("accuracy", "macro_f1", "kappa")}
    for _ in range(n_boot):
        pick = rng.choice(subjects, size=len(subjects), replace=True)
        y = np.concatenate([arrays[s][0] for s in pick])
        p = np.concatenate([arrays[s][1] for s in pick])
        m = pooled_metrics(y, p)
        for k in samples:
            samples[k].append(m[k])
    return {k: {"ci_lo": float(np.percentile(v, 2.5)), "ci_hi": float(np.percentile(v, 97.5))}
            for k, v in samples.items()}


def evaluate_setting(df: pd.DataFrame, n_class: int, n_boot: int) -> dict:
    y, p = df["label"].to_numpy(), df["pred"].to_numpy()
    names = CLASS_NAMES_5
    if n_class == 4:
        y, p, names = to_4class(y), to_4class(p), CLASS_NAMES_4
        df = df.assign(label=y, pred=p)

    out = pooled_metrics(y, p)
    report = classification_report(y, p, labels=list(range(n_class)), target_names=names,
                                   digits=3, zero_division=0, output_dict=True)
    out["per_class"] = {c: {k: report[c][k] for k in ("precision", "recall", "f1-score")} for c in names}
    out["confusion_matrix"] = confusion_matrix(y, p, labels=list(range(n_class))).tolist()
    if n_boot > 0:
        out["bootstrap_95ci"] = bootstrap_ci(df, n_boot)

    print(f"\n{'=' * 60}\n{n_class}-class evaluation  ({df['subject'].nunique()} subjects, {len(df)} epochs)\n{'=' * 60}")
    print(f"Accuracy : {out['accuracy']:.4f}")
    print(f"Macro-F1 : {out['macro_f1']:.4f}")
    print(f"Kappa    : {out['kappa']:.4f}")
    if n_boot > 0:
        for k in ("accuracy", "macro_f1", "kappa"):
            ci = out["bootstrap_95ci"][k]
            print(f"  95% CI {k:9s}: [{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]  (subject-level bootstrap, n={n_boot})")
    print(classification_report(y, p, labels=list(range(n_class)), target_names=names, digits=3, zero_division=0))
    print("Confusion matrix (rows = true, cols = predicted):")
    print(pd.DataFrame(out["confusion_matrix"], index=names, columns=names).to_string())
    return out


def main():
    parser = argparse.ArgumentParser(description="Evaluate ViNUSS per-epoch predictions")
    parser.add_argument("--predictions", required=True, help="CSV with subject, epoch, prediction, label")
    parser.add_argument("--output", default=None, help="Optional directory to write metrics_{5,4}class.json")
    parser.add_argument("--bootstrap", type=int, default=0, help="Number of subject-level bootstrap resamples (0 = off)")
    args = parser.parse_args()

    df = load_predictions(args.predictions)
    results = {5: evaluate_setting(df, 5, args.bootstrap), 4: evaluate_setting(df, 4, args.bootstrap)}

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        for n_class, res in results.items():
            path = os.path.join(args.output, f"metrics_{n_class}class.json")
            with open(path, "w") as f:
                json.dump(res, f, indent=2)
            print(f"Saved {path}")


if __name__ == "__main__":
    main()
