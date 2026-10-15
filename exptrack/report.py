"""Statistical summary of the tracked runs: leaderboards, hyperparameter effects, paired tests.

    python -m exptrack.report            # writes reports/summary.md, reports/runs.csv and two plots
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from . import tracking  # noqa: E402
from .data import load_adult, split_xy  # noqa: E402

FAMILY_LABEL = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest", "xgboost": "XGBoost", "baseline": "Baseline (class prior)"}
FAMILY_COLOUR = {"logistic_regression": "#2a6fdb", "random_forest": "#2e9d5b", "xgboost": "#d9731a", "baseline": "#888888"}


def load_runs(experiment: str = tracking.EXPERIMENT) -> pd.DataFrame:
    """One row per finished run, with hyperparameters as hp_* columns and metrics without the prefix."""
    raw = mlflow.search_runs(experiment_names=[experiment], filter_string="attributes.status = 'FINISHED'", output_format="pandas")
    table = pd.DataFrame({"run_id": raw["run_id"], "family": raw["tags.family"], "config_id": raw["tags.config_id"]})
    for column in raw.columns:
        if column.startswith("metrics."):
            table[column.removeprefix("metrics.")] = raw[column]
        elif column.startswith("params.hp."):
            table["hp_" + column.removeprefix("params.hp.")] = raw[column]
    for column in [c for c in table.columns if c.startswith("hp_")]:  # MLflow stores every parameter as a string
        values = table[column].replace("None", np.nan)
        if column == "hp_max_depth":
            values = table[column].replace("None", "30")  # unlimited depth ranks above every depth that was tried
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.notna().sum() == values.notna().sum():  # every present value is a number
            table[column] = numeric
    return table.sort_values("cv_roc_auc_mean", ascending=False).reset_index(drop=True)


def family_summary(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, group in runs.groupby("family"):
        champion = group.sort_values("cv_roc_auc_mean", ascending=False).iloc[0]  # chosen on CV only
        rows.append({
            "family": FAMILY_LABEL[family], "runs": len(group),
            "cv_auc_best": group["cv_roc_auc_mean"].max(), "cv_auc_median": group["cv_roc_auc_mean"].median(),
            "cv_auc_worst": group["cv_roc_auc_mean"].min(),
            "champion_test_auc": champion["test_roc_auc"], "champion_test_f1": champion["test_f1"],
            "median_train_seconds": group["train_seconds"].median(),
        })
    return pd.DataFrame(rows).sort_values("cv_auc_best", ascending=False).reset_index(drop=True)


def hyperparameter_effects(runs: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation between each numeric hyperparameter and CV AUC, within one family."""
    rows = []
    for family in ("logistic_regression", "random_forest", "xgboost"):
        group = runs[runs["family"] == family]
        for column in [c for c in group.columns if c.startswith("hp_") and group[c].notna().sum() >= 8]:
            values = pd.to_numeric(group[column], errors="coerce")
            if values.nunique() < 3:
                continue
            rho, p = stats.spearmanr(values, group["cv_roc_auc_mean"], nan_policy="omit")
            rows.append({"family": FAMILY_LABEL[family], "hyperparameter": column.removeprefix("hp_"), "n": int(values.notna().sum()),
                         "spearman_rho": float(rho), "p_value": float(p)})
    columns = ["family", "hyperparameter", "n", "spearman_rho", "p_value"]
    table = pd.DataFrame(rows, columns=columns).sort_values("p_value").reset_index(drop=True)
    # one test per row: with this many tests, a p of 0.04 is expected by chance alone
    table["significant_after_bonferroni"] = table["p_value"] < 0.05 / max(len(table), 1)
    return table


def categorical_effects(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family in ("logistic_regression", "random_forest"):
        group = runs[runs["family"] == family]
        for column in ("hp_penalty", "hp_class_weight", "hp_max_features"):
            if column in group and group[column].notna().any():
                for level, part in group.groupby(group[column].astype(str)):
                    rows.append({"family": FAMILY_LABEL[family], "hyperparameter": column.removeprefix("hp_"), "level": level,
                                 "runs": len(part), "mean_cv_auc": part["cv_roc_auc_mean"].mean()})
    return pd.DataFrame(rows)


def cv_vs_test_agreement(runs: pd.DataFrame) -> Tuple[float, float]:
    """Does ranking runs by cross-validation predict their ranking on the held-out test set?"""
    real = runs[runs["family"] != "baseline"]
    rho, p = stats.spearmanr(real["cv_roc_auc_mean"], real["test_roc_auc"])
    return float(rho), float(p)


def _f1(y: np.ndarray, pred: np.ndarray) -> float:
    tp = float(np.sum((y == 1) & (pred == 1)))
    denominator = 2 * tp + float(np.sum((y == 0) & (pred == 1))) + float(np.sum((y == 1) & (pred == 0)))
    return 2 * tp / denominator if denominator else 0.0


def paired_bootstrap(y: np.ndarray, proba_a: np.ndarray, proba_b: np.ndarray, metric: str, resamples: int = 1000, seed: int = 0) -> Dict[str, float]:
    """Difference in a test metric (a minus b) over resamples of the same test rows, so both models see identical data."""
    rng = np.random.default_rng(seed)
    n, diffs = len(y), np.empty(resamples)
    for i in range(resamples):
        idx = rng.integers(0, n, n)
        if metric == "roc_auc":
            diffs[i] = roc_auc_score(y[idx], proba_a[idx]) - roc_auc_score(y[idx], proba_b[idx])
        else:
            diffs[i] = _f1(y[idx], proba_a[idx] >= 0.5) - _f1(y[idx], proba_b[idx] >= 0.5)
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(diffs.mean()), "ci_low": float(low), "ci_high": float(high), "share_a_better": float((diffs > 0).mean())}


def champion_comparisons(runs: pd.DataFrame, y_test: np.ndarray, resamples: int = 1000) -> pd.DataFrame:
    champions = {f: g.sort_values("cv_roc_auc_mean", ascending=False).iloc[0] for f, g in runs[runs["family"] != "baseline"].groupby("family")}
    proba = {}
    for family, row in champions.items():
        path = mlflow.artifacts.download_artifacts(run_id=row["run_id"], artifact_path="diagnostics/test_proba.npy")
        proba[family] = np.load(path).astype("float64")
    rows = []
    order = sorted(champions, key=lambda f: -champions[f]["cv_roc_auc_mean"])
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            for metric in ("roc_auc", "f1"):
                result = paired_bootstrap(y_test, proba[a], proba[b], metric, resamples)
                label = {"roc_auc": "test ROC AUC", "f1": "test F1"}[metric]
                rows.append({"a": FAMILY_LABEL[a], "b": FAMILY_LABEL[b], "metric": label, **result})
    return pd.DataFrame(rows)


# --- output -----------------------------------------------------------------------------------

def _md(frame: pd.DataFrame, formats: Dict[str, str]) -> str:
    header = "| " + " | ".join(frame.columns) + " |"
    divider = "|" + "|".join("---" for _ in frame.columns) + "|"
    lines = [header, divider]
    for _, row in frame.iterrows():
        cells = [format(row[c], formats[c]) if c in formats and pd.notna(row[c]) else str(row[c]) for c in frame.columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_auc_by_family(runs: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    rng = np.random.default_rng(0)
    families = [f for f in ("logistic_regression", "random_forest", "xgboost") if f in set(runs["family"])]
    for position, family in enumerate(families):
        group = runs[runs["family"] == family]
        x = position + rng.uniform(-0.15, 0.15, len(group))
        ax.scatter(x, group["cv_roc_auc_mean"], s=28, alpha=0.75, color=FAMILY_COLOUR[family])
        best = group["cv_roc_auc_mean"].to_numpy().argmax()
        ax.scatter([x[best]], [group["cv_roc_auc_mean"].iloc[best]], s=110, facecolors="none", edgecolors="black", linewidths=1.4)
    ax.set_xticks(range(len(families)), [FAMILY_LABEL[f] for f in families])
    ax.set_ylabel("cross-validated ROC AUC")
    ax.set_title("One dot per run; circled = best in family", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_auc_vs_time(runs: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for family in ("logistic_regression", "random_forest", "xgboost"):
        group = runs[runs["family"] == family]
        ax.scatter(group["train_seconds"], group["cv_roc_auc_mean"], s=28, alpha=0.75, color=FAMILY_COLOUR[family], label=FAMILY_LABEL[family])
    ax.set_xscale("log")
    ax.set_xlabel("final fit time, seconds (log scale)")
    ax.set_ylabel("cross-validated ROC AUC")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def build_summary(runs: pd.DataFrame, y_test: np.ndarray, resamples: int) -> str:
    real = runs[runs["family"] != "baseline"]
    baseline = runs[runs["family"] == "baseline"]
    families = family_summary(runs[runs["family"] != "baseline"])
    effects = hyperparameter_effects(runs)
    comparisons = champion_comparisons(runs, y_test, resamples)
    rho, p = cv_vs_test_agreement(runs)
    top = real.head(10)[["family", "config_id", "cv_roc_auc_mean", "cv_roc_auc_std", "cv_f1_mean", "test_roc_auc", "test_f1", "train_seconds"]].copy()
    top["family"] = top["family"].map(FAMILY_LABEL)

    parts = [
        f"# Experiment summary\n\n{len(runs)} finished runs ({len(real)} model configurations plus a class-prior baseline), "
        f"5-fold stratified cross-validation on the training file, one evaluation on the held-out test file ({len(y_test):,} rows).\n",
    ]
    if len(baseline):
        b = baseline.iloc[0]
        parts.append(f"Baseline (always predicts the class prior): ROC AUC {b['test_roc_auc']:.3f}, accuracy {b['test_accuracy']:.3f} "
                     f"(the positive class is {1 - b['test_accuracy']:.1%} of the test rows).\n")
    parts += [
        "## Families\n\nSelection uses cross-validated ROC AUC only. The champion's test scores are what it achieved afterwards.\n",
        _md(families, {"cv_auc_best": ".4f", "cv_auc_median": ".4f", "cv_auc_worst": ".4f", "champion_test_auc": ".4f", "champion_test_f1": ".4f", "median_train_seconds": ".1f"}),
        "\n\n## Ten best runs by cross-validated ROC AUC\n",
        _md(top, {"cv_roc_auc_mean": ".4f", "cv_roc_auc_std": ".4f", "cv_f1_mean": ".4f", "test_roc_auc": ".4f", "test_f1": ".4f", "train_seconds": ".1f"}),
        f"\n\nRanking all {len(real)} runs by cross-validation or by test score agrees closely: Spearman rho = {rho:.3f} (p = {p:.1e}).\n",
        "## Do the family champions differ?\n\nPaired bootstrap over the same test rows, a minus b; the interval is the 2.5th to 97.5th percentile of "
        f"{resamples:,} resamples. An interval that contains 0 means the test set cannot separate the two.\n",
        _md(comparisons, {"diff": "+.4f", "ci_low": "+.4f", "ci_high": "+.4f", "share_a_better": ".2f"}),
        f"\n\n## Which hyperparameters mattered\n\nSpearman correlation with cross-validated AUC inside each family (random search, so each hyperparameter varies independently of the others). "
        f"{len(effects)} correlations were tested, so the last column applies a Bonferroni threshold of 0.05 / {len(effects)}; a p-value just under 0.05 above it is not evidence on its own.\n",
        _md(effects, {"spearman_rho": "+.3f", "p_value": ".4f"}),
    ]
    levels = categorical_effects(runs)
    if len(levels):
        parts += ["\n\n## Categorical hyperparameters\n\nMean cross-validated AUC per level. Levels were drawn at random, so some have only a few runs "
                  "(see `runs`) and the other hyperparameters differ between levels; read these as descriptions, not tests.\n",
                  _md(levels, {"mean_cv_auc": ".4f"})]
    return "\n".join(parts) + "\n"


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--tracking-uri", default=f"sqlite:///{root / 'mlflow.db'}")
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "raw")
    parser.add_argument("--out", type=Path, default=root / "reports")
    parser.add_argument("--resamples", type=int, default=1000)
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    runs = load_runs()
    _, test, _ = load_adult(args.data_dir)
    _, y_test = split_xy(test)

    args.out.mkdir(parents=True, exist_ok=True)
    keep = [c for c in runs.columns if not c.startswith("cv_") or c.endswith(("_mean", "_std"))]
    runs[keep].to_csv(args.out / "runs.csv", index=False)
    (args.out / "summary.md").write_text(build_summary(runs, y_test, args.resamples))
    plot_auc_by_family(runs[runs["family"] != "baseline"], args.out / "cv_auc_by_family.png")
    plot_auc_vs_time(runs[runs["family"] != "baseline"], args.out / "cv_auc_vs_fit_time.png")
    print(f"wrote {args.out}/summary.md, runs.csv and two plots from {len(runs)} runs")


if __name__ == "__main__":
    main()
