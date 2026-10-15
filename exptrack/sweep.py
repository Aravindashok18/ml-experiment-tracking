"""Run the whole experiment grid.

    python -m exptrack.sweep                       # 48 random configurations + a baseline
    python -m exptrack.sweep --lr 4 --rf 4 --xgb 4 # a smaller sweep

Safe to interrupt and re-run: configurations that already have a finished run are skipped.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlflow

from . import tracking
from .data import load_adult
from .models import Config, generate_configs
from .validate import compare_train_test, require, validate

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--tracking-uri", default=f"sqlite:///{ROOT / 'mlflow.db'}")
    parser.add_argument("--artifact-root", default=str(ROOT / "mlartifacts"))
    parser.add_argument("--lr", type=int, default=12, help="logistic regression configurations")
    parser.add_argument("--rf", type=int, default=18, help="random forest configurations")
    parser.add_argument("--xgb", type=int, default=18, help="XGBoost configurations")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-model-artifacts", action="store_true", help="skip logging fitted models (faster, smaller)")
    args = parser.parse_args()

    train, test, cleaning = load_adult(args.data_dir)
    print(f"train {cleaning['train']} \ntest  {cleaning['test']}")
    reports = {"train": validate(train, "train"), "test": validate(test, "test"), "train_vs_test": compare_train_test(train, test)}
    require(*reports.values())
    print("data validation passed:", {name: len(r.checks) for name, r in reports.items()}, "checks")

    if args.xgb:  # fail now, not halfway through the sweep
        try:
            import xgboost  # noqa: F401
        except Exception as error:
            raise SystemExit(
                f"XGBoost cannot be loaded ({str(error).splitlines()[0]}). On macOS it needs the OpenMP runtime: "
                "`brew install libomp`, or point DYLD_FALLBACK_LIBRARY_PATH at scikit-learn's copy (see the README).")

    tracking.setup(args.tracking_uri, artifact_root=args.artifact_root)
    configs = [Config("baseline", {})] + generate_configs({"logistic_regression": args.lr, "random_forest": args.rf, "xgboost": args.xgb}, args.seed)
    done = tracking.existing_config_ids()
    todo = [c for c in configs if c.config_id not in done]
    print(f"{len(configs)} configurations, {len(configs) - len(todo)} already done, {len(todo)} to run")

    started = time.perf_counter()
    for index, config in enumerate(todo, 1):
        result = tracking.run_experiment(config, train, test, reports, seed=args.seed, log_model=not args.no_model_artifacts)
        print(f"[{index}/{len(todo)}] {config.family:20s} {config.config_id}  cv_auc={result['cv_roc_auc_mean']:.4f} "
              f"cv_f1={result['cv_f1_mean']:.4f}  {result['train_seconds']:.1f}s  ({time.perf_counter() - started:.0f}s elapsed)", flush=True)


if __name__ == "__main__":
    main()
