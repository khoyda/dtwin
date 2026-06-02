"""Train the canola yield model on real ECCC weather + StatCan/AAFC yields.

    python scripts/train_yield_model.py

Downloads (and caches) ECCC daily weather and the StatCan yield table on first run,
assembles the training set, fits the configured model, prints validation metrics,
and saves the trained model + the assembled dataset to the artifacts directory.
"""

from __future__ import annotations

import pandas as pd

from canola_dt.config import load_config
from canola_dt.training import build_training_data
from canola_dt.models.yield_model import YieldModel


def main() -> None:
    pd.set_option("display.width", 160)
    cfg = load_config()

    print("== Building training set (ECCC + StatCan) — first run downloads data ==")
    data = build_training_data(cfg)
    n = len(data.X)
    print(f"  samples           : {n}")
    if n == 0:
        print("  No samples assembled — check station IDs / network. Aborting.")
        return
    print(f"  features          : {list(data.X.columns)}")
    print(f"  provinces x years : {data.meta['province'].nunique()} x "
          f"{data.meta['year'].nunique()} ({data.meta['year'].min()}-{data.meta['year'].max()})")
    print(f"  yield sources     : {data.meta['yield_source'].value_counts().to_dict()}")
    print(f"  yield range kg/ha : {data.y.min():.0f} - {data.y.max():.0f} "
          f"(mean {data.y.mean():.0f})")

    # Ablation: how much does each feature layer add? In particular, can the coupled
    # process-model (pm_*) outputs stand in for the hand-crafted weather features?
    pm_cols = [c for c in data.X.columns if c.startswith("pm_")]
    twin_cols = [c for c in data.X.columns if c not in pm_cols and c != "year"]

    def cv_r2(cols):
        m = YieldModel(cfg.model).cross_validate(data.X[cols], data.y)
        return m["cv_r2_mean"], m["cv_r2_std"]

    print("\n== Ablation: feature layers (5-fold CV R^2) ==")
    for label, cols in [
        ("year only (trend)", ["year"]),
        ("year + twin weather", ["year"] + twin_cols),
        ("year + process (pm_)", ["year"] + pm_cols),
        ("year + twin + process", ["year"] + twin_cols + pm_cols),
    ]:
        r2, sd = cv_r2(cols)
        print(f"  {label:<26}: {r2:.3f} +/- {sd:.3f}")
    full = YieldModel(cfg.model).cross_validate(data.X, data.y)

    print("\n== Training full model ==")
    model = YieldModel(cfg.model)
    metrics = model.fit(data.X, data.y)
    print(f"  model type        : {cfg.model.get('type')}")
    print(f"  {full['n_splits']}-fold CV R^2     : {full['cv_r2_mean']:.3f} +/- {full['cv_r2_std']:.3f}")
    print(f"  {full['n_splits']}-fold CV MAE     : {full['cv_mae_mean']:.1f} kg/ha")
    print(f"  holdout R^2       : {metrics['r2']:.3f}  (single {metrics['n_test']}-row split)")
    print(f"  holdout MAE       : {metrics['mae_kg_ha']:.1f} kg/ha")

    # Feature importances (tree models expose them).
    est = model.pipeline.named_steps["est"]
    if hasattr(est, "feature_importances_"):
        imp = (
            pd.Series(est.feature_importances_, index=model.feature_names)
            .sort_values(ascending=False)
            .head(8)
        )
        print("\n  top feature importances:")
        for name, val in imp.items():
            print(f"    {name:<24} {val:.3f}")

    # Persist artifacts.
    artifacts = cfg.path("artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)
    model_path = artifacts / "yield_model.joblib"
    model.save(model_path)
    dataset = pd.concat([data.meta, data.X, data.y.rename("yield_kg_ha")], axis=1)
    dataset_path = artifacts / "training_dataset.csv"
    dataset.to_csv(dataset_path, index=False)
    print(f"\n  saved model   -> {model_path}")
    print(f"  saved dataset -> {dataset_path}")


if __name__ == "__main__":
    main()
