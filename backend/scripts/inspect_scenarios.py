from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.services.model_runner import ModelRunner
from backend.app.services.scenario_engine import MACRO_SCENARIOS, ScenarioEngine


AUX_COLS = ["PM10", "NO2", "SO2", "O3", "CO", "temperature", "humidity", "pressure", "wind_speed"]
INTENSITIES_DEFAULT = [0, 10, 25, 50, 75, 100]


def _find_data_path() -> str:
    c1 = os.path.join("data", "Airware-Haikou", "2_filled_data")
    c2 = os.path.join("Airware-Haikou", "2_filled_data")
    for c in (c1, c2):
        if os.path.isdir(c):
            return c
    raise FileNotFoundError("Dataset path not found. Expected data/Airware-Haikou/2_filled_data")


def _generate_features_like_model(data: pd.DataFrame, clip_stats: tuple[float, float], aux_fill: dict, training_mode: bool) -> pd.DataFrame:
    data = data.copy()
    data = data.sort_values("station_ID", kind="mergesort").sort_index(kind="mergesort")
    data[AUX_COLS] = data.groupby("station_ID", group_keys=False)[AUX_COLS].ffill()
    data[AUX_COLS] = data[AUX_COLS].fillna(aux_fill)
    data["PM2.5"] = data["PM2.5"].astype(float)

    q1_local, q99_local = clip_stats
    if training_mode:
        data["PM2.5"] = data["PM2.5"].clip(q1_local, q99_local)

    g = data.groupby("station_ID", group_keys=False)["PM2.5"]

    data["base_lag1"] = g.shift(1)
    data["delta"] = data["PM2.5"] - data["base_lag1"]

    for lag in [1, 3, 6, 12, 24, 48, 72, 168]:
        data[f"lag{lag}"] = g.shift(lag)

    data["roll3"] = g.transform(lambda s: s.shift(1).rolling(3).mean())
    data["roll6"] = g.transform(lambda s: s.shift(1).rolling(6).mean())
    data["roll24"] = g.transform(lambda s: s.shift(1).rolling(24).mean())
    data["roll48"] = g.transform(lambda s: s.shift(1).rolling(48).mean())
    data["roll168"] = g.transform(lambda s: s.shift(1).rolling(168).mean())

    data["std6"] = g.transform(lambda s: s.shift(1).rolling(6).std())
    data["std24"] = g.transform(lambda s: s.shift(1).rolling(24).std())
    data["min24"] = g.transform(lambda s: s.shift(1).rolling(24).min())
    data["max24"] = g.transform(lambda s: s.shift(1).rolling(24).max())

    data["ewm6"] = g.transform(lambda s: s.shift(1).ewm(span=6, adjust=False).mean())
    data["ewm24"] = g.transform(lambda s: s.shift(1).ewm(span=24, adjust=False).mean())

    data["trend_24"] = data["lag1"] - data["lag24"]
    data["trend_168"] = data["lag1"] - data["lag168"]
    data["roll_diff_3_24"] = data["roll3"] - data["roll24"]
    data["roll_diff_24_168"] = data["roll24"] - data["roll168"]

    hour = data.index.hour
    day = data.index.dayofweek
    data["sin_hour"] = np.sin(2 * np.pi * hour / 24)
    data["cos_hour"] = np.cos(2 * np.pi * hour / 24)
    data["sin_day"] = np.sin(2 * np.pi * day / 7)
    data["cos_day"] = np.cos(2 * np.pi * day / 7)
    data["is_weekend"] = (day >= 5).astype(int)

    return data.sort_index()


def _load_eval_sample(meta: dict, sample_size: int, random_state: int):
    features = list(meta.get("features", []))
    data_path = _find_data_path()
    csv_files = sorted(glob.glob(os.path.join(data_path, "*.csv")))
    if not csv_files:
        raise RuntimeError(f"No CSV files found in {data_path}")

    df = pd.concat((pd.read_csv(p) for p in csv_files), ignore_index=True)
    df["datetime"] = pd.to_datetime(df["hours"], errors="coerce")
    df = df[~df["datetime"].isna()].copy()
    if "station_ID" not in df.columns:
        df["station_ID"] = "GLOBAL_STATION"
    df["station_ID"] = df["station_ID"].astype(str)
    df = df.sort_values(["station_ID", "datetime"]).set_index("datetime")
    df = df.drop(columns=["hours", "Unnamed: 0"], errors="ignore")

    unique_times = np.array(sorted(df.index.unique()))
    split_idx = int(len(unique_times) * 0.8)
    split_idx = min(max(split_idx, 1), len(unique_times) - 1)
    split_time = unique_times[split_idx]
    train_raw = df[df.index < split_time].copy()
    test_raw = df[df.index >= split_time].copy()

    q1, q99 = train_raw["PM2.5"].quantile([0.01, 0.99])
    clip_stats = (float(q1), float(q99))
    aux_fill = train_raw[AUX_COLS].ffill().mean().to_dict()

    full_feat = _generate_features_like_model(df.copy(), clip_stats=clip_stats, aux_fill=aux_fill, training_mode=False)
    test_keys = test_raw.reset_index()[["station_ID", "datetime"]].drop_duplicates()
    test_feat = (
        full_feat.reset_index()
        .merge(test_keys, on=["station_ID", "datetime"], how="inner", validate="one_to_one")
        .sort_values(["station_ID", "datetime"])
        .set_index("datetime")
    )
    test_feat = test_feat.dropna(subset=["base_lag1", "PM2.5"] + features)

    # Sanity check: no station-time duplication remains.
    pair_counts = (
        test_feat.reset_index()
        .groupby(["station_ID", "datetime"], as_index=False)
        .size()
    )
    if not (pair_counts["size"] == 1).all():
        raise RuntimeError("Duplicate station_ID-datetime pairs detected in aligned test frame.")

    work = test_feat.reset_index()[["station_ID", "datetime", "base_lag1", "PM2.5"] + features].copy()
    for c in AUX_COLS:
        if c in work.columns:
            work[c] = work[c].fillna(aux_fill.get(c, np.nan))

    n = min(int(sample_size), len(work))
    if n <= 0:
        raise RuntimeError("No test rows available for scenario inspection.")
    sampled = work.sample(n=n, random_state=int(random_state)).reset_index(drop=True)
    X_sample = sampled[features].copy()
    sample_ref = sampled[["station_ID", "datetime", "base_lag1", "PM2.5"]].copy()

    return X_sample, sample_ref, test_feat, aux_fill


def _scenario_feature_rank(importance_rows: list[dict], feature: str) -> tuple[int | None, float]:
    for i, row in enumerate(importance_rows, start=1):
        if row.get("feature") == feature:
            return i, float(row.get("pct", 0.0))
    return None, 0.0


def _card_level_stats(results: pd.DataFrame, intensities: list[int]) -> pd.DataFrame:
    rows = []
    for card_id in sorted(results["scenario_id"].unique()):
        card_df = results[results["scenario_id"] == card_id].copy()
        # Monotonic checks per sampled baseline.
        mono_abs = []
        mono_signed = []
        sat_75_100 = []

        for sample_id, g in card_df.groupby("sample_id"):
            g = g.sort_values("intensity")
            vals = g["delta_pm25"].to_numpy(dtype=float)
            abs_vals = np.abs(vals)
            mono_abs.append(bool(np.all(np.diff(abs_vals) >= -1e-9)))

            final_sign = np.sign(vals[-1]) if len(vals) else 0.0
            if final_sign == 0:
                mono_signed.append(bool(np.all(np.abs(vals) <= 1e-9)))
            else:
                mono_signed.append(bool(np.all(np.diff(final_sign * vals) >= -1e-9)))

            d75 = float(g[g["intensity"] == 75]["delta_pm25"].iloc[0]) if 75 in intensities else np.nan
            d100 = float(g[g["intensity"] == 100]["delta_pm25"].iloc[0]) if 100 in intensities else np.nan
            if np.isfinite(d75) and np.isfinite(d100):
                sat_75_100.append(bool((abs(d100) - abs(d75)) < 0.05))

        agg_100 = card_df[card_df["intensity"] == 100]
        rows.append(
            {
                "scenario_id": card_id,
                "mean_delta_at_100": float(agg_100["delta_pm25"].mean()) if len(agg_100) else np.nan,
                "mean_abs_delta_at_100": float(agg_100["abs_delta_pm25"].mean()) if len(agg_100) else np.nan,
                "p90_abs_delta_at_100": float(agg_100["abs_delta_pm25"].quantile(0.9)) if len(agg_100) else np.nan,
                "near_zero_rate_abs_lt_0.1_at_100": float((agg_100["abs_delta_pm25"] < 0.1).mean()) if len(agg_100) else np.nan,
                "near_zero_rate_abs_lt_0.5_at_100": float((agg_100["abs_delta_pm25"] < 0.5).mean()) if len(agg_100) else np.nan,
                "monotonic_abs_rate": float(np.mean(mono_abs)) if mono_abs else np.nan,
                "monotonic_signed_rate": float(np.mean(mono_signed)) if mono_signed else np.nan,
                "early_saturation_rate_75_to_100": float(np.mean(sat_75_100)) if sat_75_100 else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_abs_delta_at_100", ascending=False)


def main():
    parser = argparse.ArgumentParser(description="Inspect current scenario behavior without changing business logic.")
    parser.add_argument("--samples", type=int, default=120, help="Number of baseline rows sampled from aligned test frame.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--out-dir", type=str, default="backend/scripts/artifacts", help="Output directory for diagnostic files.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = ModelRunner()
    meta = runner.get_meta()
    scenario_engine = ScenarioEngine()
    cards = scenario_engine.list_cards()
    card_ids = [c["scenario_id"] for c in cards]

    X_sample, sample_ref, test_feat, _ = _load_eval_sample(meta=meta, sample_size=args.samples, random_state=args.seed)

    importance_rows = runner.feature_importance(top_n=len(meta.get("features", [])))
    importance_source = "global_shap_mean_abs" if bool(meta.get("global_shap_mean_abs")) else "xgboost_gain_fallback"
    scenario_features = sorted({k["feature"] for c in card_ids for k in MACRO_SCENARIOS[c]["knobs"]})

    rows = []
    feature_shift_rows = []

    for sample_i in range(len(X_sample)):
        base_row = X_sample.iloc[sample_i]
        base_X = base_row.to_frame().T
        row_ref = sample_ref.iloc[sample_i]
        ts = row_ref["datetime"]
        station_id = row_ref["station_ID"]
        base_lag1 = float(row_ref["base_lag1"])
        current_pm25 = float(row_ref["PM2.5"])
        baseline_pm25 = float(runner.predict(base_X, base_lag1=base_lag1, current_pm25=current_pm25)["pm25_t_plus_1"])

        for card_id in card_ids:
            for intensity in INTENSITIES_DEFAULT:
                scenario_req = SimpleNamespace(type="macro", scenario_id=card_id, intensity=intensity, items=None)
                scenario_X, applied, ood_events, _ = scenario_engine.apply(
                    scenario=scenario_req,
                    baseline_X=base_X,
                    meta=meta,
                )
                scenario_pm25 = float(runner.predict(scenario_X, base_lag1=base_lag1, current_pm25=current_pm25)["pm25_t_plus_1"])
                delta_pm25 = scenario_pm25 - baseline_pm25

                changed = [a for a in applied if abs(float(a["to"]) - float(a["from"])) > 1e-12]
                clamped = [a for a in applied if bool(a.get("clamped"))]
                mean_abs_shift = float(np.mean([abs(float(a["to"]) - float(a["from"])) for a in applied])) if applied else 0.0

                rows.append(
                    {
                        "sample_id": int(sample_i),
                        "timestamp": str(ts),
                        "station_id": str(station_id),
                        "scenario_id": card_id,
                        "intensity": int(intensity),
                        "baseline_pm25": baseline_pm25,
                        "scenario_pm25": scenario_pm25,
                        "delta_pm25": float(delta_pm25),
                        "abs_delta_pm25": float(abs(delta_pm25)),
                        "changed_feature_count": int(len(changed)),
                        "clamped_feature_count": int(len(clamped)),
                        "ood_event_count": int(len(ood_events)),
                        "mean_abs_feature_shift": mean_abs_shift,
                    }
                )

                for a in applied:
                    feature_shift_rows.append(
                        {
                            "sample_id": int(sample_i),
                            "timestamp": str(ts),
                            "station_id": str(station_id),
                            "scenario_id": card_id,
                            "intensity": int(intensity),
                            "feature": str(a["feature"]),
                            "from_value": float(a["from"]),
                            "to_value": float(a["to"]),
                            "signed_shift": float(a["to"] - a["from"]),
                            "abs_shift": float(abs(a["to"] - a["from"])),
                            "clamped": bool(a["clamped"]),
                            "reason": str(a["reason"]),
                        }
                    )

    results = pd.DataFrame(rows)
    feature_shifts = pd.DataFrame(feature_shift_rows)

    summary = (
        results.groupby(["scenario_id", "intensity"], as_index=False)
        .agg(
            mean_delta=("delta_pm25", "mean"),
            median_delta=("delta_pm25", "median"),
            mean_abs_delta=("abs_delta_pm25", "mean"),
            p90_abs_delta=("abs_delta_pm25", lambda s: float(s.quantile(0.9))),
            near_zero_abs_lt_0_1=("abs_delta_pm25", lambda s: float((s < 0.1).mean())),
            near_zero_abs_lt_0_5=("abs_delta_pm25", lambda s: float((s < 0.5).mean())),
            mean_changed_features=("changed_feature_count", "mean"),
            mean_clamped_features=("clamped_feature_count", "mean"),
            mean_ood_events=("ood_event_count", "mean"),
            mean_abs_feature_shift=("mean_abs_feature_shift", "mean"),
        )
        .sort_values(["scenario_id", "intensity"])
    )

    card_stats = _card_level_stats(results=results, intensities=INTENSITIES_DEFAULT)

    shift_summary = (
        feature_shifts.groupby(["scenario_id", "intensity", "feature"], as_index=False)
        .agg(
            mean_signed_shift=("signed_shift", "mean"),
            mean_abs_shift=("abs_shift", "mean"),
            clamped_rate=("clamped", "mean"),
        )
        .sort_values(["scenario_id", "intensity", "feature"])
    )

    importance_df = pd.DataFrame(importance_rows)
    scenario_feature_rank_rows = []
    for f in scenario_features:
        rank, pct = _scenario_feature_rank(importance_rows, f)
        scenario_feature_rank_rows.append(
            {
                "feature": f,
                "importance_rank": rank,
                "importance_pct": pct,
            }
        )
    scenario_feature_ranks_df = pd.DataFrame(scenario_feature_rank_rows).sort_values("importance_rank")

    results_path = out_dir / "scenario_diagnostic_runs.csv"
    summary_path = out_dir / "scenario_diagnostic_summary.csv"
    card_stats_path = out_dir / "scenario_diagnostic_card_stats.csv"
    shifts_path = out_dir / "scenario_diagnostic_feature_shifts.csv"
    importance_path = out_dir / "scenario_diagnostic_feature_importance.csv"
    scenario_rank_path = out_dir / "scenario_target_feature_ranks.csv"

    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    card_stats.to_csv(card_stats_path, index=False)
    shift_summary.to_csv(shifts_path, index=False)
    importance_df.to_csv(importance_path, index=False)
    scenario_feature_ranks_df.to_csv(scenario_rank_path, index=False)

    print("=== Scenario inspection complete ===")
    print(f"sample_rows={len(X_sample)}")
    print(f"test_rows_aligned={len(test_feat)}")
    print(f"importance_source={importance_source}")
    print("\nTop 12 model features by current importance source:")
    print(importance_df.head(12).to_string(index=False))
    print("\nScenario-targeted feature ranks:")
    print(scenario_feature_ranks_df.to_string(index=False))
    print("\nCard-level effect stats (sorted by mean_abs_delta_at_100):")
    print(card_stats.to_string(index=False))
    print("\nArtifacts:")
    print(results_path)
    print(summary_path)
    print(card_stats_path)
    print(shifts_path)
    print(importance_path)
    print(scenario_rank_path)


if __name__ == "__main__":
    main()
