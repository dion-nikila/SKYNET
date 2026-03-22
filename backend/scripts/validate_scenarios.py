from __future__ import annotations

import argparse
import glob
import json
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
from backend.app.services.scenario_engine import ScenarioEngine


AUX_COLS = ["PM10", "NO2", "SO2", "O3", "CO", "temperature", "humidity", "pressure", "wind_speed"]
DEFAULT_INTENSITIES = [0, 10, 25, 50, 75, 100]


def _find_data_path() -> str:
    candidates = [
        os.path.join("data", "Airware-Haikou", "2_filled_data"),
        os.path.join("Airware-Haikou", "2_filled_data"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError("Dataset path not found. Expected data/Airware-Haikou/2_filled_data")


def _generate_features_like_model(data: pd.DataFrame, clip_stats: tuple[float, float], aux_fill: dict, training_mode: bool) -> pd.DataFrame:
    data = data.copy()
    data = data.sort_values("station_ID", kind="mergesort").sort_index(kind="mergesort")
    data[AUX_COLS] = data.groupby("station_ID", group_keys=False)[AUX_COLS].ffill()
    data[AUX_COLS] = data[AUX_COLS].fillna(aux_fill)
    data["PM2.5"] = data["PM2.5"].astype(float)

    if training_mode:
        q1, q99 = clip_stats
        data["PM2.5"] = data["PM2.5"].clip(float(q1), float(q99))

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


def _load_sample_rows(meta: dict, sample_size: int, random_state: int):
    features = list(meta.get("features", []))
    if not features:
        raise RuntimeError("Model metadata has no features list.")

    data_path = _find_data_path()
    csv_files = sorted(glob.glob(os.path.join(data_path, "*.csv")))
    if not csv_files:
        raise RuntimeError(f"No CSV files found in {data_path}")

    frames = []
    for p in csv_files:
        try:
            frames.append(pd.read_csv(p, on_bad_lines="skip"))
        except TypeError:
            frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
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

    pair_counts = test_feat.reset_index().groupby(["station_ID", "datetime"], as_index=False).size()
    if not (pair_counts["size"] == 1).all():
        raise RuntimeError("Duplicate station_ID-datetime pairs detected after alignment.")

    sampled = (
        test_feat.reset_index()[["station_ID", "datetime", "base_lag1", "PM2.5"] + features]
        .sample(n=min(int(sample_size), len(test_feat)), random_state=int(random_state))
        .reset_index(drop=True)
    )
    return sampled[features].copy(), sampled[["station_ID", "datetime", "base_lag1", "PM2.5"]].copy(), int(len(test_feat))


def summarize_results(results_df: pd.DataFrame, intensities: list[int], saturation_eps: float = 0.05) -> pd.DataFrame:
    card_rows = []
    for scenario_id, card_df in results_df.groupby("scenario_id"):
        card_df = card_df.sort_values(["sample_id", "intensity"])
        mono_abs = []
        saturation = []

        for _, grp in card_df.groupby("sample_id"):
            grp = grp.sort_values("intensity")
            abs_deltas = grp["abs_delta_pm25"].to_numpy(dtype=float)
            mono_abs.append(bool(np.all(np.diff(abs_deltas) >= -1e-9)))

            if 75 in intensities and 100 in intensities:
                d75 = float(grp[grp["intensity"] == 75]["abs_delta_pm25"].iloc[0])
                d100 = float(grp[grp["intensity"] == 100]["abs_delta_pm25"].iloc[0])
                saturation.append(bool((d100 - d75) < saturation_eps))

        at_100 = card_df[card_df["intensity"] == 100]
        card_rows.append(
            {
                "scenario_id": str(scenario_id),
                "effect_size_mean_abs_at_100": float(at_100["abs_delta_pm25"].mean()) if len(at_100) else np.nan,
                "effect_size_median_abs_at_100": float(at_100["abs_delta_pm25"].median()) if len(at_100) else np.nan,
                "effect_size_p90_abs_at_100": float(at_100["abs_delta_pm25"].quantile(0.9)) if len(at_100) else np.nan,
                "monotonicity_rate_abs": float(np.mean(mono_abs)) if mono_abs else np.nan,
                "saturation_rate_75_to_100": float(np.mean(saturation)) if saturation else np.nan,
                "ood_rate_any_at_100": float((at_100["ood_event_count"] > 0).mean()) if len(at_100) else np.nan,
                "ood_rate_any_all_intensities": float((card_df["ood_event_count"] > 0).mean()),
                "mean_ood_events_at_100": float(at_100["ood_event_count"].mean()) if len(at_100) else np.nan,
                "near_zero_rate_abs_lt_0_1_at_100": float((at_100["abs_delta_pm25"] < 0.1).mean()) if len(at_100) else np.nan,
            }
        )
    return pd.DataFrame(card_rows).sort_values("effect_size_mean_abs_at_100", ascending=False)


def main():
    parser = argparse.ArgumentParser(description="Validate scenario card behavior across fixed intensities.")
    parser.add_argument("--samples", type=int, default=200, help="Number of aligned test rows to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    parser.add_argument("--out-dir", type=str, default="backend/scripts/validation_artifacts", help="Output directory.")
    parser.add_argument("--soft-q", type=float, default=0.05, help="Requested OOD soft quantile.")
    parser.add_argument("--hard-q", type=float, default=0.01, help="Requested OOD hard quantile.")
    parser.add_argument("--saturation-eps", type=float, default=0.05, help="Absolute-delta growth threshold from 75->100.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = ModelRunner()
    meta = runner.get_meta()
    scenario_engine = ScenarioEngine()
    cards = scenario_engine.list_cards()
    card_ids = [c["scenario_id"] for c in cards]
    intensities = list(DEFAULT_INTENSITIES)

    X_sample, refs, n_eval_rows = _load_sample_rows(meta=meta, sample_size=args.samples, random_state=args.seed)
    ood_opts = {"soft_q": float(args.soft_q), "hard_q": float(args.hard_q)}

    run_rows = []
    shift_rows = []
    for i in range(len(X_sample)):
        base_X = X_sample.iloc[i].to_frame().T
        ref = refs.iloc[i]
        base_lag1 = float(ref["base_lag1"])
        current_pm25 = float(ref["PM2.5"])
        baseline_pred = float(runner.predict(base_X, base_lag1=base_lag1, current_pm25=current_pm25)["pm25_t_plus_1"])

        for sid in card_ids:
            for intensity in intensities:
                scenario_req = SimpleNamespace(type="macro", scenario_id=sid, intensity=intensity, items=None)
                scenario_X, applied, ood_events, out_id, ood_context = scenario_engine.apply(
                    scenario=scenario_req,
                    baseline_X=base_X,
                    meta=meta,
                    ood_opts=ood_opts,
                    return_context=True,
                )
                scenario_pred = float(runner.predict(scenario_X, base_lag1=base_lag1, current_pm25=current_pm25)["pm25_t_plus_1"])
                delta = float(scenario_pred - baseline_pred)

                changed = [a for a in applied if abs(float(a["to"]) - float(a["from"])) > 1e-12]
                clamped = [a for a in applied if bool(a.get("clamped"))]

                run_rows.append(
                    {
                        "sample_id": int(i),
                        "station_id": str(ref["station_ID"]),
                        "timestamp": str(ref["datetime"]),
                        "scenario_id": str(out_id),
                        "intensity": int(intensity),
                        "baseline_pm25": baseline_pred,
                        "scenario_pm25": scenario_pred,
                        "delta_pm25": delta,
                        "abs_delta_pm25": float(abs(delta)),
                        "changed_feature_count": int(len(changed)),
                        "clamped_feature_count": int(len(clamped)),
                        "ood_event_count": int(len(ood_events)),
                        "effective_soft_q": float(ood_context.get("effective_soft_q", args.soft_q)),
                        "effective_hard_q": float(ood_context.get("effective_hard_q", args.hard_q)),
                    }
                )

                for a in applied:
                    shift_rows.append(
                        {
                            "sample_id": int(i),
                            "scenario_id": str(out_id),
                            "intensity": int(intensity),
                            "feature": str(a["feature"]),
                            "from_value": float(a["from"]),
                            "to_value": float(a["to"]),
                            "abs_shift": float(abs(a["to"] - a["from"])),
                            "clamped": bool(a.get("clamped", False)),
                            "reason": str(a.get("reason", "")),
                        }
                    )

    results_df = pd.DataFrame(run_rows).sort_values(["scenario_id", "sample_id", "intensity"])
    shifts_df = pd.DataFrame(shift_rows).sort_values(["scenario_id", "sample_id", "intensity", "feature"])
    summary_df = summarize_results(results_df, intensities=intensities, saturation_eps=float(args.saturation_eps))

    results_path = out_dir / "scenario_validation_runs.csv"
    shifts_path = out_dir / "scenario_validation_feature_shifts.csv"
    summary_path = out_dir / "scenario_validation_summary.csv"
    summary_json_path = out_dir / "scenario_validation_summary.json"

    results_df.to_csv(results_path, index=False)
    shifts_df.to_csv(shifts_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    summary_payload = {
        "samples_used": int(len(X_sample)),
        "aligned_eval_rows": int(n_eval_rows),
        "intensities": intensities,
        "requested_ood": ood_opts,
        "cards": summary_df.to_dict(orient="records"),
    }
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print("=== Scenario validation summary ===")
    print(f"samples_used={len(X_sample)} aligned_eval_rows={n_eval_rows}")
    print(summary_df.to_string(index=False))
    print("\nArtifacts:")
    print(results_path)
    print(shifts_path)
    print(summary_path)
    print(summary_json_path)


if __name__ == "__main__":
    main()
