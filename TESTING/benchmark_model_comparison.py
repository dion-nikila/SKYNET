from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

AUX_COLS = ["PM10", "NO2", "SO2", "O3", "CO", "temperature", "humidity", "pressure", "wind_speed"]


def _root_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _find_data_dir(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    root = _root_dir()
    candidates = [
        root / "data" / "processed" / "final_dataset" / "final.csv",
        root / "data" / "processed" / "final_dataset",
        root / "data" / "processed" / "raw_cleaned_audit" / "cleaned_station_files",
    ]
    for c in candidates:
        if c.exists() and (c.is_dir() or c.is_file()):
            return c.resolve()
    raise FileNotFoundError("Could not locate final/cleaned dataset path.")


def _load_dataset(data_dir: Path) -> pd.DataFrame:
    if data_dir.is_file():
        if data_dir.suffix.lower() != ".csv":
            raise FileNotFoundError(f"Input path is a file but not CSV: {data_dir}")
        files = [str(data_dir)]
    elif data_dir.is_dir():
        files = sorted(glob.glob(str(data_dir / "*.csv")))
    else:
        files = []

    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = []
    for p in files:
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
    return df


def _time_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    unique_times = np.array(sorted(df.index.unique()))
    if len(unique_times) < 2:
        raise RuntimeError("Not enough timestamps for chronological split.")
    split_idx = int(len(unique_times) * 0.8)
    split_idx = min(max(split_idx, 1), len(unique_times) - 1)
    split_time = unique_times[split_idx]
    train_raw = df[df.index < split_time].copy()
    test_raw = df[df.index >= split_time].copy()
    return train_raw, test_raw, split_time


def _generate_features(
    data: pd.DataFrame,
    clip_stats: Tuple[float, float],
    aux_fill: Dict[str, float],
    training_mode: bool,
) -> pd.DataFrame:
    data = data.copy()
    data = data.sort_values("station_ID", kind="mergesort").sort_index(kind="mergesort")
    data[AUX_COLS] = data.groupby("station_ID", group_keys=False)[AUX_COLS].ffill()
    data[AUX_COLS] = data[AUX_COLS].fillna(aux_fill)
    aux_src = data[AUX_COLS].copy()
    data["PM2.5"] = data["PM2.5"].astype(float)

    if training_mode:
        q1, q99 = clip_stats
        data["PM2.5"] = data["PM2.5"].clip(float(q1), float(q99))

    g = data.groupby("station_ID", group_keys=False)["PM2.5"]
    data["base_lag1"] = g.shift(1)
    data["delta"] = data["PM2.5"] - data["base_lag1"]
    data[AUX_COLS] = data.groupby("station_ID", group_keys=False)[AUX_COLS].shift(1)

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

    # Exogenous lag/rolling features (aligned with locked final model feature space).
    for col in ["PM10", "humidity", "wind_speed", "pressure"]:
        gcol = aux_src.groupby(data["station_ID"], group_keys=False)[col]
        data[f"{col}_lag1"] = gcol.shift(1)
        data[f"{col}_roll3"] = gcol.transform(lambda s: s.shift(1).rolling(3).mean())
        data[f"{col}_roll24"] = gcol.transform(lambda s: s.shift(1).rolling(24).mean())
    g_pm10 = aux_src.groupby(data["station_ID"], group_keys=False)["PM10"]
    data["PM10_lag3"] = g_pm10.shift(3)

    # Keep interaction features available for metadata compatibility when present.
    g_temp = aux_src.groupby(data["station_ID"], group_keys=False)["temperature"]
    data["temperature_lag1"] = g_temp.shift(1)
    data["int_windspeedlag1_pm10lag1"] = data["wind_speed_lag1"] * data["PM10_lag1"]
    data["int_humiditylag1_temperaturelag1"] = data["humidity_lag1"] * data["temperature_lag1"]
    data["int_pressurelag1_windspeedlag1"] = data["pressure_lag1"] * data["wind_speed_lag1"]

    hour = data.index.hour
    day = data.index.dayofweek
    data["sin_hour"] = np.sin(2 * np.pi * hour / 24)
    data["cos_hour"] = np.cos(2 * np.pi * hour / 24)
    data["sin_day"] = np.sin(2 * np.pi * day / 7)
    data["cos_day"] = np.cos(2 * np.pi * day / 7)
    data["is_weekend"] = (day >= 5).astype(int)

    return data.sort_index()


def _directional_accuracy_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or len(y_pred) < 2:
        return float("nan")
    return float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100.0)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R²": float(r2_score(y_true, y_pred)),
        "MBE": float(np.mean(y_pred - y_true)),
        "Directional Accuracy (%)": _directional_accuracy_pct(y_true, y_pred),
    }


def _load_xgb_model(meta: Dict, meta_path: Path):
    model = meta.get("model")
    if model is not None:
        return model

    model_path = meta.get("model_path")
    if not model_path:
        raise RuntimeError("Model metadata does not contain model or model_path.")

    p = Path(model_path)
    if not p.is_absolute():
        p = meta_path.parent / p

    booster = xgb.Booster()
    booster.load_model(str(p))
    return booster


def _prepare_frames(meta: Dict, data_dir: Path):
    features = list(meta.get("features", []))
    if not features:
        raise RuntimeError("Model metadata has no feature list.")

    df = _load_dataset(data_dir)
    train_raw, test_raw, split_time = _time_split(df)

    q1, q99 = train_raw["PM2.5"].quantile([0.01, 0.99])
    clip_stats = (float(q1), float(q99))
    aux_fill_global = train_raw[AUX_COLS].ffill().mean().to_dict()

    train_feat = _generate_features(train_raw, clip_stats=clip_stats, aux_fill=aux_fill_global, training_mode=True)
    required = ["delta", "base_lag1", "PM2.5"] + features
    missing_train_cols = [c for c in features if c not in train_feat.columns]
    if missing_train_cols:
        raise RuntimeError(f"Generated training frame is missing required feature columns: {missing_train_cols}")
    train_feat = train_feat.dropna(subset=required)

    full_feat = _generate_features(df.copy(), clip_stats=clip_stats, aux_fill=aux_fill_global, training_mode=False)
    test_keys = test_raw.reset_index()[["station_ID", "datetime"]].drop_duplicates(
        subset=["station_ID", "datetime"], keep="last"
    )
    test_feat = (
        full_feat.reset_index()
        .sort_values(["station_ID", "datetime"], kind="mergesort")
        .drop_duplicates(subset=["station_ID", "datetime"], keep="last")
        .merge(test_keys, on=["station_ID", "datetime"], how="inner", validate="one_to_one")
        .sort_values(["station_ID", "datetime"])
        .set_index("datetime")
    )
    test_feat = test_feat.dropna(subset=["base_lag1", "PM2.5"] + features)

    X_train = train_feat[features].copy()
    y_train_delta = train_feat["delta"].astype(float).values

    X_test = test_feat[features].copy()
    for c in AUX_COLS:
        if c in X_test.columns:
            X_test[c] = X_test[c].fillna(aux_fill_global[c])
    y_test_level = test_feat["PM2.5"].astype(float).values
    base_test = test_feat["base_lag1"].astype(float).values

    return {
        "split_time": split_time,
        "train_raw": train_raw,
        "test_feat": test_feat,
        "X_train": X_train,
        "y_train_delta": y_train_delta,
        "X_test": X_test,
        "y_test_level": y_test_level,
        "base_test": base_test,
    }


def _run_benchmarks(meta: Dict, meta_path: Path, frames: Dict, rf_n_estimators: int, rf_max_depth: int):
    rows = []

    y_true = frames["y_test_level"]
    X_train = frames["X_train"]
    y_train_delta = frames["y_train_delta"]
    X_test = frames["X_test"]
    base_test = frames["base_test"].copy()

    train_raw = frames["train_raw"]
    station_last_train = train_raw.groupby("station_ID")["PM2.5"].last().astype(float).to_dict()
    global_last_train = float(train_raw["PM2.5"].iloc[-1])

    # Persistence baseline
    persist = frames["test_feat"]["lag1"].astype(float).values
    persist_nan = np.isnan(persist)
    if np.any(persist_nan):
        station_fill = frames["test_feat"].loc[persist_nan, "station_ID"].map(station_last_train).astype(float).values
        station_fill = np.where(np.isnan(station_fill), global_last_train, station_fill)
        persist[persist_nan] = station_fill
    m = _metrics(y_true, persist)
    rows.append({"Model": "Persistence baseline", **m, "Notes": "Past-hour PM2.5 baseline", "Status": "ok"})

    # Roll24 baseline
    roll24 = frames["test_feat"]["roll24"].astype(float).values
    roll24_filled = roll24.copy()
    nan_mask = np.isnan(roll24_filled)
    roll24_filled[nan_mask] = persist[nan_mask]
    m = _metrics(y_true, roll24_filled)
    rows.append({"Model": "Roll24 baseline", **m, "Notes": "24h rolling mean; falls back to persistence when unavailable", "Status": "ok"})

    # XGBoost deployed model
    try:
        xgb_model = _load_xgb_model(meta=meta, meta_path=meta_path)
        dtest = xgb.DMatrix(X_test, feature_names=list(X_test.columns))
        pred_delta = xgb_model.predict(dtest).astype(float)
        bias_raw = meta.get("bias_correction", 0.0)
        if isinstance(bias_raw, dict):
            bias = float(bias_raw.get("value", 0.0))
        else:
            bias = float(bias_raw)
        pred_level = base_test + pred_delta + bias
        m = _metrics(y_true, pred_level)
        rows.append(
            {
                "Model": "XGBoost / SKYNET",
                **m,
                "Notes": "Deployed delta-target model with stored bias correction",
                "Status": "ok",
            }
        )
    except Exception as exc:
        rows.append(
            {
                "Model": "XGBoost / SKYNET",
                "MAE": np.nan,
                "RMSE": np.nan,
                "R²": np.nan,
                "MBE": np.nan,
                "Directional Accuracy (%)": np.nan,
                "Notes": f"Failed to evaluate XGBoost: {exc.__class__.__name__}: {exc}",
                "Status": "failed",
            }
        )

    # Random Forest
    try:
        rf = RandomForestRegressor(
            n_estimators=int(rf_n_estimators),
            max_depth=int(rf_max_depth) if int(rf_max_depth) > 0 else None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        )
        rf.fit(X_train, y_train_delta)
        pred_delta = rf.predict(X_test).astype(float)
        pred_level = base_test + pred_delta
        m = _metrics(y_true, pred_level)
        rows.append(
            {
                "Model": "Random Forest Regressor",
                **m,
                "Notes": (
                    f"Delta-target RF (n_estimators={int(rf_n_estimators)}, "
                    f"max_depth={int(rf_max_depth) if int(rf_max_depth) > 0 else 'None'}); no post-hoc bias correction"
                ),
                "Status": "ok",
            }
        )
    except Exception as exc:
        rows.append(
            {
                "Model": "Random Forest Regressor",
                "MAE": np.nan,
                "RMSE": np.nan,
                "R²": np.nan,
                "MBE": np.nan,
                "Directional Accuracy (%)": np.nan,
                "Notes": f"Failed to evaluate Random Forest: {exc.__class__.__name__}: {exc}",
                "Status": "failed",
            }
        )

    # Decision Tree
    try:
        dt = DecisionTreeRegressor(
            max_depth=int(rf_max_depth) if int(rf_max_depth) > 0 else None,
            min_samples_leaf=2,
            random_state=42,
        )
        dt.fit(X_train, y_train_delta)
        pred_delta = dt.predict(X_test).astype(float)
        pred_level = base_test + pred_delta
        m = _metrics(y_true, pred_level)
        rows.append(
            {
                "Model": "Decision Tree Regressor",
                **m,
                "Notes": (
                    f"Delta-target decision tree (max_depth={int(rf_max_depth) if int(rf_max_depth) > 0 else 'None'}); "
                    "no post-hoc bias correction"
                ),
                "Status": "ok",
            }
        )
    except Exception as exc:
        rows.append(
            {
                "Model": "Decision Tree Regressor",
                "MAE": np.nan,
                "RMSE": np.nan,
                "R²": np.nan,
                "MBE": np.nan,
                "Directional Accuracy (%)": np.nan,
                "Notes": f"Failed to evaluate Decision Tree: {exc.__class__.__name__}: {exc}",
                "Status": "failed",
            }
        )

    order = {
        "XGBoost / SKYNET": 0,
        "Persistence baseline": 1,
        "Roll24 baseline": 2,
        "Random Forest Regressor": 3,
        "Decision Tree Regressor": 4,
    }
    out = pd.DataFrame(rows)
    out["_order"] = out["Model"].map(order).fillna(99)
    out = out.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    return out


def main():
    parser = argparse.ArgumentParser(description="Run reproducible benchmark comparison for SKYNET PM2.5 forecasting.")
    parser.add_argument("--model-meta", default=str(_root_dir() / "model" / "xgb_haikou_model_meta.pkl"))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out-csv", default=str(Path(__file__).resolve().parent / "benchmark_model_comparison_results.csv"))
    parser.add_argument("--rf-n-estimators", type=int, default=80)
    parser.add_argument("--rf-max-depth", type=int, default=20)
    args = parser.parse_args()

    meta_path = Path(args.model_meta).resolve()
    data_dir = _find_data_dir(Path(args.data_dir).resolve() if args.data_dir else None)
    out_csv = Path(args.out_csv).resolve()

    if not meta_path.exists():
        raise FileNotFoundError(f"Model metadata file not found: {meta_path}")

    meta = joblib.load(meta_path)
    frames = _prepare_frames(meta=meta, data_dir=data_dir)
    results = _run_benchmarks(
        meta=meta,
        meta_path=meta_path,
        frames=frames,
        rf_n_estimators=int(args.rf_n_estimators),
        rf_max_depth=int(args.rf_max_depth),
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_csv, index=False)
    print(f"Benchmark results written: {out_csv}")


if __name__ == "__main__":
    main()
