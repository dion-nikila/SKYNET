from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


AUX_COLS = ["PM10", "NO2", "SO2", "O3", "CO", "temperature", "humidity", "pressure", "wind_speed"]
QUANTILE_LEVELS = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
QUANTILE_KEYS = {
    0.01: "q01",
    0.05: "q05",
    0.25: "q25",
    0.50: "q50",
    0.75: "q75",
    0.95: "q95",
    0.99: "q99",
}


def _root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_dataset(data_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(data_dir / "*.csv")))
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


def _time_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    unique_times = np.array(sorted(df.index.unique()))
    if len(unique_times) < 2:
        raise RuntimeError("Not enough timestamps for chronological split.")
    split_idx = int(len(unique_times) * 0.8)
    split_idx = min(max(split_idx, 1), len(unique_times) - 1)
    split_time = unique_times[split_idx]
    train_raw = df[df.index < split_time].copy()
    test_raw = df[df.index >= split_time].copy()
    return train_raw, test_raw


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


def _compute_feature_quantiles(train_feat: pd.DataFrame, features: List[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for f in features:
        series = pd.to_numeric(train_feat[f], errors="coerce")
        q_raw = series.quantile(QUANTILE_LEVELS).to_dict()
        row = {QUANTILE_KEYS[level]: float(q_raw.get(level, np.nan)) for level in QUANTILE_LEVELS}
        if any(not np.isfinite(v) for v in row.values()):
            raise RuntimeError(f"Quantiles for feature '{f}' contain non-finite values.")
        out[f] = row
    return out


def _compute_global_shap_mean_abs(model: xgb.Booster, X_ref: pd.DataFrame, max_rows: int, random_state: int) -> Dict[str, float]:
    if X_ref is None or len(X_ref) == 0:
        raise RuntimeError("Cannot compute global SHAP: empty feature matrix.")

    n_rows = min(int(max_rows), int(len(X_ref)))
    X_sample = X_ref.sample(n=n_rows, random_state=int(random_state)) if len(X_ref) > n_rows else X_ref.copy()

    contrib = None
    try:
        import shap  # type: ignore

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_sample)
        contrib = np.asarray(shap_vals, dtype=float)
    except Exception:
        dsample = xgb.DMatrix(X_sample, feature_names=list(X_sample.columns))
        contrib = np.asarray(model.predict(dsample, pred_contribs=True), dtype=float)

    if contrib.ndim == 1:
        contrib = contrib.reshape(1, -1)
    elif contrib.ndim == 3:
        contrib = contrib[:, :, 0]

    if contrib.shape[1] == (len(X_sample.columns) + 1):
        contrib = contrib[:, :-1]

    if contrib.shape[1] != len(X_sample.columns):
        raise RuntimeError("Global SHAP matrix does not match feature count.")

    mean_abs = np.abs(contrib).mean(axis=0).astype(float)
    out = {col: float(val) for col, val in zip(list(X_sample.columns), mean_abs)}
    if any((not np.isfinite(v)) for v in out.values()):
        raise RuntimeError("Global SHAP contains non-finite values.")
    return out


def _load_booster(meta: Dict, meta_path: Path) -> xgb.Booster:
    model = meta.get("model")
    if isinstance(model, xgb.Booster):
        return model

    model_path = meta.get("model_path")
    if not model_path:
        raise RuntimeError("Metadata does not contain model_path.")

    model_path = Path(model_path)
    if not model_path.is_absolute():
        model_path = meta_path.parent / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    booster = xgb.Booster()
    booster.load_model(str(model_path))
    return booster


def main():
    parser = argparse.ArgumentParser(description="Backfill runtime metadata with feature_quantiles and global_shap_mean_abs.")
    parser.add_argument("--model-meta", default=str(_root_dir() / "model" / "xgb_haikou_model_meta.pkl"))
    parser.add_argument("--data-dir", default=str(_root_dir() / "data" / "Airware-Haikou" / "2_filled_data"))
    parser.add_argument("--max-shap-rows", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    meta_path = Path(args.model_meta).resolve()
    data_dir = Path(args.data_dir).resolve()
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    meta = joblib.load(meta_path)
    features = list(meta.get("features", []))
    if not features:
        raise RuntimeError("Metadata has no feature list.")

    booster = _load_booster(meta, meta_path)
    df = _load_dataset(data_dir)
    train_raw, test_raw = _time_split(df)
    q1, q99 = train_raw["PM2.5"].quantile([0.01, 0.99])
    clip_stats = (float(q1), float(q99))
    aux_fill_global = train_raw[AUX_COLS].ffill().mean().to_dict()

    train_feat = _generate_features(train_raw, clip_stats=clip_stats, aux_fill=aux_fill_global, training_mode=True)
    required = ["delta", "base_lag1", "PM2.5"] + features
    train_feat = train_feat.dropna(subset=required)
    feature_quantiles = _compute_feature_quantiles(train_feat=train_feat, features=features)

    full_feat = _generate_features(df, clip_stats=clip_stats, aux_fill=aux_fill_global, training_mode=False)
    test_keys = test_raw.reset_index()[["station_ID", "datetime"]].drop_duplicates()
    test_feat = (
        full_feat.reset_index()
        .merge(test_keys, on=["station_ID", "datetime"], how="inner", validate="one_to_one")
        .sort_values(["station_ID", "datetime"])
        .set_index("datetime")
    )
    test_feat = test_feat.dropna(subset=["base_lag1", "PM2.5"] + features)
    X_test = test_feat[features].copy()
    for c in AUX_COLS:
        if c in X_test.columns:
            X_test[c] = X_test[c].fillna(aux_fill_global[c])

    global_shap_mean_abs = _compute_global_shap_mean_abs(
        model=booster,
        X_ref=X_test,
        max_rows=int(args.max_shap_rows),
        random_state=int(args.seed),
    )

    feature_set = set(features)
    if set(feature_quantiles.keys()) != feature_set:
        raise RuntimeError("feature_quantiles keys do not match metadata feature list.")
    if set(global_shap_mean_abs.keys()) != feature_set:
        raise RuntimeError("global_shap_mean_abs keys do not match metadata feature list.")

    meta["feature_quantiles"] = feature_quantiles
    meta["global_shap_mean_abs"] = global_shap_mean_abs
    meta["global_shap_sample_rows"] = int(min(len(X_test), int(args.max_shap_rows)))
    meta["metadata_backfill_utc"] = datetime.now(timezone.utc).isoformat()
    meta.pop("model", None)

    if args.dry_run:
        print("Dry run only. Metadata was not written.")
    else:
        backup = meta_path.with_suffix(meta_path.suffix + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        joblib.dump(joblib.load(meta_path), backup)
        joblib.dump(meta, meta_path)
        print(f"Backup written: {backup}")
        print(f"Updated metadata written: {meta_path}")

    print(
        json.dumps(
            {
                "features_count": len(features),
                "has_feature_quantiles": bool(meta.get("feature_quantiles")),
                "has_global_shap_mean_abs": bool(meta.get("global_shap_mean_abs")),
                "global_shap_sample_rows": int(meta.get("global_shap_sample_rows", 0)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
