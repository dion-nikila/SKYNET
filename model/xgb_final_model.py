# final model for now 
import os
import glob
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# toggles
SHOW_PLOTS = False
SAVE_PLOTS = False
PLOT_PREFIX = "delta_xgb_final"

plt = None
if SHOW_PLOTS or SAVE_PLOTS:
    import matplotlib

    if not SHOW_PLOTS:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt


warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
np.random.seed(SEED)
xgb.set_config(verbosity=0)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_from_root(path_value: str) -> str:
    p = Path(path_value)
    return str(p if p.is_absolute() else (PROJECT_ROOT / p))


MODEL_DIR = _resolve_from_root(os.getenv("SKYNET_MODEL_DIR", "model"))
MODEL_META_FILENAME = os.getenv("SKYNET_MODEL_META_FILENAME", "xgb_haikou_model_meta.pkl")
MODEL_NATIVE_FILENAME = os.getenv("SKYNET_MODEL_NATIVE_FILENAME", "xgb_model.json")
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_META_FILENAME)

FINAL_NUM_BOOST_ROUND = 4000
FINAL_EARLY_STOPPING_ROUNDS = 100
FINAL_XGB_PARAMS = {
    "objective": "reg:pseudohubererror",
    "eval_metric": "mae",
    "tree_method": "hist",
    "nthread": 1,
    "seed": SEED,
    "max_depth": 8,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5.0,
    "gamma": 0.5,
    "reg_alpha": 1.0,
    "reg_lambda": 1.0,
}

# Single promoted feature 
USE_EXOGENOUS_LAGROLL_FEATURES = True
USE_INTERACTION_FEATURES = False
ENABLE_GLOBAL_SHAP = os.getenv("SKYNET_ENABLE_GLOBAL_SHAP", "0") == "1"


AUX_COLS = ["PM10", "NO2", "SO2", "O3", "CO", "temperature", "humidity", "pressure", "wind_speed"]

BASE_FEATURES = [
    "lag1", "lag3", "lag6", "lag12", "lag24", "lag48", "lag72", "lag168",
    "roll3", "roll6", "roll24", "roll48", "roll168",
    "std6", "std24", "min24", "max24",
    "ewm6", "ewm24",
    "trend_24", "trend_168", "roll_diff_3_24", "roll_diff_24_168",
    "PM10", "NO2", "SO2", "O3", "CO",
    "temperature", "humidity", "pressure", "wind_speed",
    "sin_hour", "cos_hour", "sin_day", "cos_day", "is_weekend",
]

EXOGENOUS_LAGROLL_FEATURES = [
    "PM10_lag1", "PM10_lag3", "PM10_roll3", "PM10_roll24",
    "humidity_lag1", "humidity_roll3", "humidity_roll24",
    "wind_speed_lag1", "wind_speed_roll3", "wind_speed_roll24",
    "pressure_lag1", "pressure_roll3", "pressure_roll24",
]

INTERACTION_FEATURES = [
    "int_windspeedlag1_pm10lag1",
    "int_humiditylag1_temperaturelag1",
    "int_pressurelag1_windspeedlag1",
]


def _find_data_path():
    # finalized cleaned dataset
    candidates = [
        _resolve_from_root(os.path.join("data", "processed", "final_dataset", "final.csv")),
        _resolve_from_root(os.path.join("data", "processed", "final_dataset")),
        _resolve_from_root(os.path.join("data", "processed", "raw_cleaned_audit", "cleaned_station_files")),
    ]
    for c in candidates:
        if os.path.isfile(c) or os.path.isdir(c):
            return c
    return _resolve_from_root(os.path.join("data", "processed", "final_dataset", "final.csv"))


def _dataset_stage_from_path(data_path: str):
    norm = str(data_path).replace("\\", "/")
    if "data/processed/final_dataset/final.csv" in norm or norm.endswith("data/processed/final_dataset"):
        return (
            "processed/final_dataset/final.csv (final cleaned raw 10-station subset)",
            "Final curated dataset after raw cleaning, station QA, and subset selection.",
        )
    if "raw_cleaned_audit/cleaned_station_files" in norm:
        return (
            "raw_cleaned_audit/cleaned_station_files (cleaned raw 95-station stage)",
            "Dataset generated via raw-cleaning pipeline with station-wise QC, corruption filtering, and conservative imputation.",
        )
    return (norm, "Dataset stage inferred from resolved training path.")


def _read_csv_robust(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, on_bad_lines="skip")
    except TypeError:
        return pd.read_csv(path)


def _resolve_input_csv_files(data_path: str):
    if os.path.isfile(data_path):
        if str(data_path).lower().endswith(".csv"):
            return [data_path]
        raise RuntimeError(f"Input path is a file but not CSV: {data_path}")
    if os.path.isdir(data_path):
        return sorted(glob.glob(os.path.join(data_path, "*.csv")))
    raise RuntimeError(f"Input data path does not exist: {data_path}")


def _attach_model_from_path(meta: dict, model_meta_path: str | None = None) -> dict:
    """
    Ensure a Booster object is attached using metadata model_path when available.
    Keeps metadata loading compatible even when disk metadata is saved without
    embedded model pickle payload.
    """
    if not isinstance(meta, dict):
        return meta
    if meta.get("model") is not None:
        return meta

    model_path = meta.get("model_path")
    if not isinstance(model_path, str) or not model_path.strip():
        return meta

    p = model_path
    if not os.path.isabs(p):
        base_dir = os.path.dirname(str(model_meta_path or MODEL_PATH))
        p = os.path.join(base_dir, p)
    if not os.path.exists(p):
        return meta

    booster = xgb.Booster()
    booster.load_model(str(p))
    meta["model"] = booster
    return meta


def _save_or_show(fig, filename):
    if plt is None:
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    if SAVE_PLOTS:
        fig.savefig(os.path.join(MODEL_DIR, filename), bbox_inches="tight", dpi=160)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def _compute_global_shap_mean_abs(model, x_ref, max_rows=3000, random_state=SEED):
    """
    Compute global TreeSHAP importance as mean absolute contribution per feature.
    Uses SHAP TreeExplainer first, then XGBoost pred_contribs as fallback.
    """
    if x_ref is None or len(x_ref) == 0:
        return {}

    n_rows = min(int(max_rows), int(len(x_ref)))
    x_sample = x_ref.sample(n=n_rows, random_state=int(random_state)) if int(len(x_ref)) > n_rows else x_ref.copy()

    contrib = None
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(x_sample)
        contrib = np.asarray(shap_vals, dtype=float)
    except Exception:
        try:
            dsample = xgb.DMatrix(x_sample, feature_names=list(x_sample.columns))
            contrib = model.predict(dsample, pred_contribs=True)
            contrib = np.asarray(contrib, dtype=float)
        except Exception:
            return {}

    if contrib.ndim == 1:
        contrib = contrib.reshape(1, -1)
    elif contrib.ndim == 3:
        contrib = contrib[:, :, 0]

    if contrib.shape[1] == (len(x_sample.columns) + 1):
        contrib = contrib[:, :-1]

    if contrib.shape[1] != len(x_sample.columns):
        return {}

    mean_abs = np.abs(contrib).mean(axis=0).astype(float)
    return {col: float(val) for col, val in zip(list(x_sample.columns), mean_abs)}


def _directional_accuracy_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    return float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100.0)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mbe": float(np.mean(y_pred - y_true)),
        "directional_acc_pct": _directional_accuracy_pct(y_true, y_pred),
    }


def _safe_subset_metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> dict:
    n = int(mask.sum())
    if n <= 0:
        return {"count": 0, "mae": float("nan"), "rmse": float("nan"), "mbe": float("nan")}
    yt = y_true[mask]
    yp = y_pred[mask]
    return {
        "count": n,
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mbe": float(np.mean(yp - yt)),
    }


def _tail_diagnostics(y_true_test: np.ndarray, y_pred_test: np.ndarray, y_train_reference: np.ndarray) -> dict:
    """
    Tail diagnostics use thresholds derived only from pre-validation training PM2.5.
    """
    p90_thr = float(np.quantile(y_train_reference, 0.90))
    p95_thr = float(np.quantile(y_train_reference, 0.95))
    p90 = _safe_subset_metrics(y_true_test, y_pred_test, y_true_test >= p90_thr)
    p95 = _safe_subset_metrics(y_true_test, y_pred_test, y_true_test >= p95_thr)
    return {
        "train_threshold_p90": p90_thr,
        "train_threshold_p95": p95_thr,
        "p90": p90,
        "p95": p95,
    }


def _ensure_station_lag_features(data: pd.DataFrame, cols, lag: int = 1) -> pd.DataFrame:
    """
    Create per-station lagged inputs if absent.
    Used to keep interaction features explicit and temporally consistent.
    """
    data = data.copy()
    for col in cols:
        lag_col = f"{col}_lag{lag}"
        if lag_col not in data.columns:
            gcol = data.groupby("station_ID", group_keys=False)[col]
            data[lag_col] = gcol.shift(lag)
    return data


def _add_interaction_feature_set(data: pd.DataFrame) -> pd.DataFrame:
    """
    Interaction features use lagged (t-1) inputs only.
    """
    data = _ensure_station_lag_features(
        data=data,
        cols=["PM10", "humidity", "wind_speed", "pressure", "temperature"],
        lag=1,
    )
    data["int_windspeedlag1_pm10lag1"] = data["wind_speed_lag1"] * data["PM10_lag1"]
    data["int_humiditylag1_temperaturelag1"] = data["humidity_lag1"] * data["temperature_lag1"]
    data["int_pressurelag1_windspeedlag1"] = data["pressure_lag1"] * data["wind_speed_lag1"]
    return data


def train_and_save(
    data_path_override=None,
    station_subset=None,
    model_dir=None,
    model_meta_path=None,
    native_model_filename=None,
    experiment_name=None,
):
    print("No saved model found — running final training pipeline.")

    data_path = _resolve_from_root(str(data_path_override)) if data_path_override else _find_data_path()
    dataset_stage, dataset_stage_note = _dataset_stage_from_path(data_path)

    station_subset = [str(s) for s in station_subset] if station_subset else None
    station_subset_set = set(station_subset) if station_subset else None

    out_model_dir = _resolve_from_root(str(model_dir or MODEL_DIR))
    out_model_meta_path = (
        _resolve_from_root(str(model_meta_path))
        if model_meta_path
        else os.path.join(out_model_dir, MODEL_META_FILENAME)
    )
    native_model_filename = str(native_model_filename or MODEL_NATIVE_FILENAME)

    print(f"Resolved data path: {data_path}")
    if station_subset_set:
        print(f"Dataset mode: explicit subset ({len(station_subset_set)} stations).")
    else:
        print("Dataset mode: full dataset (no station subset restriction).")

    csv_files = _resolve_input_csv_files(data_path)
    if len(csv_files) == 0:
        raise SystemExit(f"No CSV files found in {data_path}")

    if station_subset_set:
        filtered = [
            p for p in csv_files
            if os.path.splitext(os.path.basename(p))[0] in station_subset_set
        ]
        if len(filtered) > 0:
            csv_files = filtered

    df = pd.concat((_read_csv_robust(p) for p in csv_files), ignore_index=True)
    n_rows_loaded = int(len(df))

    df["datetime"] = pd.to_datetime(df["hours"], errors="coerce")
    malformed_datetime_rows = int(df["datetime"].isna().sum())
    if malformed_datetime_rows > 0:
        print(f"Dropping {malformed_datetime_rows} row(s) with invalid datetime values during training load.")
    df = df[df["datetime"].notna()].copy()

    if "station_ID" not in df.columns:
        df["station_ID"] = "GLOBAL_STATION"
    df["station_ID"] = df["station_ID"].astype(str)

    if station_subset_set:
        df = df[df["station_ID"].isin(station_subset_set)].copy()
        if len(df) == 0:
            raise RuntimeError(
                "Station subset filtering removed all rows. "
                f"Requested stations: {sorted(station_subset_set)}"
            )

    df = df.sort_values(["station_ID", "datetime"]).set_index("datetime")
    df = df.drop(columns=["hours", "Unnamed: 0"], errors="ignore")

    # Chronological split across all stations (same cutoff timestamp for every station).
    unique_times = np.array(sorted(df.index.unique()))
    split_idx = int(len(unique_times) * 0.8)
    split_idx = min(max(split_idx, 1), len(unique_times) - 1)
    split_time = unique_times[split_idx]

    train_raw = df[df.index < split_time].copy()
    test_raw = df[df.index >= split_time].copy()

    # Training-only clipping stats and exogenous fallback.
    q1, q99 = train_raw["PM2.5"].quantile([0.01, 0.99])
    clip_stats = (float(q1), float(q99))
    aux_fill_global = train_raw[AUX_COLS].ffill().mean().to_dict()

    def generate_features(data, clip_stats_local, aux_fill, training_mode=False):
        data = data.copy()
        data = data.sort_values("station_ID", kind="mergesort").sort_index(kind="mergesort")

        data[AUX_COLS] = data.groupby("station_ID", group_keys=False)[AUX_COLS].ffill()
        data[AUX_COLS] = data[AUX_COLS].fillna(aux_fill)
        data["PM2.5"] = data["PM2.5"].astype(float)

        q1_local, q99_local = clip_stats_local
        if training_mode:
            data["PM2.5"] = data["PM2.5"].clip(q1_local, q99_local)

        g = data.groupby("station_ID", group_keys=False)["PM2.5"]
        aux_src = data[AUX_COLS].copy()

        # Delta target with strictly past base per station.
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

        # One-hour-ahead semantics: direct exogenous features at target hour t
        # must come from information available by t-1, matching runtime forecast rows.
        data[AUX_COLS] = data.groupby("station_ID", group_keys=False)[AUX_COLS].shift(1)

        if USE_EXOGENOUS_LAGROLL_FEATURES:
            for col in ["PM10", "humidity", "wind_speed", "pressure"]:
                gcol = aux_src.groupby(data["station_ID"], group_keys=False)[col]
                data[f"{col}_lag1"] = gcol.shift(1)
                data[f"{col}_roll3"] = gcol.transform(lambda s: s.shift(1).rolling(3).mean())
                data[f"{col}_roll24"] = gcol.transform(lambda s: s.shift(1).rolling(24).mean())
            g_pm10 = aux_src.groupby(data["station_ID"], group_keys=False)["PM10"]
            data["PM10_lag3"] = g_pm10.shift(3)

        if USE_INTERACTION_FEATURES:
            interaction_src = data.copy()
            interaction_src[AUX_COLS] = aux_src
            data = _add_interaction_feature_set(interaction_src)
            data[AUX_COLS] = interaction_src[AUX_COLS].groupby(interaction_src["station_ID"], group_keys=False).shift(1)

        hour = data.index.hour
        day = data.index.dayofweek
        data["sin_hour"] = np.sin(2 * np.pi * hour / 24)
        data["cos_hour"] = np.cos(2 * np.pi * hour / 24)
        data["sin_day"] = np.sin(2 * np.pi * day / 7)
        data["cos_day"] = np.cos(2 * np.pi * day / 7)
        data["is_weekend"] = (day >= 5).astype(int)

        return data.sort_index()

    features = list(BASE_FEATURES)
    if USE_EXOGENOUS_LAGROLL_FEATURES:
        features.extend(EXOGENOUS_LAGROLL_FEATURES)
    if USE_INTERACTION_FEATURES:
        features.extend(INTERACTION_FEATURES)

    train_feat = generate_features(train_raw, clip_stats_local=clip_stats, aux_fill=aux_fill_global, training_mode=True)

    required = ["delta", "base_lag1", "PM2.5"] + features
    train_feat = train_feat.dropna(subset=required).copy()

    feature_defaults = train_feat[features].median(numeric_only=True).to_dict()
    feature_quantiles = {}
    quantile_levels = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    quantile_keys = {0.01: "q01", 0.05: "q05", 0.25: "q25", 0.50: "q50", 0.75: "q75", 0.95: "q95", 0.99: "q99"}
    for f in features:
        q = train_feat[f].quantile(quantile_levels).to_dict()
        feature_quantiles[f] = {quantile_keys[level]: float(q.get(level, np.nan)) for level in quantile_levels}

    training_pm25_mean = float(train_feat["PM2.5"].mean())

    x_train = train_feat[features].copy()
    y_train = train_feat["delta"].copy()
    y_train_level = train_feat["PM2.5"].copy()
    base_train = train_feat["base_lag1"].copy()

    n_rows_train_raw = int(len(train_raw))
    n_rows_test_raw = int(len(test_raw))
    n_rows_train_feat = int(len(train_feat))
    train_row_retention_pct = float((n_rows_train_feat / n_rows_train_raw) * 100.0) if n_rows_train_raw > 0 else float("nan")

    def grouped_temporal_holdout(index: pd.Index, val_frac: float = 0.10, min_val_times: int = 24):
        unique_times_local = np.array(sorted(pd.Index(index).unique()))
        n_unique = int(len(unique_times_local))
        if n_unique < 2:
            raise RuntimeError("Not enough unique timestamps for grouped temporal holdout.")

        n_val_times = max(int(np.ceil(n_unique * float(val_frac))), int(min_val_times))
        n_val_times = min(max(1, n_val_times), n_unique - 1)

        train_times = unique_times_local[:-n_val_times]
        val_times = unique_times_local[-n_val_times:]

        idx_series = pd.Index(index)
        tr_rows = np.flatnonzero(idx_series.isin(train_times))
        val_rows = np.flatnonzero(idx_series.isin(val_times))
        if len(tr_rows) == 0 or len(val_rows) == 0:
            raise RuntimeError("Grouped temporal holdout produced empty train/validation rows.")

        info = {
            "train_unique_timestamps": int(len(train_times)),
            "val_unique_timestamps": int(len(val_times)),
            "val_fraction_unique_timestamps": float(n_val_times / n_unique),
            "train_start": str(train_times[0]),
            "train_end": str(train_times[-1]),
            "val_start": str(val_times[0]),
            "val_end": str(val_times[-1]),
        }
        return tr_rows, val_rows, info

    tr_idx_final, va_idx_final, final_val_info = grouped_temporal_holdout(
        x_train.index,
        val_frac=0.10,
        min_val_times=24,
    )

    x_tr = x_train.iloc[tr_idx_final].copy()
    x_va = x_train.iloc[va_idx_final].copy()
    y_tr = y_train.iloc[tr_idx_final].copy()
    y_va = y_train.iloc[va_idx_final].copy()

    # Tail thresholds come from pre-validation training rows only.
    y_train_reference = y_train_level.iloc[tr_idx_final].astype(float).values

    dtrain = xgb.DMatrix(x_tr, label=y_tr, feature_names=features)
    dval = xgb.DMatrix(x_va, label=y_va, feature_names=features)

    final_model = xgb.train(
        FINAL_XGB_PARAMS,
        dtrain,
        num_boost_round=int(FINAL_NUM_BOOST_ROUND),
        evals=[(dval, "val")],
        early_stopping_rounds=int(FINAL_EARLY_STOPPING_ROUNDS),
        verbose_eval=False,
    )

    # Global bias correction from validation only.
    pred_delta_va = final_model.predict(dval).astype(float)
    base_va = base_train.iloc[va_idx_final].astype(float).values
    y_va_level = y_train_level.iloc[va_idx_final].astype(float).values
    pred_va_level = base_va + pred_delta_va
    bias_correction = float(np.mean(y_va_level - pred_va_level))

    full_feat = generate_features(df.copy(), clip_stats_local=clip_stats, aux_fill=aux_fill_global, training_mode=False)

    # Align test rows by (station_ID, datetime) to avoid duplicate-timestamp expansion.
    test_keys = (
        test_raw.reset_index()[["station_ID", "datetime"]]
        .drop_duplicates(subset=["station_ID", "datetime"], keep="last")
    )
    full_feat_keys = (
        full_feat.reset_index()
        .sort_values(["station_ID", "datetime"], kind="mergesort")
        .drop_duplicates(subset=["station_ID", "datetime"], keep="last")
    )
    test_feat = (
        full_feat_keys
        .merge(test_keys, on=["station_ID", "datetime"], how="inner", validate="one_to_one")
        .sort_values(["station_ID", "datetime"])
        .set_index("datetime")
    )

    n_rows_test_aligned = int(len(test_feat))
    test_feat = test_feat.dropna(subset=["base_lag1", "PM2.5"] + features)
    n_rows_test_feat = int(len(test_feat))
    test_row_retention_pct = float((n_rows_test_feat / n_rows_test_raw) * 100.0) if n_rows_test_raw > 0 else float("nan")

    if len(test_feat) != len(test_keys):
        print(
            "Warning: test alignment retained "
            f"{len(test_feat)} / {len(test_keys)} rows after dedupe and feature-availability filtering."
        )

    x_test = test_feat[features].copy()
    for c in AUX_COLS:
        x_test[c] = x_test[c].fillna(aux_fill_global[c])

    if ENABLE_GLOBAL_SHAP:
        global_shap_mean_abs = _compute_global_shap_mean_abs(
            model=final_model,
            x_ref=x_test,
            max_rows=3000,
            random_state=SEED,
        )
    else:
        global_shap_mean_abs = {}

    dtest = xgb.DMatrix(x_test, feature_names=features)
    pred_delta = final_model.predict(dtest).astype(float)

    y_true = test_feat["PM2.5"].astype(float).values
    base = test_feat["base_lag1"].astype(float).values

    preds = base + pred_delta
    preds = preds + bias_correction

    metrics_test = _metrics(y_true, preds)

    # Baselines aligned to test rows.
    station_last_train = train_raw.groupby("station_ID")["PM2.5"].last().astype(float).to_dict()
    global_last_train = float(train_raw["PM2.5"].iloc[-1])

    persist = test_feat["lag1"].astype(float).values
    persist_nan = np.isnan(persist)
    if np.any(persist_nan):
        station_fill = test_feat.loc[persist_nan, "station_ID"].map(station_last_train).astype(float).values
        station_fill = np.where(np.isnan(station_fill), global_last_train, station_fill)
        persist[persist_nan] = station_fill

    roll24_test = test_feat["roll24"].astype(float).values
    roll24_filled = roll24_test.copy()
    nan_mask = np.isnan(roll24_filled)
    roll24_filled[nan_mask] = persist[nan_mask]

    baseline_persistence = _metrics(y_true, persist)
    baseline_roll24 = _metrics(y_true, roll24_filled)

    tail = _tail_diagnostics(
        y_true_test=y_true,
        y_pred_test=preds,
        y_train_reference=y_train_reference,
    )

    print("\nFinal Results (DELTA TARGET, robust loss + global validation bias correction):")
    print(f"MAE : {metrics_test['mae']:.4f}")
    print(f"RMSE: {metrics_test['rmse']:.4f}")
    print(f"R²  : {metrics_test['r2']:.4f}")
    print(f"MBE : {metrics_test['mbe']:.4f}")
    print(f"DirAcc: {metrics_test['directional_acc_pct']:.1f}%")

    print("\nBaseline comparison:")
    print(
        "Persistence - "
        f"MAE: {baseline_persistence['mae']:.4f}, "
        f"RMSE: {baseline_persistence['rmse']:.4f}, "
        f"R2: {baseline_persistence['r2']:.4f}"
    )
    print(
        "24h RollMean - "
        f"MAE: {baseline_roll24['mae']:.4f}, "
        f"RMSE: {baseline_roll24['rmse']:.4f}, "
        f"R2: {baseline_roll24['r2']:.4f}"
    )
    print(
        "XGBoost      - "
        f"MAE: {metrics_test['mae']:.4f}, "
        f"RMSE: {metrics_test['rmse']:.4f}, "
        f"R2: {metrics_test['r2']:.4f}"
    )

    print("\nTail diagnostics (thresholds derived from pre-validation training PM2.5 only):")
    print(
        f"P90 threshold={tail['train_threshold_p90']:.4f} | "
        f"count={tail['p90']['count']} | "
        f"MAE={tail['p90']['mae']:.4f} | RMSE={tail['p90']['rmse']:.4f} | MBE={tail['p90']['mbe']:.4f}"
    )
    print(
        f"P95 threshold={tail['train_threshold_p95']:.4f} | "
        f"count={tail['p95']['count']} | "
        f"MAE={tail['p95']['mae']:.4f} | RMSE={tail['p95']['rmse']:.4f} | MBE={tail['p95']['mbe']:.4f}"
    )

    print("\nRow retention diagnostics:")
    print(
        f"Train feature retention: {n_rows_train_feat}/{n_rows_train_raw} "
        f"({train_row_retention_pct:.2f}%)"
    )
    print(
        f"Test feature retention: {n_rows_test_feat}/{n_rows_test_raw} "
        f"({test_row_retention_pct:.2f}%)"
    )

    os.makedirs(out_model_dir, exist_ok=True)
    native_model_path = os.path.join(out_model_dir, native_model_filename)
    final_model.save_model(native_model_path)

    trained_at_utc = datetime.now(timezone.utc).isoformat()
    model_meta = {
        "schema_version": 5,
        "experiment_name": str(experiment_name) if experiment_name else None,
        "trained_at_utc": trained_at_utc,
        "data_path_used": data_path,
        "station_subset_requested": list(station_subset) if station_subset else None,
        "station_subset_applied": sorted(df["station_ID"].unique().tolist()) if station_subset_set else None,
        "subset_restriction_applied": bool(station_subset_set),
        "n_csv_files": int(len(csv_files)),
        "n_stations": int(df["station_ID"].nunique()),
        "n_rows_total": int(len(df)),
        "n_rows_loaded_raw": int(n_rows_loaded),
        "n_rows_invalid_datetime_dropped": int(malformed_datetime_rows),
        "n_rows_train": n_rows_train_raw,
        "n_rows_test": n_rows_test_raw,
        "n_rows_train_feat": n_rows_train_feat,
        "n_rows_train_fit": int(len(x_tr)),
        "n_rows_val_fit": int(len(x_va)),
        "n_rows_test_aligned_pre_dropna": n_rows_test_aligned,
        "n_rows_test_feat": n_rows_test_feat,
        "row_retention": {
            "train_feature_retention_pct": train_row_retention_pct,
            "test_feature_retention_pct": test_row_retention_pct,
        },
        "model": final_model,
        "model_path": native_model_filename,
        "model_dir": out_model_dir,
        "model_meta_path": out_model_meta_path,
        "features": features,
        "feature_config": {
            "use_exogenous_lagroll_features": bool(USE_EXOGENOUS_LAGROLL_FEATURES),
            "use_interaction_features": bool(USE_INTERACTION_FEATURES),
        },
        "final_xgb_params": dict(FINAL_XGB_PARAMS),
        "final_num_boost_round": int(FINAL_NUM_BOOST_ROUND),
        "final_early_stopping_rounds": int(FINAL_EARLY_STOPPING_ROUNDS),
        "best_iter": int(final_model.best_iteration if final_model.best_iteration is not None else FINAL_NUM_BOOST_ROUND),
        "target_type": "delta",
        "clip_q1_q99": (float(clip_stats[0]), float(clip_stats[1])),
        "aux_fill": aux_fill_global,
        "feature_defaults": feature_defaults,
        "feature_quantiles": feature_quantiles,
        "global_shap_mean_abs": global_shap_mean_abs,
        "global_shap_enabled": bool(ENABLE_GLOBAL_SHAP),
        "global_shap_sample_rows": int(min(3000, len(x_test))),
        "training_pm25_mean": training_pm25_mean,
        "bias_correction": {
            "type": "global_validation_mean_residual",
            "applicability": "applicable",
            "value": float(bias_correction),
        },
        "station_bias_coverage": {
            "applicability": "not_applicable",
            "station_coverage_pct": None,
            "row_coverage_pct": None,
            "low_coverage_flag": None,
        },
        "metrics_test": metrics_test,
        "baseline_metrics": {
            "persistence": baseline_persistence,
            "roll24": baseline_roll24,
        },
        "tail_diagnostics_test": tail,
        "time_coverage": {
            "full_start": str(df.index.min()),
            "full_end": str(df.index.max()),
            "train_start": str(train_raw.index.min()) if len(train_raw) > 0 else None,
            "train_end": str(train_raw.index.max()) if len(train_raw) > 0 else None,
            "test_start": str(test_raw.index.min()) if len(test_raw) > 0 else None,
            "test_end": str(test_raw.index.max()) if len(test_raw) > 0 else None,
            "test_feat_start": str(test_feat.index.min()) if len(test_feat) > 0 else None,
            "test_feat_end": str(test_feat.index.max()) if len(test_feat) > 0 else None,
        },
        "preprocessing": {
            "dataset_stage": str(dataset_stage),
            "dataset_stage_note": str(dataset_stage_note),
            "scaling_applied": False,
            "scaling_note": "Tree-based XGBoost model; no feature scaling was applied.",
            "outlier_policy": {
                "pm25_training_clip": "q01-q99 clip applied only during training-mode feature generation",
                "feature_quantiles": "q01/q05/q25/q50/q75/q95/q99 saved for runtime plausibility bounds",
                "runtime_clamping": "Scenario/custom interventions are quantile-bounded in backend scenario engine",
            },
        },
        "training_lineage": {
            "training_scheme": "xgboost_delta_target_next_hour_pm25",
            "horizon_hours": 1,
            "target_definition": "delta = PM2.5(t) - PM2.5(t-1)",
            "feature_availability": "each target-hour row t uses PM2.5 history and exogenous values available by t-1 only",
            "train_test_split_scheme": "global_timestamp_cutoff_80pct_unique_timestamps",
            "train_test_split_cutoff": str(split_time),
            "final_validation_scheme": "grouped_temporal_holdout_last_10pct_unique_timestamps",
            "final_validation": final_val_info,
            "training_mode": "locked_final_params",
            "artifact_generated_utc": trained_at_utc,
        },
    }

    # Persist a slim metadata artifact (without embedded Booster object).
    disk_model_meta = dict(model_meta)
    disk_model_meta.pop("model", None)
    joblib.dump(disk_model_meta, out_model_meta_path)
    print("\nModel and metadata saved to", out_model_meta_path)

    return model_meta


def load_model():
    if os.path.exists(MODEL_PATH):
        print("Loading saved final model meta from", MODEL_PATH)
        meta = joblib.load(MODEL_PATH)
        meta.setdefault("target_type", "delta")
        meta.setdefault("bias_correction", {"type": "global_validation_mean_residual", "applicability": "applicable", "value": 0.0})
        meta.setdefault("feature_defaults", {})
        meta.setdefault("feature_quantiles", {})
        meta.setdefault("global_shap_mean_abs", {})
        meta.setdefault("training_pm25_mean", None)
        return _attach_model_from_path(meta, model_meta_path=MODEL_PATH)
    return train_and_save()


if __name__ == "__main__":
    load_model()
