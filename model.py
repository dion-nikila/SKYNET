#final model for now
import os
import glob
import warnings
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import matplotlib

try:
    import optuna
except Exception:
    optuna = None

# toggles
SHOW_PLOTS = False    # True in Jupyter, false in server/.py runs
SAVE_PLOTS = False   # True to save PNGs in /model
PLOT_PREFIX = "delta_xgb"

if not SHOW_PLOTS:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore", category=FutureWarning)
if optuna is not None:
    warnings.filterwarnings("ignore", category=UserWarning, module="optuna")

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_haikou_model_meta.pkl")


def _find_data_path():
    # SKYNET trains from Airware-Haikou 2_filled_data (interpolated stage),
    # not directly from 1_row_data and not from pre-split 3_MTSAM artifacts.
    candidate1 = os.path.join("data", "Airware-Haikou", "2_filled_data")
    candidate2 = os.path.join("Airware-Haikou", "2_filled_data")
    for c in (candidate1, candidate2):
        if os.path.isdir(c):
            return c
    return "Airware-Haikou/2_filled_data"


def _read_csv_robust(path: str) -> pd.DataFrame:
    # Keep training resilient to occasional malformed trailing lines.
    try:
        return pd.read_csv(path, on_bad_lines="skip")
    except TypeError:
        return pd.read_csv(path)


def _save_or_show(fig, filename):
    os.makedirs(MODEL_DIR, exist_ok=True)
    if SAVE_PLOTS:
        fig.savefig(os.path.join(MODEL_DIR, filename), bbox_inches="tight", dpi=160)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def _compute_global_shap_mean_abs(model, X_ref, max_rows=3000, random_state=42):
    """
    Compute global TreeSHAP importance as mean absolute contribution per feature.
    Uses SHAP TreeExplainer first, then XGBoost pred_contribs as fallback.
    """
    if X_ref is None or len(X_ref) == 0:
        return {}

    n_rows = min(int(max_rows), int(len(X_ref)))
    if int(len(X_ref)) > n_rows:
        X_sample = X_ref.sample(n=n_rows, random_state=int(random_state))
    else:
        X_sample = X_ref.copy()

    contrib = None
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_sample)
        contrib = np.asarray(shap_vals, dtype=float)
    except Exception:
        # Fallback to XGBoost pred_contribs (TreeSHAP) if SHAP package path fails.
        try:
            dsample = xgb.DMatrix(X_sample, feature_names=list(X_sample.columns))
            contrib = model.predict(dsample, pred_contribs=True)
            contrib = np.asarray(contrib, dtype=float)
        except Exception:
            return {}

    if contrib.ndim == 1:
        contrib = contrib.reshape(1, -1)
    elif contrib.ndim == 3:
        contrib = contrib[:, :, 0]

    # XGBoost appends one bias term column for pred_contribs.
    if contrib.shape[1] == (len(X_sample.columns) + 1):
        contrib = contrib[:, :-1]

    if contrib.shape[1] != len(X_sample.columns):
        return {}

    mean_abs = np.abs(contrib).mean(axis=0).astype(float)
    return {col: float(val) for col, val in zip(list(X_sample.columns), mean_abs)}


def train_and_save():
    if optuna is None:
        raise RuntimeError("optuna is required for training; install with `pip install optuna`.")

    print("No saved model found — running full training pipeline.")
    DATA_PATH = _find_data_path()

    csv_files = sorted(glob.glob(os.path.join(DATA_PATH, "*.csv")))
    if len(csv_files) == 0:
        raise SystemExit(f"No CSV files found in {DATA_PATH}")

    # Train on all available CSV files.
    # NOTE: We intentionally do not apply feature scaling (StandardScaler/MinMaxScaler).
    # The deployed estimator is tree-based XGBoost, where split logic is typically scale-insensitive.
    df = pd.concat((_read_csv_robust(p) for p in csv_files), ignore_index=True)
    n_rows_loaded = int(len(df))
    df["datetime"] = pd.to_datetime(df["hours"], errors="coerce")
    malformed_datetime_rows = int(df["datetime"].isna().sum())
    if malformed_datetime_rows > 0:
        print(f"Dropping {malformed_datetime_rows} row(s) with invalid datetime values during training load.")
    df = df[~df["datetime"].isna()].copy()
    if "station_ID" not in df.columns:
        df["station_ID"] = "GLOBAL_STATION"
    df["station_ID"] = df["station_ID"].astype(str)
    df = df.sort_values(["station_ID", "datetime"]).set_index("datetime")
    df = df.drop(columns=["hours", "Unnamed: 0"], errors="ignore")

    # Time-based split across all stations (same cutoff timestamp for every station).
    unique_times = np.array(sorted(df.index.unique()))
    split_idx = int(len(unique_times) * 0.8)
    split_idx = min(max(split_idx, 1), len(unique_times) - 1)
    split_time = unique_times[split_idx]
    train_raw = df[df.index < split_time].copy()
    test_raw = df[df.index >= split_time].copy()

    aux_cols = ["PM10", "NO2", "SO2", "O3", "CO", "temperature", "humidity", "pressure", "wind_speed"]

    # clip stats from training only (no leakage)
    # This is the explicit training-time outlier policy for PM2.5 target construction.
    q1, q99 = train_raw["PM2.5"].quantile([0.01, 0.99])
    clip_stats = (float(q1), float(q99))

    # aux fill from training only (no leakage to test)
    aux_fill_global = train_raw[aux_cols].ffill().mean().to_dict()

    def generate_features(data, clip_stats, aux_fill, training_mode=False):
        data = data.copy()
        data = data.sort_values("station_ID", kind="mergesort").sort_index(kind="mergesort")

        # fill exogenous features station-wise first, then global train means
        data[aux_cols] = data.groupby("station_ID", group_keys=False)[aux_cols].ffill()
        data[aux_cols] = data[aux_cols].fillna(aux_fill)

        data["PM2.5"] = data["PM2.5"].astype(float)

        q1_local, q99_local = clip_stats
        if training_mode:
            data["PM2.5"] = data["PM2.5"].clip(q1_local, q99_local)

        g = data.groupby("station_ID", group_keys=False)["PM2.5"]

        # base + delta target (strictly past-only base, per station)
        data["base_lag1"] = g.shift(1)
        data["delta"] = data["PM2.5"] - data["base_lag1"]

        # lags (past-only, per station)
        for lag in [1, 3, 6, 12, 24, 48, 72, 168]:
            data[f"lag{lag}"] = g.shift(lag)

        # rolling means (past-only by shift(1), per station)
        data["roll3"] = g.transform(lambda s: s.shift(1).rolling(3).mean())
        data["roll6"] = g.transform(lambda s: s.shift(1).rolling(6).mean())
        data["roll24"] = g.transform(lambda s: s.shift(1).rolling(24).mean())
        data["roll48"] = g.transform(lambda s: s.shift(1).rolling(48).mean())
        data["roll168"] = g.transform(lambda s: s.shift(1).rolling(168).mean())

        # volatility/regime (past-only, per station)
        data["std6"] = g.transform(lambda s: s.shift(1).rolling(6).std())
        data["std24"] = g.transform(lambda s: s.shift(1).rolling(24).std())
        data["min24"] = g.transform(lambda s: s.shift(1).rolling(24).min())
        data["max24"] = g.transform(lambda s: s.shift(1).rolling(24).max())

        # EWM (past-only, per station)
        data["ewm6"] = g.transform(lambda s: s.shift(1).ewm(span=6, adjust=False).mean())
        data["ewm24"] = g.transform(lambda s: s.shift(1).ewm(span=24, adjust=False).mean())

        # trend features (past-only because all inputs are shifted/rolled)
        data["trend_24"] = data["lag1"] - data["lag24"]
        data["trend_168"] = data["lag1"] - data["lag168"]
        data["roll_diff_3_24"] = data["roll3"] - data["roll24"]
        data["roll_diff_24_168"] = data["roll24"] - data["roll168"]

        # time features
        hour = data.index.hour
        day = data.index.dayofweek
        data["sin_hour"] = np.sin(2 * np.pi * hour / 24)
        data["cos_hour"] = np.cos(2 * np.pi * hour / 24)
        data["sin_day"] = np.sin(2 * np.pi * day / 7)
        data["cos_day"] = np.cos(2 * np.pi * day / 7)
        data["is_weekend"] = (day >= 5).astype(int)

        return data.sort_index()

    features = [
        # level lags/rolls
        "lag1", "lag3", "lag6", "lag12", "lag24", "lag48", "lag72", "lag168",
        "roll3", "roll6", "roll24", "roll48", "roll168",
        # volatility/regime
        "std6", "std24", "min24", "max24",
        # ewm
        "ewm6", "ewm24",
        # trend
        "trend_24", "trend_168", "roll_diff_3_24", "roll_diff_24_168",
        # exogenous (assumed available at forecast time t)
        "PM10", "NO2", "SO2", "O3", "CO",
        "temperature", "humidity", "pressure", "wind_speed",
        # time
        "sin_hour", "cos_hour", "sin_day", "cos_day", "is_weekend",
    ]

    # build training feature frame using training only (no peeking into test)
    train_feat = generate_features(train_raw, clip_stats=clip_stats, aux_fill=aux_fill_global, training_mode=True)

    required = ["delta", "base_lag1", "PM2.5"] + features
    train_feat = train_feat.dropna(subset=required)
    feature_defaults = train_feat[features].median(numeric_only=True).to_dict()
    feature_quantiles = {}
    quantile_levels = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    quantile_keys = {
        0.01: "q01",
        0.05: "q05",
        0.25: "q25",
        0.50: "q50",
        0.75: "q75",
        0.95: "q95",
        0.99: "q99",
    }
    for f in features:
        q = train_feat[f].quantile(quantile_levels).to_dict()
        feature_quantiles[f] = {
            quantile_keys[level]: float(q.get(level, np.nan))
            for level in quantile_levels
        }
    training_pm25_mean = float(train_feat["PM2.5"].mean())

    X_train = train_feat[features].copy()
    y_train = train_feat["delta"].copy()
    y_train_level = train_feat["PM2.5"].copy()
    base_train = train_feat["base_lag1"].copy()

    def grouped_temporal_splits(index: pd.Index, n_splits: int = 3):
        """
        Build leakage-resistant CV folds by unique timestamp, not by row position.

        Why this matters:
        - The dataset can contain multiple stations per timestamp.
        - Row-wise TimeSeriesSplit can place same-timestamp rows in both train/val.
        - That leaks cross-station same-time context into validation folds.
        """
        unique_times = np.array(sorted(pd.Index(index).unique()))
        if len(unique_times) <= int(n_splits):
            raise RuntimeError(
                f"Not enough unique timestamps ({len(unique_times)}) for grouped temporal CV with n_splits={n_splits}."
            )

        tscv_times = TimeSeriesSplit(n_splits=int(n_splits))
        idx_series = pd.Index(index)
        splits = []
        for tr_time_idx, val_time_idx in tscv_times.split(unique_times):
            tr_times = unique_times[tr_time_idx]
            val_times = unique_times[val_time_idx]
            tr_rows = np.flatnonzero(idx_series.isin(tr_times))
            val_rows = np.flatnonzero(idx_series.isin(val_times))
            if len(tr_rows) == 0 or len(val_rows) == 0:
                continue
            splits.append((tr_rows, val_rows))

        if not splits:
            raise RuntimeError("Grouped temporal CV produced no valid folds.")
        return splits

    def grouped_temporal_holdout(index: pd.Index, val_frac: float = 0.10, min_val_times: int = 24):
        """
        Build a leakage-resistant final train/validation holdout by unique timestamp.

        Why this matters:
        - After grouped CV tuning, the final early-stopping split should follow
          the same timestamp-grouped rule.
        - Row-index splits can place same-timestamp rows into both train/val when
          multiple stations share timestamps.
        """
        unique_times = np.array(sorted(pd.Index(index).unique()))
        n_unique = int(len(unique_times))
        if n_unique < 2:
            raise RuntimeError("Not enough unique timestamps for grouped temporal holdout.")

        n_val_times = max(int(np.ceil(n_unique * float(val_frac))), int(min_val_times))
        n_val_times = min(max(1, n_val_times), n_unique - 1)

        train_times = unique_times[:-n_val_times]
        val_times = unique_times[-n_val_times:]

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

    cv_row_splits = grouped_temporal_splits(X_train.index, n_splits=3)

    def objective(trial):
        params = {
            "objective": "reg:pseudohubererror",
            "eval_metric": "mae",
            "tree_method": "hist",
            "seed": 42,
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        }

        fold_scores = []
        best_iters = []

        for tr_idx, val_idx in cv_row_splits:
            X_tr = X_train.iloc[tr_idx].copy()
            X_val = X_train.iloc[val_idx].copy()

            y_tr = y_train.iloc[tr_idx].copy()
            y_val = y_train.iloc[val_idx].copy()

            dtr = xgb.DMatrix(X_tr, label=y_tr, feature_names=features)
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)

            m = xgb.train(
                params,
                dtr,
                num_boost_round=4000,
                evals=[(dval, "val")],
                early_stopping_rounds=80,
                verbose_eval=False,
            )

            it = max(1, int(m.best_iteration) if m.best_iteration is not None else 1)
            best_iters.append(it)

            # score on reconstructed LEVEL (this matches your final evaluation target)
            pred_delta = m.predict(dval).astype(float)
            base_val = base_train.iloc[val_idx].astype(float).values
            y_val_level = y_train_level.iloc[val_idx].astype(float).values
            pred_level = base_val + pred_delta

            fold_scores.append(mean_absolute_error(y_val_level, pred_level))

        trial.set_user_attr("best_iter", int(np.median(best_iters)))
        return float(np.mean(fold_scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    best_params = study.best_params
    best_params.update({
        "objective": "reg:pseudohubererror",
        "eval_metric": "mae",
        "tree_method": "hist",
        "seed": 42
    })
    best_iter = max(1, int(study.best_trial.user_attrs.get("best_iter", 300)))

    # Final fit with grouped temporal holdout validation.
    # This avoids same-timestamp row leakage across stations during early stopping.
    tr_idx_final, va_idx_final, final_val_info = grouped_temporal_holdout(
        X_train.index,
        val_frac=0.10,
        min_val_times=24,
    )

    X_tr = X_train.iloc[tr_idx_final].copy()
    X_va = X_train.iloc[va_idx_final].copy()
    y_tr = y_train.iloc[tr_idx_final].copy()
    y_va = y_train.iloc[va_idx_final].copy()

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=features)
    dval = xgb.DMatrix(X_va, label=y_va, feature_names=features)

    final_model = xgb.train(
        best_params,
        dtrain,
        num_boost_round=4000,          # no forced minimum; early stopping decides
        evals=[(dval, "val")],
        early_stopping_rounds=100,
        verbose_eval=False,
    )

    # bias correction estimated ONLY on validation slice (no test leakage)
    pred_delta_va = final_model.predict(dval).astype(float)
    base_va = base_train.iloc[va_idx_final].astype(float).values
    y_va_level = y_train_level.iloc[va_idx_final].astype(float).values
    pred_va_level = base_va + pred_delta_va
    bias_correction = float(np.mean(y_va_level - pred_va_level))  # add to preds to reduce bias

    # fast test evaluation (no per-row loop; no leakage because all features are past-only via shift/rolling)
    full_feat = generate_features(df.copy(), clip_stats=clip_stats, aux_fill=aux_fill_global, training_mode=False)

    # Align test rows by (station_ID, datetime) to avoid duplicate-label cross expansion
    # from datetime-only indexing when timestamps repeat across stations.
    test_keys = (
        test_raw.reset_index()[["station_ID", "datetime"]]
        .drop_duplicates()
    )
    test_feat = (
        full_feat.reset_index()
        .merge(test_keys, on=["station_ID", "datetime"], how="inner", validate="one_to_one")
        .sort_values(["station_ID", "datetime"])
        .set_index("datetime")
    )
    test_feat = test_feat.dropna(subset=["base_lag1", "PM2.5"] + features)
    assert len(test_feat) == len(test_keys), "Unexpected duplication after test alignment"

    X_test = test_feat[features].copy()
    for c in aux_cols:
        X_test[c] = X_test[c].fillna(aux_fill_global[c])

    global_shap_mean_abs = _compute_global_shap_mean_abs(
        model=final_model,
        X_ref=X_test,
        max_rows=3000,
        random_state=42,
    )

    dtest = xgb.DMatrix(X_test, feature_names=features)
    pred_delta = final_model.predict(dtest).astype(float)

    y_true = test_feat["PM2.5"].astype(float).values
    base = test_feat["base_lag1"].astype(float).values

    preds = base + pred_delta
    preds = preds + bias_correction

    mae = mean_absolute_error(y_true, preds)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    r2 = r2_score(y_true, preds)
    mbe = float(np.mean(preds - y_true))
    dir_acc = float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(preds))) * 100.0) if len(y_true) >= 2 else float("nan")

    print("\nFinal Results (DELTA TARGET, robust loss + bias correction):")
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²  : {r2:.4f}")
    print(f"MBE : {mbe:.4f}")
    print(f"DirAcc: {dir_acc:.1f}%")

    # baselines aligned to test_feat rows (station-aware, no leakage)
    station_last_train = train_raw.groupby("station_ID")["PM2.5"].last().astype(float).to_dict()
    global_last_train = float(train_raw["PM2.5"].iloc[-1])

    persist = test_feat["lag1"].astype(float).values
    persist_nan = np.isnan(persist)
    if np.any(persist_nan):
        station_fill = test_feat.loc[persist_nan, "station_ID"].map(station_last_train).astype(float).values
        station_fill = np.where(np.isnan(station_fill), global_last_train, station_fill)
        persist[persist_nan] = station_fill

    mae_persist = mean_absolute_error(y_true, persist)
    rmse_persist = np.sqrt(mean_squared_error(y_true, persist))
    r2_persist = r2_score(y_true, persist)

    roll24_test = test_feat["roll24"].astype(float).values

    roll24_filled = roll24_test.copy()
    nan_mask = np.isnan(roll24_filled)
    roll24_filled[nan_mask] = persist[nan_mask]

    mae_roll24 = mean_absolute_error(y_true, roll24_filled)
    rmse_roll24 = np.sqrt(mean_squared_error(y_true, roll24_filled))
    r2_roll24 = r2_score(y_true, roll24_filled)

    print("\nBaseline comparison:")
    print(f"Persistence - MAE: {mae_persist:.4f}, RMSE: {rmse_persist:.4f}, R2: {r2_persist:.4f}")
    print(f"24h RollMean - MAE: {mae_roll24:.4f}, RMSE: {rmse_roll24:.4f}, R2: {r2_roll24:.4f}")
    print(f"XGBoost      - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    native_model_filename = "xgb_model.json"
    native_model_path = os.path.join(MODEL_DIR, native_model_filename)
    final_model.save_model(native_model_path)

    trained_at_utc = datetime.now(timezone.utc).isoformat()
    model_meta = {
        "schema_version": 2,
        "trained_at_utc": trained_at_utc,
        "data_path_used": DATA_PATH,
        "n_csv_files": int(len(csv_files)),
        "n_stations": int(df["station_ID"].nunique()),
        "n_rows_total": int(len(df)),
        "n_rows_loaded_raw": int(n_rows_loaded),
        "n_rows_invalid_datetime_dropped": int(malformed_datetime_rows),
        "n_rows_train": int(len(train_raw)),
        "n_rows_test": int(len(test_raw)),
        "n_rows_train_feat": int(len(train_feat)),
        "model": final_model,
        "model_path": native_model_filename,
        "features": features,
        "best_params": best_params,
        "best_iter": int(final_model.best_iteration if final_model.best_iteration is not None else best_iter),
        "target_type": "delta",
        "clip_q1_q99": (float(clip_stats[0]), float(clip_stats[1])),
        "aux_fill": aux_fill_global,
        "feature_defaults": feature_defaults,
        "feature_quantiles": feature_quantiles,
        "global_shap_mean_abs": global_shap_mean_abs,
        "global_shap_sample_rows": int(min(3000, len(X_test))),
        "training_pm25_mean": training_pm25_mean,
        "bias_correction": float(bias_correction),
        "metrics_test": {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "mbe": float(mbe),
            "directional_acc_pct": float(dir_acc),
        },
        "baseline_metrics": {
            "persistence": {"mae": float(mae_persist), "rmse": float(rmse_persist), "r2": float(r2_persist)},
            "roll24": {"mae": float(mae_roll24), "rmse": float(rmse_roll24), "r2": float(r2_roll24)},
        },
        "plot_data": {
            "index": [str(x) for x in test_feat.index],
            "y_true": y_true.tolist(),
            "preds": preds.tolist(),
            "persist": persist.tolist(),
            "roll24": roll24_filled.tolist(),
        },
        "preprocessing": {
            "dataset_stage": "Airware-Haikou/2_filled_data (interpolated filled stage)",
            "dataset_stage_note": "Runtime/training pipeline is custom-built on top of this stage; final 3_MTSAM split is not used directly.",
            "scaling_applied": False,
            "scaling_note": "Tree-based XGBoost model; no feature scaling was applied.",
            "outlier_policy": {
                "pm25_training_clip": "q01-q99 clip applied only on training_mode feature generation",
                "feature_quantiles": "q01/q05/q25/q50/q75/q95/q99 saved for runtime plausibility bounds",
                "runtime_clamping": "Scenario/custom interventions are quantile-bounded in backend scenario engine",
            },
        },
        "training_lineage": {
            "training_scheme": "xgboost_delta_target_next_hour_pm25",
            "horizon_hours": 1,
            "target_definition": "delta = PM2.5(t) - lag1",
            "data_scope": {
                "dataset_stage": "Airware-Haikou/2_filled_data",
                "n_csv_files": int(len(csv_files)),
                "n_rows_total": int(len(df)),
                "n_stations": int(df["station_ID"].nunique()),
            },
            "train_test_split_scheme": "global_timestamp_cutoff_80pct_unique_timestamps",
            "train_test_split_cutoff": str(split_time),
            "cv_scheme": "grouped_temporal_timeseriessplit_unique_timestamps_n3",
            "cv_unique_timestamps_train": int(len(pd.Index(X_train.index).unique())),
            "final_validation_scheme": "grouped_temporal_holdout_last_10pct_unique_timestamps",
            "final_validation": final_val_info,
            "optuna_trials": 30,
            "optuna_best_trial": int(study.best_trial.number),
            "artifact_generated_utc": trained_at_utc,
        },
    }

    joblib.dump(model_meta, MODEL_PATH)
    print("\nModel and metadata saved to", MODEL_PATH)

    #comment in final version 
    # idx = pd.to_datetime(model_meta["plot_data"]["index"])
    # y = np.array(model_meta["plot_data"]["y_true"], dtype=float)
    # p = np.array(model_meta["plot_data"]["preds"], dtype=float)

    # N = min(300, len(y))
    # fig = plt.figure(figsize=(12, 4))
    # plt.plot(idx[:N], y[:N], label="Actual", alpha=0.75)
    # plt.plot(idx[:N], p[:N], label="Predicted", alpha=0.85)
    # plt.title("Actual vs Predicted (first 300 test points)")
    # plt.xlabel("Time")
    # plt.ylabel("PM2.5")
    # plt.legend()
    # _save_or_show(fig, f"{PLOT_PREFIX}_actual_vs_pred.png")

    return model_meta

def load_model():
    if os.path.exists(MODEL_PATH):
        print("Loading saved model meta from", MODEL_PATH)
        meta = joblib.load(MODEL_PATH)
        # Backward compatibility with older metadata produced by log-target models.
        meta.setdefault("log_target", False)
        meta.setdefault("target_type", "delta")
        meta.setdefault("bias_correction", 0.0)
        meta.setdefault("feature_defaults", {})
        meta.setdefault("feature_quantiles", {})
        meta.setdefault("global_shap_mean_abs", {})
        meta.setdefault("training_pm25_mean", None)
        return meta
    return train_and_save()


if __name__ == "__main__":
    load_model()
