# SKYNET Runtime + Deployment Guide

SKYNET is an interactive one-hour-ahead PM2.5 forecasting system with:
- FastAPI backend (`backend/app`)
- React + Vite frontend (`frontend/src`)
- XGBoost model artifacts (`model/`)

## Active runtime source of truth
- `backend/app`
- `frontend/src`
- `model/xgb_final_model.py`
- `model/xgb_haikou_model_meta.pkl`
- `model/xgb_model.json`

## Dataset stage used by SKYNET
- Training/reference profile source: `data/processed/final_dataset/final.csv`
- This project uses the finalized cleaned-raw curated dataset for locked-model training and reference fallback profiles.
- Raw lineage/provenance bundles are optional and are not required for the active runtime or a lean deployment.
- SKYNET does **not** train directly from raw `1_row_data` files at runtime.
- SKYNET does **not** directly consume the pre-split `3_MTSAM` artifacts for runtime or training.

## Canonical scenario set
- `traffic_gridlock`
- `strong_dispersion`
- `heatwave`
- `dust_resuspension`
- `trapped_pollution`
- `industrial_plume`

Legacy aliases are still accepted:
- `heavy_rainstorm` -> `strong_dispersion`
- `stagnation` -> `trapped_pollution`
- `windy_dispersion` -> `dust_resuspension`

## Forecast modes
- `live`: uses Open-Meteo live/history data.
- `custom`: baseline-anchored bounded what-if model-space exploration (never history-free).

Scenario outputs are model-behavior what-if simulations, not causal intervention estimates.

Custom baseline fallback priority:
1. recent cached/live context
2. dataset-derived reference profile (`data/processed/final_dataset/final.csv`)
3. demo default profile

## Demo-safe location policy
- Default location: `haikou_cn` (training-aligned)
- Also available: `colombo_lk`, `berlin_de`, `paris_fr`, `amsterdam_nl`
- Rationale:
  - Haikou is the training domain.
  - Colombo is retained for user-facing relevance.
  - Europe demo cities are included because Open-Meteo air-quality support is typically stronger/coherent there (CAMS Europe domain), while still treated as exploratory for this Haikou-trained model.
- Non-Haikou runs are explicitly exploratory and should not be reported as externally validated geographic generalization.

## Environment variable setup

### Backend (`backend/.env`)
Copy from `backend/.env.example` and set:
- `SKYNET_CORS_ORIGINS` (comma-separated frontend origins)
- `SKYNET_SQLITE_PATH` (recommended `/tmp/skynet_runs.db` on free hosts)
- `MPLCONFIGDIR` (recommended `/tmp/mplconfig` on free hosts)

Additional optional variables:
- `SKYNET_ENABLE_RUN_LOGGING` (`1`/`0`)
- `SKYNET_MODEL_META_PATH`
- `SKYNET_ROOT_DIR`

### Frontend (`frontend/.env`)
Copy from `frontend/.env.example` and set:
- `VITE_API_BASE_URL` (recommended in production)
- `VITE_API_TIMEOUT_MS` (optional, default `20000`)

## Local development run

Dependency purpose:
- `backend/requirements.txt`: backend API runtime dependencies.
- `requirements.txt`: training/evaluation/support-script dependencies used outside the API runtime.

### 1) Backend
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Health check:
```bash
curl http://127.0.0.1:8000/healthz
```

Readiness check:
```bash
curl http://127.0.0.1:8000/readyz
```

### 2) Frontend
```bash
cd frontend
npm install
npm run dev
```

## Production-style run commands

### Backend
```bash
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

### Frontend build
```bash
cd frontend
npm install
npm run build
npm run preview
```

## Deployment Guide

### Backend deployment (Render example)
- `render.yaml` is included in repo root.
- Root directory: repository root (do not set to `backend/` when using this `render.yaml`).
- Build command: `pip install -r backend/requirements.txt`
- Start command: `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`
- Health check path: `/readyz`
- Required env:
  - `SKYNET_CORS_ORIGINS=https://<your-frontend-domain>`
  - `SKYNET_SQLITE_PATH=/tmp/skynet_runs.db`
  - `MPLCONFIGDIR=/tmp/mplconfig`
- Recommended explicit path envs on Render:
  - `SKYNET_ROOT_DIR=/opt/render/project/src`
  - `SKYNET_MODEL_META_PATH=/opt/render/project/src/model/xgb_haikou_model_meta.pkl`

### Frontend deployment (Vercel example)
- Set project root to `frontend/`
- Build command: `npm run build`
- Output directory: `dist`
- Required env:
  - `VITE_API_BASE_URL=https://<your-backend-domain>/api/v1`
- Note: when frontend and backend are hosted on different domains (Vercel + Render), `VITE_API_BASE_URL` must be set explicitly.
- `frontend/vercel.json` includes SPA rewrite to `index.html`

## Runtime dependency notes
- Required model artifacts:
  - `model/xgb_haikou_model_meta.pkl`
  - `model/xgb_model.json`
- Optional reference dataset for custom fallback profile:
  - `data/processed/final_dataset/final.csv`
- If dataset is missing, custom mode still works via live context or demo defaults.
- SQLite logging on free hosts is ephemeral unless external persistence is added.
- Run logging behavior:
  - If `SKYNET_ENABLE_RUN_LOGGING=0`, backend startup no longer depends on SQLite path writability.
- Current shipped metadata artifact (`model/xgb_haikou_model_meta.pkl`) is used as runtime source-of-truth for:
  - feature list/order, feature defaults, feature quantiles, global SHAP means, bias correction, scored-subset historical test metrics, and native model path.
- Historical test metrics in the shipped artifact are reported on the scored subset of chronological test rows after feature-availability filtering, not on every raw row in the test window.
- Empirical uncertainty guidance can be derived from saved test residual traces in metadata (`plot_data.y_true/preds`) when available.
- Provenance extras from newer training comments (for example `preprocessing` block or raw-row counters) may be absent in the current artifact unless it is regenerated.
- Training code uses grouped temporal logic by unique timestamp for both hyperparameter CV and final early-stopping holdout validation (to avoid same-timestamp multi-station leakage across folds/splits). If the model artifact is not regenerated after methodology updates, deployed metrics/artifacts still reflect the previous training run.

## Testing and validation

Backend tests:
```bash
python -m unittest discover -s backend/tests -p 'test_*.py' -v
```

Frontend lightweight contract tests:
```bash
cd frontend
npm test
```

Scenario validation harness:
```bash
python backend/scripts/validate_scenarios.py --samples 120 --seed 42 --out-dir backend/scripts/validation_artifacts
```

Frontend build check:
```bash
cd frontend
npm run build
```

## Reproducibility Trace (Quick Commands)

Retrain and regenerate runtime artifacts:
```bash
source .venv/bin/activate
python model/xgb_final_model.py
```

Check artifact lineage fields:
```bash
python - <<'PY'
import joblib, json
m = joblib.load("model/xgb_haikou_model_meta.pkl")
print(json.dumps(m.get("training_lineage", {}), indent=2))
PY
```

Run scenario validation harness:
```bash
python backend/scripts/validate_scenarios.py --samples 120 --seed 42 --out-dir backend/scripts/validation_artifacts
```

## Repository hygiene for hosting
- `frontend/dist` is a generated build output and should be produced during CI/deploy (`npm run build`), not committed as source-of-truth.
- `backend/skynet_runs.db` is local runtime state and should not be committed.
- Scenario validation/inspection outputs under `backend/scripts/validation_artifacts*` and `backend/scripts/artifacts*` are generated artifacts; keep only the outputs you intentionally want as evidence snapshots.
- Model backup files like `model/*.bak_*` are local safeguards and should not be committed as active runtime artifacts.
- Non-runtime materials such as local archive bundles or report extracts are optional and are not part of the active runtime path.

## Preprocessing decisions (for report/viva)

- **Scaling:** No `StandardScaler`/`MinMaxScaler` is applied in training or runtime. This is intentional for the current tree-based XGBoost model, which is typically scale-insensitive for split logic.
- **Outlier handling (training):** PM2.5 is clipped to training `q01-q99` during training-mode feature generation to reduce extreme target influence.
- **Outlier handling (metadata):** Feature quantiles (`q01/q05/q25/q50/q75/q95/q99`) are saved in model metadata for runtime plausibility bounds.
- **Outlier handling (runtime):**
  - Scenario/custom interventions are quantile-bounded in the scenario engine.
  - Health diagnostics now flag current exogenous values that sit outside training `q01-q99` bounds.
- **Operational history window:** Runtime default is `72h` (`history_hours_target=72`).
- **Weekly-feature imputation disclosure:** The model includes weekly lag-derived features (for example `lag168`, `trend_168`). When available history is `<168h`, those features are explicitly imputed from trained defaults; imputation is tracked in health diagnostics as heuristic reliability context.
- **Dataset-stage caution:** SKYNET runtime/training uses `data/processed/final_dataset/final.csv` as the active final dataset.
- **Unit mapping rationale:** For `CO`, `pressure`, and `wind_speed`, runtime mapping is validated against active model metadata quantiles from the final locked artifact.
- **Scenario interpretation caution:** Scenario templates encode directional intent only; realized effects are context-dependent and are not guaranteed monotonic for every sample.
- **Causality caution:** Local explanations and scenario outputs describe model behavior in feature space; they do not establish real-world causal mechanisms or intervention effects.
- **Reliability interpretation caution:** Health quality/reliability is a heuristic diagnostic signal, not a calibrated probability of correctness.
- **Uncertainty interpretation caution:** Runtime uncertainty bands are empirical residual ranges from historical Haikou test behavior; they are decision-support ranges, not probabilistic guarantees.
- **Live unit normalization:** Open-Meteo live/history values are converted before feature build:
  - `carbon_monoxide` `ug/m^3` -> model `CO` mg/m^3-equivalent
  - `surface_pressure` `hPa` -> model `pressure` kPa-equivalent
  - `wind_speed_10m` `km/h` -> model `wind_speed` m/s-equivalent
- **Custom pressure input contract:** UI/API keeps pressure input in human-readable `hPa` (850-1100), then backend converts to model pressure representation internally.
- **Malformed row handling:** malformed timestamp rows are ignored by datetime validation (`errors='coerce'` + drop invalid rows), so training and summary scripts remain robust.

## Reliability + uncertainty methodology note
- Reliability guidance is a weighted heuristic composite built from: data completeness, domain plausibility, imputation burden, fallback severity, scenario validity, and explainability integrity.
- Reliability outputs are intended for run-quality interpretation, not statistical confidence.
- Uncertainty guidance uses empirical residual quantiles from historical Haikou test traces (when metadata supports it), then widens scenario bands based on reliability and scenario mode.
- Keep report/viva wording aligned with the runtime reliability and uncertainty notes above.

## Usability evidence scaffold
- Keep usability evidence files/versioning in sync with your current submission package.

## Export
- `Export CSV`: one-row structured export for reporting.
- `Export PDF (Print)`: opens print layout; use browser **Save as PDF**.

## Common deployment issues
1. **Frontend cannot reach backend**
- Check `VITE_API_BASE_URL` and backend CORS (`SKYNET_CORS_ORIGINS`).

2. **Run logs missing on hosted backend**
- Ensure `SKYNET_ENABLE_RUN_LOGGING=1` and writable `SKYNET_SQLITE_PATH` (use `/tmp/...` on free hosts).

3. **Explainability warnings related to matplotlib cache**
- Set `MPLCONFIGDIR` to a writable path (for example `/tmp/mplconfig`).

4. **Live forecast fails in hosted demo**
- Open-Meteo may be temporarily unavailable. Switch to `Custom What-If` mode (baseline-anchored fallback remains available).

5. **Scenario intent differs from observed direction**
- This is expected in context-dependent models. Scenario templates indicate typical intent; final direction is always model-computed.
