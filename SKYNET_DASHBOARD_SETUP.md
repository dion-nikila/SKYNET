# SKYNET Local Run Guide (No Docker)

This project now runs as a normal local full-stack app:
- Backend: FastAPI (`backend/app`)
- Frontend: React JS + Vite (`frontend`)

No Docker is required.

## 0) Final file structure (important)

### New final system files
- Backend API: `backend/app/...`
- Frontend app: `frontend/...`
- Model training pipeline: `model/xgb_final_model.py`
- Model artifact path: `model/xgb_haikou_model_meta.pkl` (metadata) and `model/xgb_model.json` (native XGBoost model)
- Training dataset stage used by this project: `data/processed/final_dataset/final.csv`
- Raw lineage/provenance bundles are optional and are not part of the active runtime path for the lean working copy.
- This project does not directly train from raw `1_row_data` at runtime and does not directly consume `3_MTSAM`.
- Current shipped metadata artifact contains runtime essentials (features/defaults/quantiles/metrics/model path); some provenance extras from newer training comments may be absent unless the artifact is regenerated.
- Current training code uses grouped temporal validation by unique timestamp for both hyperparameter CV and final early-stopping holdout. If you keep an older artifact, it remains usable for demo/runtime but reflects the previous training run.

## 1) Prerequisites
- Python 3.10+ (3.11 recommended)
- Node.js 18+ (20 recommended)
- npm

Check versions:
```bash
python --version
node --version
npm --version
```

## 2) Create and activate Python environment
From project root:
```bash
cd /Users/dionnikila/Desktop/SKYNET-IPD-01
python -m venv .venv
source .venv/bin/activate
```

## 3) Install backend dependencies
```bash
pip install --upgrade pip
pip install -r backend/requirements.txt
```

Dependency purpose:
- `backend/requirements.txt`: backend API runtime dependencies.
- `requirements.txt`: training/evaluation/support-script dependencies outside the API runtime.

## 4) Ensure model artifact exists
Backend needs this file:
- `model/xgb_haikou_model_meta.pkl`

If it does not exist, train it:
```bash
python /Users/dionnikila/Desktop/SKYNET-IPD-01/model/xgb_final_model.py
```

If it exists already, you can skip retraining.

## 5) Start backend API
From project root (same terminal, with venv active):
```bash
cd /Users/dionnikila/Desktop/SKYNET-IPD-01
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend URLs:
- API base: `http://localhost:8000/api/v1`
- Health: `http://localhost:8000/healthz`
- Readiness: `http://localhost:8000/readyz`
- Docs: `http://localhost:8000/docs`

## 6) Start frontend (second terminal)
Open a new terminal:
```bash
cd /Users/dionnikila/Desktop/SKYNET-IPD-01/frontend
cp .env.example .env
npm install
npm run dev
```

Frontend URL:
- `http://localhost:5173`

## 7) End-to-end test flow
1. Open `http://localhost:5173`
2. Confirm baseline run appears.
3. Apply a macro scenario card.
4. Confirm these sections update:
- Trust panel
- Baseline vs Scenario metrics
- Applied overrides
- SHAP and ΔSHAP panels
- Run history

## 8) Common issues

### `ModuleNotFoundError: fastapi`
You are not in venv or dependencies were not installed.

Fix:
```bash
source /Users/dionnikila/Desktop/SKYNET-IPD-01/.venv/bin/activate
pip install -r /Users/dionnikila/Desktop/SKYNET-IPD-01/backend/requirements.txt
```

### Backend starts but forecast fails
Usually missing/corrupt model artifact.

Fix:
```bash
python /Users/dionnikila/Desktop/SKYNET-IPD-01/model/xgb_final_model.py
```

### Frontend cannot call backend (CORS/network)
Ensure backend is running on `:8000` and frontend `.env` has:
```env
VITE_API_BASE_URL=http://localhost:8000/api/v1
```

## 9) What to run day-to-day
Normal usage (no retrain):
1. Start backend
2. Start frontend
3. Open dashboard

Retrain only when model/data logic changes.

Artifact lineage quick check:
```bash
python - <<'PY'
import joblib, json
m = joblib.load('/Users/dionnikila/Desktop/SKYNET-IPD-01/model/xgb_haikou_model_meta.pkl')
print(json.dumps(m.get('training_lineage', {}), indent=2))
PY
```

## 10) Notes
This project is now maintained as the dashboard/API system only (no legacy CLI layer).

Submission-safe interpretation notes:
- CO/pressure/wind mapping in SKYNET is validated against the active final model metadata quantiles.
- The project uses `data/processed/final_dataset/final.csv` as the active final dataset.
- Scenario cards encode directional intent; final effects are context-dependent and are not guaranteed monotonic per sample.
- Scenario outputs are bounded what-if model-space exploration, not causal intervention estimates.
- Local explanations describe model feature contributions, not real-world causal mechanisms.
- Reliability guidance output is heuristic run-quality guidance, not a calibrated probability of correctness.
- Runtime default history window is `72h`; weekly lag-derived features (for example `lag168`, `trend_168`) may be imputed when `<168h` history is available.
- This weekly-feature imputation is explicit in runtime feature building and should be interpreted as part of heuristic reliability diagnostics.
- Uncertainty bands (when available) are empirical residual ranges from historical Haikou test behavior and should be treated as decision-support context, not probabilistic guarantees.
- Demo location policy is intentionally narrow: Haikou (training-aligned) + Colombo (user-relevant) + selected Europe cities (Berlin/Paris/Amsterdam) for more coherent Open-Meteo air-quality coverage; non-Haikou use remains exploratory.
- Keep usability evidence files/versioning aligned with your current submission package.
- Repository hygiene:
  - `frontend/dist` is generated (`npm run build`) and should not be treated as source-of-truth.
  - `backend/skynet_runs.db` is local runtime state and should not be committed.
  - `backend/scripts/validation_artifacts*` and `backend/scripts/artifacts*` are generated outputs; keep only intentionally curated evidence snapshots.
