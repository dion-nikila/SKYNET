# SKYNET

SKYNET is a one-hour-ahead PM2.5 forecasting system.

- Backend: FastAPI (`backend/app`)
- Frontend: React + Vite (`frontend/src`)
- Runtime model artifacts: `model/xgb_haikou_model_meta.pkl`, `model/xgb_model.json`

## Run Locally

From the extracted project folder, open a terminal in the repo root.

### Backend
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

Frontend runs at:
- `http://localhost:5173`

## Minimum Environment Variables

### Frontend
- `VITE_API_BASE_URL=http://localhost:8000/api/v1` (local)

### Backend (deploy)
- `SKYNET_CORS_ORIGINS=https://<your-frontend-domain>` (no trailing slash)
- `SKYNET_SQLITE_PATH=/tmp/skynet_runs.db`
- `MPLCONFIGDIR=/tmp/mplconfig`

## Deployment details

### Render (backend)
- Build: `pip install -r backend/requirements.txt`
- Start: `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`
- Health check path: `/readyz`

### Vercel (frontend)
- Root directory: `frontend`
- Build command: `npm run build`
- Output directory: `dist`

## AI Usage Declaration

AI tools were used in a limited capacity to assist with code structuring and debugging. All core design decisions, implementation and validation of the system were conducted independently by the author.
