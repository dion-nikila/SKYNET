from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.routes import router, runtime_status_payload
from .core.settings import settings


app = FastAPI(title=settings.API_TITLE, version=settings.API_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/healthz")
def healthz():
    return {"ok": True, "type": "liveness"}


@app.get("/readyz")
def readyz():
    status = runtime_status_payload()
    code = 200 if bool(status.get("ready")) else 503
    return JSONResponse(status_code=code, content=status)
