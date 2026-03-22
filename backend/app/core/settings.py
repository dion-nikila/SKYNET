import os
from pathlib import Path


class Settings:
    ROOT_DIR = Path(os.getenv("SKYNET_ROOT_DIR", str(Path(__file__).resolve().parents[3]))).resolve()
    MODEL_META_PATH = Path(
        os.getenv(
            "SKYNET_MODEL_META_PATH",
            str(ROOT_DIR / "model" / "xgb_haikou_model_meta.pkl"),
        )
    ).expanduser()
    SQLITE_PATH = Path(
        os.getenv(
            "SKYNET_SQLITE_PATH",
            str(ROOT_DIR / "backend" / "skynet_runs.db"),
        )
    ).expanduser()
    MPLCONFIGDIR = Path(
        os.getenv(
            "MPLCONFIGDIR",
            str(ROOT_DIR / ".mplconfig"),
        )
    ).expanduser()

    API_TITLE = "SKYNET Interactive PM2.5 Forecast API"
    API_VERSION = "1.0.0"

    DEFAULT_HISTORY_HOURS = 72
    DEFAULT_TOP_K_DRIVERS = 6
    DEFAULT_SOFT_Q = 0.05
    DEFAULT_HARD_Q = 0.01

    ENABLE_RUN_LOGGING = os.getenv("SKYNET_ENABLE_RUN_LOGGING", "1") == "1"
    _RAW_CORS = os.getenv(
        "SKYNET_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).strip()
    if _RAW_CORS == "*":
        CORS_ORIGINS = ["*"]
        CORS_ALLOW_CREDENTIALS = False
    else:
        CORS_ORIGINS = [
            x.strip()
            for x in _RAW_CORS.split(",")
            if x.strip()
        ]
        CORS_ALLOW_CREDENTIALS = True


settings = Settings()

# Ensure matplotlib/shap cache points to a writable location by default.
os.environ.setdefault("MPLCONFIGDIR", str(settings.MPLCONFIGDIR))
try:
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
except Exception:
    fallback = Path("/tmp/skynet_mplconfig")
    fallback.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(fallback)
