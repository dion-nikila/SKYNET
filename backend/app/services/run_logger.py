from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Dict, List, Optional

from .scenario_engine import SCENARIO_ALIASES


class RunLogger:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with closing(sqlite3.connect(self.db_path)) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    location_lat REAL,
                    location_lon REAL,
                    location_name TEXT,
                    scenario_request_json TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    model_version TEXT
                )
                """
            )
            con.commit()

    def save_run(
        self,
        run_id: str,
        created_at: str,
        location: Dict,
        scenario_request: Dict,
        response_json: Dict,
        model_version: str,
    ):
        with closing(sqlite3.connect(self.db_path)) as con:
            con.execute(
                """
                INSERT OR REPLACE INTO runs (
                    run_id, created_at, location_lat, location_lon, location_name,
                    scenario_request_json, response_json, model_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    created_at,
                    float(location.get("lat", 0.0)),
                    float(location.get("lon", 0.0)),
                    location.get("name"),
                    json.dumps(scenario_request),
                    json.dumps(response_json),
                    model_version,
                ),
            )
            con.commit()

    def list_runs(self, limit: int = 50) -> List[Dict]:
        with closing(sqlite3.connect(self.db_path)) as con:
            cur = con.execute(
                """
                SELECT run_id, created_at, scenario_request_json, response_json
                FROM runs
                ORDER BY rowid DESC
                LIMIT ?
                """,
                (int(limit),),
            )
            rows = cur.fetchall()

        def _as_int(v, default=0):
            try:
                return int(v)
            except Exception:
                return int(default)

        def _as_float(v, default=0.0):
            try:
                return float(v)
            except Exception:
                return float(default)

        def _canonicalize_scenario_id(v):
            sid = str(v or "").strip()
            if not sid:
                return "custom"
            return str(SCENARIO_ALIASES.get(sid, sid))

        def _normalize_scenario_mode(v):
            mode = str(v or "").strip().lower()
            if mode in {"macro", "guided_intervention", "manual_custom", "baseline"}:
                return mode
            return None

        out = []
        for run_id, created_at, req_json, resp_json in rows:
            try:
                req = json.loads(req_json) if req_json else {}
            except Exception:
                req = {}
            try:
                resp = json.loads(resp_json) if resp_json else {}
            except Exception:
                resp = {}

            req_scenario = req.get("scenario") or {}
            resp_scenario = resp.get("scenario") or {}

            intensity = _as_int(req_scenario.get("intensity", 0), 0)
            # Prefer canonical scenario ID emitted in response payload.
            scenario_id = _canonicalize_scenario_id(
                resp_scenario.get("scenario_id")
                or req_scenario.get("scenario_id")
                or "custom"
            )
            if scenario_id == "custom" and intensity == 0:
                scenario_id = "baseline"
            scenario_mode = _normalize_scenario_mode(resp_scenario.get("scenario_mode"))
            if scenario_mode is None:
                forecast_mode = str(req.get("forecast_mode", "")).strip().lower()
                if forecast_mode == "custom" or scenario_id == "custom_what_if":
                    scenario_mode = "manual_custom"
                elif scenario_id == "guided_intervention":
                    scenario_mode = "guided_intervention"
                elif scenario_id == "baseline":
                    scenario_mode = "baseline"
                else:
                    scenario_mode = "macro"

            out.append(
                {
                    "run_id": run_id,
                    "created_at": created_at,
                    "scenario_id": str(scenario_id),
                    "scenario_mode": scenario_mode,
                    "intensity": intensity,
                    "pm25_change": _as_float((resp.get("delta") or {}).get("pm25_change", 0.0), 0.0),
                    "ood_flag": bool(((resp.get("health") or {}).get("ood") or {}).get("flag", False)),
                }
            )
        return out

    def get_run(self, run_id: str) -> Optional[Dict]:
        with closing(sqlite3.connect(self.db_path)) as con:
            cur = con.execute(
                "SELECT run_id, created_at, response_json FROM runs WHERE run_id = ?",
                (run_id,),
            )
            row = cur.fetchone()

        if not row:
            return None

        rid, created_at, response_json = row
        payload = json.loads(response_json)
        if isinstance(payload, dict):
            scenario = payload.get("scenario")
            if isinstance(scenario, dict):
                sid = str(scenario.get("scenario_id", "") or "").strip()
                if sid:
                    canonical = str(SCENARIO_ALIASES.get(sid, sid))
                    if canonical != sid:
                        normalized = dict(payload)
                        normalized_scenario = dict(scenario)
                        normalized_scenario["scenario_id"] = canonical
                        normalized["scenario"] = normalized_scenario
                        payload = normalized

        return {"run_id": rid, "created_at": created_at, "response_json": payload}
