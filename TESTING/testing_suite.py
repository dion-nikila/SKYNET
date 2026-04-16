#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib


ROOT_DIR = Path(__file__).resolve().parents[1]
TESTING_DIR = Path(__file__).resolve().parent


@dataclass
class StepResult:
    name: str
    passed: bool
    duration_s: float
    returncode: int
    command: list[str]
    stdout: str
    stderr: str


def _line(char: str = "=", width: int = 90) -> str:
    return char * width


def _title(text: str) -> None:
    print("\n" + _line("="))
    print(text)
    print(_line("="))


def _subtitle(text: str) -> None:
    print("\n" + _line("-"))
    print(text)
    print(_line("-"))


def _format_cmd(cmd: Iterable[str]) -> str:
    return " ".join(shlex.quote(p) for p in cmd)


def _tail(text: str, max_lines: int = 80) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text.strip()
    omitted = len(lines) - max_lines
    tail_lines = "\n".join(lines[-max_lines:])
    return f"... ({omitted} lines omitted; showing last {max_lines}) ...\n{tail_lines}".strip()


def _run_step(name: str, cmd: list[str], cwd: Path) -> StepResult:
    _subtitle(f"[RUN] {name}")
    print(f"[CMD] {_format_cmd(cmd)}")
    start = time.perf_counter()
    completed = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    duration = time.perf_counter() - start
    print(f"[EXIT] {completed.returncode}")
    print(f"[TIME] {duration:.2f}s")
    if completed.stdout.strip():
        print("[STDOUT]")
        print(_tail(completed.stdout, max_lines=120))
    if completed.stderr.strip():
        print("[STDERR]")
        print(_tail(completed.stderr, max_lines=60))
    return StepResult(
        name=name,
        passed=(completed.returncode == 0),
        duration_s=duration,
        returncode=completed.returncode,
        command=cmd,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _print_model_snapshot(model_meta_path: Path) -> None:
    _subtitle("MODEL SNAPSHOT")
    if not model_meta_path.exists():
        print(f"[WARN] Model metadata not found: {model_meta_path}")
        return

    meta = joblib.load(model_meta_path)
    print(f"model_meta_path: {model_meta_path}")
    print(f"target_type: {meta.get('target_type')}")
    print(f"data_path_used: {meta.get('data_path_used')}")
    print(f"subset_restriction_applied: {meta.get('subset_restriction_applied')}")
    print(f"n_stations: {meta.get('n_stations')}")
    print(f"features_count: {len(meta.get('features', []))}")

    metrics = meta.get("metrics_test", {}) or {}
    baselines = meta.get("baseline_metrics", {}) or {}
    tail = meta.get("tail_diagnostics_test", {}) or {}
    retention = meta.get("row_retention", {}) or {}
    bias = meta.get("bias_correction", {}) or {}

    print("\n[METRICS_TEST]")
    for k in ["mae", "rmse", "r2", "mbe", "directional_acc_pct"]:
        if k in metrics:
            print(f"{k}: {metrics[k]}")

    if "persistence" in baselines:
        print("\n[BASELINE_PERSISTENCE]")
        for k, v in baselines["persistence"].items():
            print(f"{k}: {v}")
    if "roll24" in baselines:
        print("\n[BASELINE_ROLL24]")
        for k, v in baselines["roll24"].items():
            print(f"{k}: {v}")

    if tail:
        print("\n[TAIL_DIAGNOSTICS_TEST]")
        print(json.dumps(tail, indent=2))
    if retention:
        print("\n[ROW_RETENTION]")
        print(json.dumps(retention, indent=2))
    if bias:
        print("\n[BIAS_CORRECTION]")
        print(json.dumps(bias, indent=2))


def _print_benchmark_table(benchmark_csv: Path) -> None:
    _subtitle("BENCHMARK TABLE (5 MODELS)")
    if not benchmark_csv.exists():
        print(f"[WARN] Benchmark CSV not found: {benchmark_csv}")
        return

    with benchmark_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"[WARN] Benchmark CSV is empty: {benchmark_csv}")
        return

    print("Model | MAE | RMSE | R² | MBE | Directional Accuracy (%) | Status")
    print("-" * 95)
    for r in rows:
        print(
            f"{r.get('Model','')} | "
            f"{r.get('MAE','')} | "
            f"{r.get('RMSE','')} | "
            f"{r.get('R²','')} | "
            f"{r.get('MBE','')} | "
            f"{r.get('Directional Accuracy (%)','')} | "
            f"{r.get('Status','')}"
        )


def _print_response_table(response_csv: Path) -> None:
    _subtitle("RESPONSE TIME TABLE")
    if not response_csv.exists():
        print(f"[WARN] Response-time CSV not found: {response_csv}")
        return

    with response_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"[WARN] Response-time CSV is empty: {response_csv}")
        return

    print("Test Case | Runs | Mean (s) | Median (s) | Min (s) | Max (s)")
    print("-" * 90)
    for r in rows:
        print(
            f"{r.get('Test Case','')} | "
            f"{r.get('Runs','')} | "
            f"{r.get('Mean (s)','')} | "
            f"{r.get('Median (s)','')} | "
            f"{r.get('Min (s)','')} | "
            f"{r.get('Max (s)','')}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-run testing suite with screenshot-friendly terminal output.")
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python interpreter used for test commands (default: current interpreter).",
    )
    parser.add_argument(
        "--model-meta",
        default=str(ROOT_DIR / "model" / "xgb_haikou_model_meta.pkl"),
        help="Path to model metadata file.",
    )
    parser.add_argument("--skip-backend", action="store_true", help="Skip backend unittest suite.")
    parser.add_argument("--skip-frontend", action="store_true", help="Skip frontend contract tests.")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip benchmark model comparison.")
    parser.add_argument("--skip-response", action="store_true", help="Skip prototype response-time measurement.")
    parser.add_argument("--rf-n-estimators", type=int, default=80, help="Random Forest trees for benchmark script.")
    parser.add_argument("--rf-max-depth", type=int, default=20, help="Random Forest max depth for benchmark script.")
    parser.add_argument("--response-runs", type=int, default=10, help="Runs per case for response timing.")
    parser.add_argument("--response-warmup-runs", type=int, default=2, help="Warm-up runs per case for response timing.")
    args = parser.parse_args()

    _title("SKYNET TESTING SUITE")
    print(f"repo_root: {ROOT_DIR}")
    print(f"python_bin: {args.python_bin}")
    print(f"timestamp_utc: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}")

    results: list[StepResult] = []
    model_meta_path = Path(args.model_meta).resolve()
    benchmark_csv = TESTING_DIR / "benchmark_model_comparison_results.csv"
    response_csv = TESTING_DIR / "prototype_response_time_results.csv"

    _print_model_snapshot(model_meta_path=model_meta_path)

    if not args.skip_backend:
        cmd = [
            args.python_bin,
            "-m",
            "unittest",
            "discover",
            "-s",
            "backend/tests",
            "-p",
            "test_*.py",
            "-v",
        ]
        results.append(_run_step("Backend tests", cmd=cmd, cwd=ROOT_DIR))
    else:
        print("\n[SKIP] Backend tests")

    if not args.skip_frontend:
        npm_bin = "npm"
        cmd = [npm_bin, "test", "--", "--runInBand"]
        results.append(_run_step("Frontend contract tests", cmd=cmd, cwd=ROOT_DIR / "frontend"))
    else:
        print("\n[SKIP] Frontend tests")

    if not args.skip_benchmark:
        cmd = [
            args.python_bin,
            str(TESTING_DIR / "benchmark_model_comparison.py"),
            "--rf-n-estimators",
            str(args.rf_n_estimators),
            "--rf-max-depth",
            str(args.rf_max_depth),
            "--out-csv",
            str(benchmark_csv),
        ]
        results.append(_run_step("Benchmark model comparison", cmd=cmd, cwd=ROOT_DIR))
        _print_benchmark_table(benchmark_csv=benchmark_csv)
    else:
        print("\n[SKIP] Benchmark comparison")

    if not args.skip_response:
        cmd = [
            args.python_bin,
            str(TESTING_DIR / "measure_prototype_response_times.py"),
            "--runs",
            str(args.response_runs),
            "--warmup-runs",
            str(args.response_warmup_runs),
            "--out-csv",
            str(response_csv),
        ]
        results.append(_run_step("Prototype response-time measurement", cmd=cmd, cwd=ROOT_DIR))
        _print_response_table(response_csv=response_csv)
    else:
        print("\n[SKIP] Response-time measurement")

    _title("FINAL SUMMARY")
    if not results:
        print("No steps were executed.")
        return 0

    print("Step | Status | Exit | Time(s)")
    print("-" * 60)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"{r.name} | {status} | {r.returncode} | {r.duration_s:.2f}")

    all_pass = all(r.passed for r in results)
    print("\nArtifacts:")
    print(f"- benchmark_csv: {benchmark_csv}")
    print(f"- response_csv: {response_csv}")
    print(f"- model_meta: {model_meta_path}")

    if all_pass:
        print("\nOVERALL STATUS: PASS")
        return 0

    print("\nOVERALL STATUS: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
