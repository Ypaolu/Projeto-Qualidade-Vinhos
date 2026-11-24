"""Configurações e caminhos comuns ao projeto Wine Quality N3."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "params.json"
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
MLRUNS = ROOT / "mlruns"
EVIDENCIAS = ROOT / "evidencias"


def ensure_dirs() -> None:
    for folder in (RAW, PROCESSED, REPORTS, MODELS, MLRUNS, EVIDENCIAS):
        folder.mkdir(parents=True, exist_ok=True)


def get_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
