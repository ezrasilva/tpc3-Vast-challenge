"""Funções de carregamento para os dados do VAST Challenge 2026 MC2."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_ZIP_CANDIDATES = (
    Path("data/VAST_Challenge_2026_MC2.zip"),
    Path("data/VAST_Challenge_2026_MC2 (2).zip"),
    Path("VAST_Challenge_2026_MC2.zip"),
)
DEFAULT_JSON_CANDIDATES = (
    Path("data/MC2 data.json"),
    Path("VAST_Challenge_2026_MC2/MC2 data.json"),
)


def _events_to_dataframe(payload: dict[str, Any]) -> pd.DataFrame:
    if not isinstance(payload, dict):
        raise ValueError("O JSON deve conter um objeto na raiz.")
    events = payload.get("events")
    if not isinstance(events, list):
        raise ValueError("Não foi encontrada uma lista válida no campo 'events'.")
    df = pd.DataFrame(events)
    required = {"id", "when", "short_name", "parties", "details"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError("Colunas obrigatórias ausentes: " + ", ".join(sorted(missing)))
    return df


def load_json_path(path: str | Path) -> tuple[pd.DataFrame, dict | None]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return _events_to_dataframe(payload), None


def load_json_bytes(content: bytes) -> tuple[pd.DataFrame, dict | None]:
    payload = json.loads(content.decode("utf-8"))
    return _events_to_dataframe(payload), None


def load_zip_bytes(content: bytes) -> tuple[pd.DataFrame, dict | None]:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        json_candidates = [name for name in archive.namelist() if Path(name).name == "MC2 data.json"]
        if not json_candidates:
            raise FileNotFoundError("O ZIP não contém 'MC2 data.json'.")
        payload = json.loads(archive.read(json_candidates[0]).decode("utf-8"))
        org_candidates = [name for name in archive.namelist() if Path(name).name == "org_chart.json"]
        org = json.loads(archive.read(org_candidates[0]).decode("utf-8")) if org_candidates else None
    return _events_to_dataframe(payload), org


def load_zip_path(path: str | Path) -> tuple[pd.DataFrame, dict | None]:
    return load_zip_bytes(Path(path).read_bytes())


def find_local_dataset() -> Path | None:
    for candidate in DEFAULT_ZIP_CANDIDATES:
        if candidate.exists():
            return candidate
    for candidate in DEFAULT_JSON_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def load_local_dataset(path: str | Path) -> tuple[pd.DataFrame, dict | None]:
    path = Path(path)
    if path.suffix.lower() == ".zip":
        return load_zip_path(path)
    return load_json_path(path)
