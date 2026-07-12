"""Pré-processamento e enriquecimento dos eventos."""

from __future__ import annotations

import pandas as pd


def detail(details: object, key: str) -> object:
    return details.get(key) if isinstance(details, dict) else None


def nested_arg(details: object, key: str) -> object:
    if isinstance(details, dict) and isinstance(details.get("args"), dict):
        return details["args"].get(key)
    return None


def first_party(parties: object) -> str | None:
    if isinstance(parties, list) and len(parties) > 0:
        return parties[0]
    return None


def second_party(parties: object) -> str | None:
    if isinstance(parties, list) and len(parties) > 1:
        return parties[1]
    return None


def entity_type(entity: object) -> str:
    if not isinstance(entity, str):
        return "desconhecido"
    if entity.startswith("Agent/person:"):
        return "agente"
    if entity.startswith("person:"):
        return "pessoa"
    if entity.startswith("system:"):
        return "sistema"
    if entity.startswith("department:"):
        return "departamento"
    if entity.startswith("team:"):
        return "time"
    if entity.startswith("company:"):
        return "empresa"
    return "outro"


def clean_entity_name(entity: object) -> str:
    if not isinstance(entity, str):
        return ""
    for prefix in ["Agent/person:", "person:", "system:", "department:", "team:", "company:"]:
        if entity.startswith(prefix):
            entity = entity.removeprefix(prefix)
            break
    return entity.replace("_", " ").title()


def preprocess_events(df: pd.DataFrame, utc_offset_hours: int = -7) -> pd.DataFrame:
    result = df.copy()
    result["datetime_utc"] = pd.to_datetime(result["when"], unit="s", utc=True, errors="coerce")
    result["datetime_local_desafio"] = result["datetime_utc"] + pd.Timedelta(hours=utc_offset_hours)

    for key in ["forum", "content", "content_source", "poster_id", "target_agent", "target", "action", "task", "size_hint", "from", "to", "subject", "status"]:
        result[key] = result["details"].apply(lambda item, k=key: detail(item, k))

    result["args_path"] = result["details"].apply(lambda item: nested_arg(item, "path"))
    result["file_target"] = result["target"].where(result["target"].notna(), result["args_path"])
    result["path"] = result["file_target"]

    result["source_entity"] = result["parties"].apply(first_party)
    result["target_entity"] = result["parties"].apply(second_party)
    result["target_agent_clean"] = result["target_agent"].where(result["target_agent"].notna(), result["target_entity"])
    result["source_label"] = result["source_entity"].apply(clean_entity_name)
    result["target_label"] = result["target_agent_clean"].apply(clean_entity_name)
    result["source_type"] = result["source_entity"].apply(entity_type)
    result["target_type"] = result["target_agent_clean"].apply(entity_type)

    result["details_str"] = result["details"].astype(str)
    result["parties_str"] = result["parties"].astype(str)
    result["search_text"] = result["details_str"] + " " + result["parties_str"]

    return result.sort_values("datetime_utc").reset_index(drop=True)
