"""Cálculos analíticos utilizados no notebook e no dashboard."""

from __future__ import annotations

import pandas as pd

from .preprocessing import clean_entity_name, entity_type

KEY_EVENTS = ["queue_subordinate_task", "saidit_post", "delete_file", "create_file", "read_file"]


def file_based_posts(df: pd.DataFrame) -> pd.DataFrame:
    posts = df[df["short_name"].eq("saidit_post") & df["content_source"].notna()].copy()
    posts["case_name"] = posts["content_source"].astype(str).str.replace(".txt", "", regex=False)
    return posts.sort_values("datetime_utc").reset_index(drop=True)


def case_events(df: pd.DataFrame, case_name: str) -> pd.DataFrame:
    return df[df["search_text"].str.contains(case_name, case=False, na=False)].copy().sort_values("datetime_utc").reset_index(drop=True)


def post_for_case(posts: pd.DataFrame, case_name: str) -> pd.Series:
    matches = posts[posts["case_name"].eq(case_name)]
    if matches.empty:
        raise KeyError(f"Caso não encontrado: {case_name}")
    return matches.iloc[0]


def relative_to_post(events: pd.DataFrame, post_time: pd.Timestamp) -> pd.DataFrame:
    result = events.copy()
    result["segundos_relativos_post"] = (result["datetime_utc"] - post_time).dt.total_seconds()
    return result


def delegation_edges(events: pd.DataFrame) -> pd.DataFrame:
    chain = events[events["short_name"].eq("queue_subordinate_task")].copy()
    chain = chain[chain["source_entity"].notna() & chain["target_agent_clean"].notna()]
    return chain[["id", "datetime_utc", "datetime_local_desafio", "source_entity", "source_label", "target_agent_clean", "target_label", "path"]].sort_values("datetime_utc").reset_index(drop=True)


def final_sequence(events: pd.DataFrame, n_delegations: int = 12) -> pd.DataFrame:
    key = events[events["short_name"].isin(["queue_subordinate_task", "saidit_post", "delete_file", "create_file", "read_file"])].copy()
    post_idx = key.index[key["short_name"].eq("saidit_post")]
    if len(post_idx) == 0:
        return key
    post_pos = key.index.get_loc(post_idx[0])
    before = key.iloc[max(0, post_pos - n_delegations):post_pos]
    after = key.iloc[post_pos:post_pos + 5]
    seq = pd.concat([before, after]).copy().reset_index(drop=True)
    seq["ordem"] = range(1, len(seq) + 1)
    seq["descricao"] = seq.apply(describe_event, axis=1)
    seq["delta_s"] = seq["datetime_utc"].diff().dt.total_seconds().fillna(0)
    return seq


def describe_event(row: pd.Series) -> str:
    event = row.get("short_name")
    if event == "queue_subordinate_task":
        return f"{row.get('source_label')} → {row.get('target_label')}"
    if event == "saidit_post":
        return f"{row.get('source_label')} publica no SaidIt"
    if event == "delete_file":
        return f"{row.get('source_label')} apaga {row.get('path')}"
    if event == "create_file":
        return f"{row.get('source_label')} cria {row.get('path')}"
    if event == "read_file":
        return f"{row.get('source_label')} lê {row.get('path')}"
    return str(event)


def summarize_case(df: pd.DataFrame, posts: pd.DataFrame, case_name: str) -> dict:
    events = case_events(df, case_name)
    post = post_for_case(posts, case_name)
    edges = delegation_edges(events)
    duration = events["datetime_utc"].max() - events["datetime_utc"].min()
    deletes_after = events[events["short_name"].eq("delete_file") & events["datetime_utc"].ge(post["datetime_utc"])]
    seconds_to_delete = None
    if not deletes_after.empty:
        seconds_to_delete = float((deletes_after["datetime_utc"].min() - post["datetime_utc"]).total_seconds())
    agents = set()
    for parties in events["parties"]:
        if isinstance(parties, list):
            agents.update([p for p in parties if isinstance(p, str) and p.startswith("Agent/person:")])
    creator = None
    created = events[events["short_name"].eq("create_file") & events["path"].astype(str).str.contains(case_name, case=False, na=False)]
    if not created.empty:
        creator = created.iloc[0]["source_entity"]
    last_sender = edges.iloc[-1]["source_entity"] if not edges.empty else None
    return {
        "caso": case_name,
        "data_post": post["datetime_local_desafio"],
        "forum": post.get("forum"),
        "arquivo_fonte": post.get("content_source"),
        "criador_arquivo": creator,
        "ultimo_repassador": last_sender,
        "duracao_horas": round(duration.total_seconds() / 3600, 2),
        "qtd_eventos": int(len(events)),
        "qtd_delegacoes": int(len(edges)),
        "agentes_unicos": int(len(agents)),
        "qtd_delete_file": int(events["short_name"].eq("delete_file").sum()),
        "segundos_ate_exclusao": seconds_to_delete,
    }


def comparison_table(df: pd.DataFrame, posts: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([summarize_case(df, posts, c) for c in posts["case_name"].tolist()]).sort_values("data_post").reset_index(drop=True)


def similarity_table(df: pd.DataFrame, posts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in posts["case_name"]:
        ev = case_events(df, c)
        post = post_for_case(posts, c)
        rows.append({
            "Característica": "Usa content_source",
            c: "sim" if pd.notna(post.get("content_source")) else "não",
        })
    # easier construct wide table explicitly
    characteristics = []
    for label, func in [
        ("Usa content_source", lambda ev, post: "sim" if pd.notna(post.get("content_source")) else "não"),
        ("Possui *_further_instructions.md", lambda ev, post: "sim" if ev["path"].astype(str).str.contains("further_instructions", na=False).any() else "não"),
        ("Publicador", lambda ev, post: clean_entity_name(post["source_entity"])),
        ("Fórum", lambda ev, post: str(post.get("forum"))),
        ("Exclusões após o post", lambda ev, post: str(int((ev["short_name"].eq("delete_file") & ev["datetime_utc"].ge(post["datetime_utc"])).sum()))),
        ("Arquivo-fonte", lambda ev, post: str(post.get("content_source"))),
    ]:
        row = {"Característica": label}
        for c in posts["case_name"]:
            ev = case_events(df, c)
            post = post_for_case(posts, c)
            row[c] = func(ev, post)
        characteristics.append(row)
    return pd.DataFrame(characteristics)


def provenance_table(events: pd.DataFrame, case_name: str) -> pd.DataFrame:
    relevant = events[
        events["short_name"].isin(["create_file", "read_file", "queue_subordinate_task", "saidit_post", "delete_file"])
    ].copy()
    relevant = relevant[
        relevant["search_text"].str.contains(case_name, case=False, na=False)
    ].sort_values("datetime_utc").copy()
    relevant["nivel"] = relevant["short_name"].map({
        "create_file": "origem operacional",
        "read_file": "instruções",
        "queue_subordinate_task": "propagação",
        "saidit_post": "publicação",
        "delete_file": "remoção",
    })
    relevant["evidencia"] = relevant.apply(describe_event, axis=1)
    return relevant[["id", "datetime_local_desafio", "nivel", "short_name", "source_label", "target_label", "path", "content_source", "evidencia"]]


def post_baseline(df: pd.DataFrame, window_seconds: int = 5) -> pd.DataFrame:
    posts = df[df["short_name"].eq("saidit_post")].copy().sort_values("datetime_utc")
    rows = []
    for _, post in posts.iterrows():
        agent = post.get("source_entity")
        t = post["datetime_utc"]
        after = df[(df["datetime_utc"] > t) & (df["datetime_utc"] <= t + pd.Timedelta(seconds=window_seconds))]
        same_agent_deletes = []
        for _, row in after[after["short_name"].eq("delete_file")].iterrows():
            if isinstance(row.get("parties"), list) and agent in row["parties"]:
                same_agent_deletes.append(row["id"])
        rows.append({
            "id": post["id"],
            "datetime_local_desafio": post["datetime_local_desafio"],
            "publicador": clean_entity_name(agent),
            "forum": post.get("forum"),
            "usa_content_source": pd.notna(post.get("content_source")),
            "content_source": post.get("content_source"),
            "delete_file_ate_5s": len(same_agent_deletes),
            "sinalizado_regra": pd.notna(post.get("content_source")) and len(same_agent_deletes) >= 2,
        })
    return pd.DataFrame(rows)


def anomaly_funnel(df: pd.DataFrame, posts: pd.DataFrame, window_seconds: int = 5) -> pd.DataFrame:
    baseline = post_baseline(df, window_seconds=window_seconds)
    instruction_cases = 0
    for case_name in posts["case_name"]:
        events = case_events(df, case_name)
        if events["path"].astype(str).str.contains("further_instructions", case=False, na=False).any():
            instruction_cases += 1
    return pd.DataFrame([
        {"criterio": "Todas as publicações SaidIt", "quantidade": int(len(baseline))},
        {"criterio": "Com content_source", "quantidade": int(baseline["usa_content_source"].sum())},
        {"criterio": "Com arquivo de instruções", "quantidade": int(instruction_cases)},
        {"criterio": "Com 2 exclusões até 5 s", "quantidade": int((baseline["delete_file_ate_5s"] >= 2).sum())},
        {"criterio": "Sinalizadas pela regra", "quantidade": int(baseline["sinalizado_regra"].sum())},
    ])


def cumulative_events(df: pd.DataFrame, posts: pd.DataFrame) -> pd.DataFrame:
    out = []
    for case_name in posts["case_name"]:
        ev = case_events(df, case_name)
        post = post_for_case(posts, case_name)
        ev = relative_to_post(ev, post["datetime_utc"]).sort_values("datetime_utc")
        ev["eventos_acumulados"] = range(1, len(ev) + 1)
        ev["caso"] = case_name
        out.append(ev)
    return pd.concat(out, ignore_index=True)


def agent_event_matrix(events: pd.DataFrame) -> pd.DataFrame:
    valid = events[events["source_label"].ne("")].copy()
    return pd.crosstab(valid["source_label"], valid["short_name"])


def interaction_edges(df: pd.DataFrame, max_edges: int = 180) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = []
    for _, row in df.iterrows():
        parties = row.get("parties")
        if isinstance(parties, list) and len(parties) >= 2:
            src, tgt = parties[0], parties[1]
            pairs.append({"source": src, "target": tgt, "event": row.get("short_name")})
    edges = pd.DataFrame(pairs)
    if edges.empty:
        return pd.DataFrame(), pd.DataFrame()
    edges = edges.groupby(["source", "target"], as_index=False).size().rename(columns={"size": "weight"})
    edges = edges.sort_values("weight", ascending=False).head(max_edges)
    nodes = pd.unique(edges[["source", "target"]].values.ravel("K"))
    nodes_df = pd.DataFrame({"id": nodes})
    nodes_df["label"] = nodes_df["id"].apply(clean_entity_name)
    nodes_df["type"] = nodes_df["id"].apply(entity_type)
    nodes_df["degree_weight"] = nodes_df["id"].map(
        pd.concat([
            edges.groupby("source")["weight"].sum(),
            edges.groupby("target")["weight"].sum(),
        ], axis=1).fillna(0).sum(axis=1)
    ).fillna(1)
    return nodes_df, edges


def intervention_coverage(df: pd.DataFrame, posts: pd.DataFrame) -> pd.DataFrame:
    baseline = post_baseline(df)
    rows = []
    for _, row in baseline.iterrows():
        rows.append({
            "grupo": "Com content_source + 2 exclusões até 5s" if row["sinalizado_regra"] else "Não sinalizado",
            "quantidade": 1,
        })
    grouped = pd.DataFrame(rows).groupby("grupo", as_index=False)["quantidade"].sum()
    return grouped
