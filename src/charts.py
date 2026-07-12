"""Visualizações Plotly para o dashboard Streamlit."""

from __future__ import annotations

import math
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

from .analysis import case_events, delegation_edges, interaction_edges

EVENT_COLORS = {
    "queue_subordinate_task": "#2563EB",
    "saidit_post": "#F97316",
    "delete_file": "#DC2626",
    "create_file": "#16A34A",
    "read_file": "#7C3AED",
    "post_saidit": "#94A3B8",
}
TYPE_COLORS = {
    "agente": "#2563EB",
    "pessoa": "#16A34A",
    "sistema": "#F97316",
    "outro": "#64748B",
    "departamento": "#8B5CF6",
    "time": "#0EA5E9",
}


def style_figure(fig: go.Figure, height: int = 480) -> go.Figure:
    fig.update_layout(
        height=height,
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=35),
        font=dict(size=13),
        legend_title_text="",
        hoverlabel=dict(font_size=13),
    )
    return fig


def event_distribution(df: pd.DataFrame, limit: int = 18) -> go.Figure:
    counts = df["short_name"].value_counts().head(limit).rename_axis("tipo").reset_index(name="quantidade").sort_values("quantidade")
    fig = px.bar(counts, x="quantidade", y="tipo", orientation="h", text="quantidade", title="Distribuição dos tipos de eventos")
    fig.update_traces(marker_color="#2563EB")
    return style_figure(fig, 520)


def system_network(df: pd.DataFrame, highlight_entities: list[str] | None = None) -> go.Figure:
    """Rede completa, adequada para exploração interativa."""
    highlight_entities = highlight_entities or []
    nodes, edges = interaction_edges(df, max_edges=160)
    return _network_figure(nodes, edges, highlight_entities, "Visão completa das interações registradas", 690, labels_all=True)


def system_network_summary(df: pd.DataFrame, case_names: list[str]) -> go.Figure:
    """Rede resumida aos eventos dos três casos, adequada para apresentação."""
    pattern = "|".join(case_names)
    subset = df[df["search_text"].str.contains(pattern, case=False, na=False)].copy()
    nodes, edges = interaction_edges(subset, max_edges=100)
    highlights = ["system:saidit", "system:file_system", "Agent/person:john_windward"]
    return _network_figure(
        nodes,
        edges,
        highlights,
        "Sistema resumido: agentes, arquivos e SaidIt nos três casos",
        640,
        labels_all=True,
    )


def _network_figure(nodes, edges, highlight_entities, title, height, labels_all=False):
    if nodes.empty:
        return go.Figure().update_layout(title="Sem dados de rede")
    graph = nx.Graph()
    for _, row in edges.iterrows():
        graph.add_edge(row["source"], row["target"], weight=float(row["weight"]))
    pos = nx.spring_layout(graph, seed=12, k=1.05 if len(nodes) < 35 else 0.65, iterations=140, weight="weight")
    traces = []
    for _, row in edges.iterrows():
        x0, y0 = pos[row["source"]]
        x1, y1 = pos[row["target"]]
        traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None], mode="lines",
            line=dict(width=max(0.8, min(5, math.log1p(row["weight"]))), color="rgba(100,116,139,0.25)"),
            hoverinfo="skip", showlegend=False,
        ))
    for entity_type, color in TYPE_COLORS.items():
        view = nodes[nodes["type"].eq(entity_type)]
        if view.empty:
            continue
        xs, ys, labels, sizes, hover = [], [], [], [], []
        for _, node in view.iterrows():
            if node["id"] not in pos:
                continue
            x, y = pos[node["id"]]
            xs.append(x); ys.append(y)
            labels.append(node["label"] if labels_all or node["id"] in highlight_entities else "")
            sizes.append(12 + min(25, math.sqrt(node["degree_weight"]) * 1.3))
            hover.append(f"{node['label']}<br>tipo: {node['type']}<br>peso de interação: {node['degree_weight']:.0f}")
        traces.append(go.Scatter(
            x=xs, y=ys, mode="markers+text", text=labels, textposition="top center",
            marker=dict(size=sizes, color=color, line=dict(width=1.5, color="#FFFFFF")),
            hovertext=hover, hoverinfo="text", name=entity_type.title(),
        ))
    # sobreposição dos destaques para garantir visibilidade
    for entity in highlight_entities:
        row = nodes[nodes["id"].eq(entity)]
        if row.empty or entity not in pos:
            continue
        x, y = pos[entity]
        traces.append(go.Scatter(
            x=[x], y=[y], mode="markers+text", text=[row.iloc[0]["label"]], textposition="bottom center",
            marker=dict(size=28, color="#DC2626", line=dict(width=3, color="#FFFFFF")),
            hovertext=[row.iloc[0]["label"]], hoverinfo="text", showlegend=False,
        ))
    fig = go.Figure(traces)
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    fig.update_layout(title=title, legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0))
    return style_figure(fig, height)


def timeline(events: pd.DataFrame, title: str) -> go.Figure:
    fig = px.scatter(
        events, x="datetime_local_desafio", y="short_name", color="short_name",
        color_discrete_map=EVENT_COLORS,
        hover_data=["id", "source_label", "target_label", "path", "content_source", "forum"],
        labels={"datetime_local_desafio": "Tempo analítico (UTC-7)", "short_name": "Evento"}, title=title,
    )
    fig.update_traces(marker=dict(size=10, opacity=0.85))
    return style_figure(fig, 560)


def exact_sequence(seq: pd.DataFrame, title: str) -> go.Figure:
    """Sequência legível com descrição do agente/ação no eixo vertical."""
    if seq.empty:
        return go.Figure().update_layout(title="Sem sequência disponível")
    view = seq.copy()
    start = view["datetime_local_desafio"].min()
    view["horas_desde_inicio"] = (view["datetime_local_desafio"] - start).dt.total_seconds() / 3600
    view["rotulo"] = view.apply(
        lambda r: f"{int(r['ordem']):02d}. {r['descricao']}  |  {r['datetime_local_desafio'].strftime('%d/%m %H:%M:%S')}", axis=1
    )
    colors = [EVENT_COLORS.get(event, "#64748B") for event in view["short_name"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=view["horas_desde_inicio"], y=view["rotulo"], mode="lines+markers",
        line=dict(color="rgba(100,116,139,0.35)", width=2),
        marker=dict(size=14, color=colors, line=dict(width=1, color="#FFFFFF")),
        hovertext=[f"{r.descricao}<br>{r.datetime_local_desafio}<br>Intervalo: {r.delta_s:.0f} s<br>Arquivo: {r.path}<br>Fonte: {r.content_source}" for r in view.itertuples()],
        hoverinfo="text", showlegend=False,
    ))
    fig.update_yaxes(autorange="reversed", title="")
    fig.update_xaxes(title="Horas desde o primeiro evento exibido")
    fig.update_layout(title=title)
    return style_figure(fig, max(560, 44 * len(view)))


def final_outcome_chart(seq: pd.DataFrame, title: str) -> go.Figure:
    """Ampliação do último encaminhamento, publicação e exclusões."""
    if seq.empty:
        return go.Figure().update_layout(title="Sem sequência disponível")
    post_rows = seq.index[seq["short_name"].eq("saidit_post")].tolist()
    if not post_rows:
        view = seq.tail(4).copy()
    else:
        post_pos = seq.index.get_loc(post_rows[0])
        before = seq.iloc[max(0, post_pos - 1):post_pos]
        after = seq.iloc[post_pos:post_pos + 3]
        view = pd.concat([before, after]).copy()
    post_time = view.loc[view["short_name"].eq("saidit_post"), "datetime_local_desafio"].iloc[0]
    view["segundos_post"] = (view["datetime_local_desafio"] - post_time).dt.total_seconds()
    view["evento_slide"] = view.apply(lambda r: f"{r['descricao']}\n{r['datetime_local_desafio'].strftime('%H:%M:%S')}", axis=1)
    colors = [EVENT_COLORS.get(event, "#64748B") for event in view["short_name"]]
    fig = go.Figure(go.Scatter(
        x=view["segundos_post"], y=view["evento_slide"], mode="lines+markers",
        line=dict(color="rgba(100,116,139,0.35)", width=2),
        marker=dict(size=17, color=colors, line=dict(width=1, color="#FFFFFF")),
        hovertext=view["descricao"], hoverinfo="text", showlegend=False,
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="#111827", annotation_text="publicação (t = 0)")
    fig.update_xaxes(title="Segundos em relação à publicação", dtick=1)
    fig.update_yaxes(autorange="reversed", title="")
    fig.update_layout(title=title)
    return style_figure(fig, 430)


def relative_timeline(events: pd.DataFrame, title: str) -> go.Figure:
    view = events[events["short_name"].isin(["queue_subordinate_task", "saidit_post", "delete_file", "create_file", "read_file"])].copy()
    fig = px.scatter(
        view, x="segundos_relativos_post", y="short_name", color="short_name", color_discrete_map=EVENT_COLORS,
        hover_data=["datetime_local_desafio", "source_label", "target_label", "path", "content_source"],
        labels={"segundos_relativos_post": "Segundos em relação ao post", "short_name": "Evento"}, title=title,
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#111827", annotation_text="post", annotation_position="top")
    fig.update_traces(marker=dict(size=11))
    return style_figure(fig, 440)


def t0_small_multiples(events: pd.DataFrame) -> go.Figure:
    """Compara somente o último encaminhamento, post e exclusões de cada caso."""
    selected = []
    for case_name, group in events.groupby("caso", sort=False):
        relevant = group[group["short_name"].isin(["saidit_post", "delete_file"])].copy()
        delegations = group[(group["short_name"].eq("queue_subordinate_task")) & (group["segundos_relativos_post"] < 0)]
        if not delegations.empty:
            relevant = pd.concat([delegations.nlargest(1, "segundos_relativos_post"), relevant], ignore_index=True)
        relevant = relevant[relevant["segundos_relativos_post"].between(-15, 5)]
        selected.append(relevant)
    view = pd.concat(selected, ignore_index=True) if selected else events.iloc[0:0]
    # separa as duas exclusões visualmente
    view["ordem_local"] = view.groupby(["caso", "short_name"]).cumcount()
    view["evento_rotulo"] = view.apply(
        lambda r: "delete_file 1" if r["short_name"] == "delete_file" and r["ordem_local"] == 0
        else ("delete_file 2" if r["short_name"] == "delete_file" else r["short_name"]), axis=1
    )
    fig = px.scatter(
        view, x="segundos_relativos_post", y="evento_rotulo", color="short_name", facet_row="caso",
        color_discrete_map=EVENT_COLORS, title="Desfecho dos três casos alinhado pela publicação (t = 0)",
        labels={"segundos_relativos_post": "Segundos em relação ao post", "evento_rotulo": "Evento"},
        hover_data=["datetime_local_desafio", "source_label", "target_label", "path", "content_source"],
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#111827")
    fig.update_xaxes(range=[-15, 5], dtick=2)
    fig.update_yaxes(matches=None, categoryorder="array", categoryarray=["queue_subordinate_task", "saidit_post", "delete_file 1", "delete_file 2"])
    fig.update_traces(marker=dict(size=12))
    return style_figure(fig, 760)


def provenance_flow(case_name: str, provenance: pd.DataFrame) -> go.Figure:
    labels = [
        f"{case_name}.txt",
        f"{case_name}_further_instructions.md",
        "Cadeia de agentes",
        "Post no SaidIt",
        "Exclusão dos dois arquivos",
    ]
    edges = [(0, 3, 1), (1, 2, 1), (2, 3, 1), (3, 4, 2)]
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=labels, pad=28, thickness=24, color=["#16A34A", "#7C3AED", "#2563EB", "#F97316", "#DC2626"], line=dict(color="#334155", width=0.8)),
        link=dict(source=[e[0] for e in edges], target=[e[1] for e in edges], value=[e[2] for e in edges], color="rgba(100,116,139,0.32)"),
        textfont=dict(size=15, color="#111827"),
    ))
    fig.update_layout(title=f"Proveniência operacional e processual - {case_name}")
    return style_figure(fig, 500)


def sankey_chain(events: pd.DataFrame, case_name: str) -> go.Figure:
    chain = delegation_edges(events)
    if chain.empty:
        return go.Figure().update_layout(title="Sem delegações")
    counts = chain.groupby(["source_label", "target_label"], as_index=False).size().rename(columns={"size": "weight"})
    counts = pd.concat([counts, pd.DataFrame([{"source_label": "John Windward", "target_label": "SaidIt", "weight": 1}])], ignore_index=True)
    nodes = pd.unique(counts[["source_label", "target_label"]].values.ravel("K")).tolist()
    idx = {node: i for i, node in enumerate(nodes)}
    fig = go.Figure(go.Sankey(node=dict(label=nodes, pad=16, thickness=18), link=dict(source=counts["source_label"].map(idx), target=counts["target_label"].map(idx), value=counts["weight"], hovertemplate="%{source.label} -> %{target.label}<br>delegações: %{value}<extra></extra>")))
    fig.update_layout(title=f"Fluxo agregado das delegações - {case_name}")
    return style_figure(fig, 650)


def comparison_bars(summary: pd.DataFrame, metric: str, title: str) -> go.Figure:
    fig = px.bar(summary, x="caso", y=metric, text_auto=True, title=title, labels={metric: title, "caso": "Caso"})
    fig.update_traces(marker_color="#2563EB")
    return style_figure(fig, 390)


def cumulative_chart(cumulative: pd.DataFrame) -> go.Figure:
    view = cumulative.copy()
    view["dias_antes_post"] = view["segundos_relativos_post"] / 86400
    fig = px.line(view, x="dias_antes_post", y="eventos_acumulados", color="caso", title="Eventos acumulados antes da publicação", labels={"dias_antes_post": "Dias em relação ao post", "eventos_acumulados": "Eventos acumulados"})
    fig.add_vline(x=0, line_dash="dash", line_color="#111827")
    return style_figure(fig, 480)


def anomaly_funnel_chart(funnel: pd.DataFrame) -> go.Figure:
    fig = px.funnel(funnel, x="quantidade", y="criterio", title="Funil de raridade da regra de detecção")
    fig.update_traces(textinfo="value")
    return style_figure(fig, 430)


def baseline_post_chart(baseline: pd.DataFrame) -> go.Figure:
    common = int((~baseline["sinalizado_regra"]).sum())
    flagged = int(baseline["sinalizado_regra"].sum())
    counts = pd.DataFrame({"categoria": ["Postagens comuns", "content_source + 2 exclusões"], "quantidade": [common, flagged]})
    fig = px.bar(counts, x="categoria", y="quantidade", text_auto=True, title="Baseline das publicações no SaidIt")
    fig.update_traces(marker_color=["#2563EB", "#F97316"])
    return style_figure(fig, 410)


def deletion_latency_chart(baseline: pd.DataFrame) -> go.Figure:
    counts = pd.DataFrame({
        "categoria": ["Sem exclusão rápida", "Duas exclusões em até 5 s"],
        "quantidade": [int((baseline["delete_file_ate_5s"] < 2).sum()), int((baseline["delete_file_ate_5s"] >= 2).sum())],
    })
    fig = px.bar(counts, x="categoria", y="quantidade", text_auto=True, title="Comportamento após a publicação")
    fig.update_traces(marker_color=["#2563EB", "#DC2626"])
    return style_figure(fig, 410)


def agent_event_heatmap(matrix: pd.DataFrame) -> go.Figure:
    if matrix.empty:
        return go.Figure().update_layout(title="Sem matriz")
    selected = [c for c in ["queue_subordinate_task", "create_file", "read_file", "saidit_post", "delete_file"] if c in matrix.columns]
    view = matrix[selected].copy()
    view = view.loc[view.sum(axis=1).sort_values(ascending=False).head(25).index]
    fig = px.imshow(view, text_auto=True, aspect="auto", title="Agente x tipo de evento", labels={"x": "Evento", "y": "Agente", "color": "Quantidade"})
    return style_figure(fig, 630)


def intervention_convergence(posts: pd.DataFrame) -> go.Figure:
    sources = posts["case_name"].tolist()
    labels = sources + ["Validação no SaidIt", "Publicação autorizada / revisão"]
    gate_idx = len(sources)
    output_idx = len(sources) + 1
    fig = go.Figure(go.Sankey(
        node=dict(label=labels, pad=28, thickness=24, color=["#2563EB"] * len(sources) + ["#DC2626", "#F97316"], line=dict(color="#334155", width=0.8)),
        link=dict(source=list(range(len(sources))) + [gate_idx], target=[gate_idx] * len(sources) + [output_idx], value=[1] * len(sources) + [len(sources)], color="rgba(100,116,139,0.30)"),
        textfont=dict(size=15, color="#111827"),
    ))
    fig.update_layout(title="Um único ponto de controle cobre os três incidentes")
    return style_figure(fig, 480)
