"""Exporta figuras estáticas, em PNG e PDF, para uso nos slides."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import networkx as nx
import numpy as np
import pandas as pd

from src.analysis import (
    anomaly_funnel,
    case_events,
    comparison_table,
    cumulative_events,
    file_based_posts,
    final_sequence,
    post_baseline,
    post_for_case,
    relative_to_post,
)
from src.data_loader import load_local_dataset
from src.preprocessing import preprocess_events, clean_entity_name

OUT = ROOT / "exports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, filename: str, pdf: PdfPages) -> None:
    fig.tight_layout()
    path = OUT / filename
    fig.savefig(path, dpi=220, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def baseline_figure(baseline: pd.DataFrame):
    common = int((~baseline["sinalizado_regra"]).sum())
    flagged = int(baseline["sinalizado_regra"].sum())
    fig, ax = plt.subplots(figsize=(12, 6.3))
    labels = ["Postagens comuns", "content_source +\n2 exclusões em até 5 s"]
    values = [common, flagged]
    bars = ax.bar(labels, values)
    ax.set_title("Baseline das publicações no SaidIt", fontsize=18, weight="bold")
    ax.set_ylabel("Quantidade de publicações")
    ax.set_ylim(0, max(values) * 1.15)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, value + 2, str(value), ha="center", fontsize=15, weight="bold")
    ax.text(0.5, -0.18, "Apenas 3 de 108 publicações combinam content_source e duas exclusões rápidas.", transform=ax.transAxes, ha="center", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def funnel_figure(funnel: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(13, 7))
    y = np.arange(len(funnel))
    widths = funnel["quantidade"].to_numpy()
    bars = ax.barh(y, widths)
    ax.set_yticks(y, funnel["criterio"])
    ax.invert_yaxis()
    ax.set_title("Funil de raridade da regra de detecção", fontsize=18, weight="bold")
    ax.set_xlabel("Quantidade de publicações")
    for bar, value in zip(bars, widths):
        ax.text(max(value / 2, 1.5), bar.get_y() + bar.get_height()/2, str(int(value)), va="center", ha="center", fontsize=13, weight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.text(0.5, -0.12, "A combinação completa dos sinais ocorre somente nos três casos investigados.", transform=ax.transAxes, ha="center", fontsize=12)
    return fig


def summarized_network_figure(df: pd.DataFrame, case_names: list[str]):
    subset = df[df["search_text"].str.contains("|".join(case_names), case=False, na=False)].copy()
    pairs = []
    for _, row in subset.iterrows():
        parties = row.get("parties")
        if isinstance(parties, list) and len(parties) >= 2:
            pairs.append((parties[0], parties[1]))
    graph = nx.Graph()
    for source, target in pairs:
        if graph.has_edge(source, target):
            graph[source][target]["weight"] += 1
        else:
            graph.add_edge(source, target, weight=1)
    pos = nx.spring_layout(graph, seed=11, k=1.15, iterations=180, weight="weight")
    fig, ax = plt.subplots(figsize=(14, 8))
    weights = [0.6 + np.log1p(graph[u][v]["weight"]) for u, v in graph.edges()]
    nx.draw_networkx_edges(graph, pos, width=weights, alpha=0.25, ax=ax)
    systems = [n for n in graph if n.startswith("system:")]
    agents = [n for n in graph if n.startswith("Agent/person:")]
    nx.draw_networkx_nodes(graph, pos, nodelist=agents, node_size=520, ax=ax)
    nx.draw_networkx_nodes(graph, pos, nodelist=systems, node_size=850, node_shape="s", ax=ax)
    highlight = [n for n in ["Agent/person:john_windward", "system:saidit", "system:file_system"] if n in graph]
    nx.draw_networkx_nodes(graph, pos, nodelist=highlight, node_size=1100, linewidths=2.5, edgecolors="black", ax=ax)
    labels = {node: clean_entity_name(node) for node in graph}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8.5, ax=ax)
    ax.set_title("Visão resumida do sistema nos três incidentes", fontsize=18, weight="bold")
    ax.text(0.5, -0.04, "A rede foi filtrada aos eventos de HiddenOrca, MellowOtter e SwiftWren.", transform=ax.transAxes, ha="center", fontsize=11)
    ax.axis("off")
    return fig


def final_chain_figure(seq: pd.DataFrame):
    post_positions = seq.index[seq["short_name"].eq("saidit_post")].tolist()
    post_pos = seq.index.get_loc(post_positions[0])
    view = pd.concat([seq.iloc[max(0, post_pos-1):post_pos], seq.iloc[post_pos:post_pos+3]]).copy().reset_index(drop=True)
    post_time = view.loc[view["short_name"].eq("saidit_post"), "datetime_local_desafio"].iloc[0]
    view["seconds"] = (view["datetime_local_desafio"] - post_time).dt.total_seconds()
    fig, ax = plt.subplots(figsize=(14, 7.5))
    y = np.arange(len(view))[::-1]
    ax.plot(view["seconds"], y, linewidth=2, alpha=0.5)
    ax.scatter(view["seconds"], y, s=180, zorder=3)
    for idx, row in view.iterrows():
        label = f"{row['datetime_local_desafio'].strftime('%H:%M:%S')}\n{row['descricao']}"
        ax.annotate(label, (row["seconds"], y[idx]), xytext=(10, 0), textcoords="offset points", va="center", fontsize=11)
    ax.axvline(0, linestyle="--", linewidth=1.5)
    ax.text(0.1, max(y)+0.2, "publicação (t = 0)", fontsize=11)
    ax.set_title("Desfecho cronológico do caso SwiftWren", fontsize=18, weight="bold")
    ax.set_xlabel("Segundos em relação à publicação")
    ax.set_yticks([])
    ax.set_xlim(min(-5, view["seconds"].min()-1), max(4, view["seconds"].max()+1))
    ax.spines[["top", "right", "left"]].set_visible(False)
    return fig


def provenance_figure(case_name: str):
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.axis("off")
    boxes = [
        (0.03, 0.58, 0.21, 0.18, f"Arquivo-fonte\n{case_name}.txt"),
        (0.03, 0.22, 0.25, 0.18, f"Arquivo de instruções\n{case_name}_further_instructions.md"),
        (0.36, 0.22, 0.20, 0.18, "Cadeia de agentes"),
        (0.64, 0.43, 0.18, 0.18, "Post no SaidIt\nfórum general"),
        (0.86, 0.43, 0.12, 0.18, "Exclusão dos\ndois arquivos"),
    ]
    for x, y, w, h, text in boxes:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", linewidth=1.5, facecolor="white")
        ax.add_patch(patch)
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=11, weight="bold")
    arrows = [((0.24, 0.67), (0.64, 0.52)), ((0.28, 0.31), (0.36, 0.31)), ((0.56, 0.31), (0.64, 0.48)), ((0.82, 0.52), (0.86, 0.52))]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18, linewidth=2))
    ax.set_title("Proveniência operacional e processual - SwiftWren", fontsize=18, weight="bold", pad=18)
    ax.text(0.5, 0.05, "O texto literal não foi preservado; a visualização representa a origem e o comportamento observados nos registros.", ha="center", fontsize=11)
    return fig


def comparison_figure(summary: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    cases = summary["caso"].tolist()
    values = [("qtd_delegacoes", "Delegações por caso"), ("duracao_horas", "Duração da cadeia (horas)")]
    for ax, (metric, title) in zip(axes, values):
        bars = ax.bar(cases, summary[metric])
        ax.set_title(title, fontsize=15, weight="bold")
        for bar, value in zip(bars, summary[metric]):
            ax.text(bar.get_x()+bar.get_width()/2, value + max(summary[metric])*0.025, f"{value:g}", ha="center", fontsize=12, weight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Comparação histórica dos incidentes", fontsize=18, weight="bold")
    fig.text(0.5, 0.01, "SwiftWren é o outlier quantitativo: maior cadeia, duração e volume de delegações.", ha="center", fontsize=11)
    return fig


def t0_figure(df: pd.DataFrame, posts: pd.DataFrame):
    rows = []
    for case_name in posts["case_name"]:
        events = case_events(df, case_name)
        post = post_for_case(posts, case_name)
        rel = relative_to_post(events, post["datetime_utc"])
        delegation = rel[(rel["short_name"].eq("queue_subordinate_task")) & (rel["segundos_relativos_post"] < 0)].nlargest(1, "segundos_relativos_post")
        key = pd.concat([delegation, rel[rel["short_name"].isin(["saidit_post", "delete_file"])]]).copy()
        key = key[key["segundos_relativos_post"].between(-15, 5)]
        key["case"] = case_name
        key["delete_order"] = key.groupby("short_name").cumcount()
        rows.append(key)
    view = pd.concat(rows, ignore_index=True)
    fig, ax = plt.subplots(figsize=(14, 7))
    ybase = {case: i*5 for i, case in enumerate(reversed(posts["case_name"].tolist()))}
    for _, row in view.iterrows():
        offset = 0
        if row["short_name"] == "saidit_post": offset = 1
        elif row["short_name"] == "delete_file": offset = 2 + int(row["delete_order"])
        y = ybase[row["case"]] + offset
        ax.scatter(row["segundos_relativos_post"], y, s=120)
        ax.annotate(row["short_name"].replace("queue_subordinate_task", "última delegação"), (row["segundos_relativos_post"], y), xytext=(5,5), textcoords="offset points", fontsize=9)
    for case, base in ybase.items():
        ax.text(-15.5, base+1.5, case, ha="right", va="center", fontsize=12, weight="bold")
        ax.hlines([base, base+1, base+2, base+3], -15, 5, alpha=0.12)
    ax.axvline(0, linestyle="--", linewidth=1.5)
    ax.set_xlim(-15, 5)
    ax.set_yticks([])
    ax.set_xlabel("Segundos em relação à publicação")
    ax.set_title("Desfecho dos três casos alinhado em t = 0", fontsize=18, weight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    return fig


def intervention_figure(case_names: list[str]):
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.axis("off")
    ys = np.linspace(0.75, 0.25, len(case_names))
    for case, y in zip(case_names, ys):
        ax.add_patch(FancyBboxPatch((0.05, y-0.055), 0.19, 0.11, boxstyle="round,pad=0.02", facecolor="white", linewidth=1.5))
        ax.text(0.145, y, case, ha="center", va="center", fontsize=12, weight="bold")
        ax.add_patch(FancyArrowPatch((0.24, y), (0.48, 0.5), arrowstyle="-|>", mutation_scale=18, linewidth=2))
    ax.add_patch(FancyBboxPatch((0.48, 0.40), 0.22, 0.20, boxstyle="round,pad=0.02", facecolor="white", linewidth=2.5))
    ax.text(0.59, 0.5, "Validação no\nsystem:saidit", ha="center", va="center", fontsize=14, weight="bold")
    ax.add_patch(FancyArrowPatch((0.70, 0.5), (0.87, 0.5), arrowstyle="-|>", mutation_scale=20, linewidth=2))
    ax.add_patch(FancyBboxPatch((0.87, 0.40), 0.10, 0.20, boxstyle="round,pad=0.02", facecolor="white", linewidth=1.5))
    ax.text(0.92, 0.5, "Publicar\nou revisar", ha="center", va="center", fontsize=12, weight="bold")
    ax.set_title("Um único ponto de intervenção cobre os três incidentes", fontsize=18, weight="bold")
    ax.text(0.5, 0.08, "Cobertura: 3 de 3 casos | Pontos modificados: 1 | Posts comuns sinalizados: 0", ha="center", fontsize=12)
    return fig


def main() -> None:
    raw, _ = load_local_dataset(ROOT / "data" / "VAST_Challenge_2026_MC2.zip")
    df = preprocess_events(raw)
    posts = file_based_posts(df)
    baseline = post_baseline(df)
    funnel = anomaly_funnel(df, posts)
    summary = comparison_table(df, posts)
    swift = case_events(df, "SwiftWren")
    seq = final_sequence(swift, n_delegations=15)

    pdf_path = OUT / "figuras_para_slides.pdf"
    with PdfPages(pdf_path) as pdf:
        save(baseline_figure(baseline), "01_baseline.png", pdf)
        save(funnel_figure(funnel), "02_funil_raridade.png", pdf)
        save(summarized_network_figure(df, posts["case_name"].tolist()), "03_rede_resumida.png", pdf)
        save(final_chain_figure(seq), "04_cadeia_final_swiftwren.png", pdf)
        save(provenance_figure("SwiftWren"), "05_proveniencia_swiftwren.png", pdf)
        save(comparison_figure(summary), "06_comparacao_quantitativa.png", pdf)
        save(t0_figure(df, posts), "07_desfecho_t0.png", pdf)
        save(intervention_figure(posts["case_name"].tolist()), "08_intervencao.png", pdf)
    print(f"Figuras exportadas para {OUT}")


if __name__ == "__main__":
    main()
