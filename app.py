"""Dashboard narrativo e exploratório do VAST Challenge 2026 — Mini-Challenge 2."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.analysis import (
    agent_event_matrix,
    anomaly_funnel,
    case_events,
    comparison_table,
    cumulative_events,
    file_based_posts,
    final_sequence,
    post_baseline,
    post_for_case,
    provenance_table,
    relative_to_post,
    similarity_table,
)
from src.charts import (
    agent_event_heatmap,
    anomaly_funnel_chart,
    baseline_post_chart,
    comparison_bars,
    cumulative_chart,
    deletion_latency_chart,
    event_distribution,
    exact_sequence,
    final_outcome_chart,
    intervention_convergence,
    provenance_flow,
    relative_timeline,
    system_network,
    system_network_summary,
    t0_small_multiples,
    timeline,
)
from src.data_loader import find_local_dataset, load_local_dataset, load_json_bytes, load_zip_bytes
from src.preprocessing import preprocess_events


st.set_page_config(page_title="VAST 2026 — MC2", page_icon="🔎", layout="wide")


@st.cache_data(show_spinner=False)
def load_local_cached(path: str):
    raw, org = load_local_dataset(path)
    return preprocess_events(raw), org


@st.cache_data(show_spinner=False)
def load_uploaded_cached(name: str, content: bytes):
    if name.lower().endswith(".zip"):
        raw, org = load_zip_bytes(content)
    else:
        raw, org = load_json_bytes(content)
    return preprocess_events(raw), org


def callout(title: str, text: str, kind: str = "info") -> None:
    icons = {"question": "❓", "evidence": "🔎", "finding": "💡", "limit": "⚠️", "decision": "✅"}
    icon = icons.get(kind, "ℹ️")
    st.markdown(
        f"""
        <div style="padding:0.9rem 1rem;border-radius:0.65rem;background:#f6f8fc;border-left:5px solid #315EFB;margin:0.4rem 0 1rem 0;">
        <strong>{icon} {title}</strong><br>{text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def progress_header(step: int, labels: list[str]) -> None:
    st.progress(step / len(labels))
    st.caption(" → ".join([f"**{label}**" if i + 1 == step else label for i, label in enumerate(labels)]))


st.title("VAST Challenge 2026 — Mini-Challenge 2")
st.caption("Investigação visual da cadeia anômala de publicação no SaidIt")

with st.sidebar:
    st.header("Base de dados")
    uploaded = st.file_uploader("Carregar ZIP original ou MC2 data.json", type=["zip", "json"])

try:
    if uploaded:
        df, org = load_uploaded_cached(uploaded.name, uploaded.getvalue())
    else:
        path = find_local_dataset()
        if path is None:
            st.warning("Carregue o ZIP original ou coloque o arquivo em data/.")
            st.stop()
        df, org = load_local_cached(str(path))
        with st.sidebar:
            st.success(f"Usando base local: {path}")
except Exception as exc:
    st.error(f"Erro no carregamento dos dados: {exc}")
    st.stop()

posts = file_based_posts(df)
summary = comparison_table(df, posts)
baseline = post_baseline(df)
funnel = anomaly_funnel(df, posts)
case_names = posts["case_name"].tolist()

with st.sidebar:
    st.header("Modo de uso")
    app_mode = st.radio(
        "Escolha a experiência",
        ["História guiada", "Exploração livre", "Figuras para slides"],
        help="A história guiada é indicada para a apresentação em vídeo; a exploração livre mantém todos os filtros analíticos.",
    )

    st.header("Filtros globais")
    selected_case = st.selectbox(
        "Caso investigado",
        case_names,
        index=case_names.index("SwiftWren") if "SwiftWren" in case_names else len(case_names) - 1,
    )
    event_options = sorted(df["short_name"].dropna().unique())
    selected_event_types = st.multiselect(
        "Tipos de evento",
        event_options,
        default=[e for e in ["queue_subordinate_task", "saidit_post", "delete_file", "create_file", "read_file"] if e in event_options],
    )
    show_final_n = st.slider("Delegações finais na sequência exata", 5, 40, 15)

selected_events = case_events(df, selected_case)
selected_post = post_for_case(posts, selected_case)
relative_events = relative_to_post(selected_events, selected_post["datetime_utc"])
filtered_events = selected_events[selected_events["short_name"].isin(selected_event_types)] if selected_event_types else selected_events.copy()

if app_mode == "História guiada":
    story_labels = ["Incidente", "Raridade", "Sistema", "Cadeia", "Origem", "Histórico", "Intervenção", "Respostas"]
    story_step = st.sidebar.select_slider("Etapa da investigação", options=list(range(1, 9)), value=1, format_func=lambda x: f"{x}. {story_labels[x-1]}")
    progress_header(story_step, story_labels)

    if story_step == 1:
        st.header("1. O incidente")
        callout(
            "Pergunta investigativa",
            "Como uma publicação baseada em arquivo chegou ao SaidIt em 17 de maio de 2046 e por que os arquivos relacionados foram apagados segundos depois?",
            "question",
        )
        swift = summary[summary["caso"].eq("SwiftWren")].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Caso", "SwiftWren")
        c2.metric("Publicador", "John Windward")
        c3.metric("Fórum", "general")
        c4.metric("Exclusões após o post", int(swift["qtd_delete_file"]))
        st.plotly_chart(final_outcome_chart(final_sequence(case_events(df, "SwiftWren"), n_delegations=5), "Desfecho do incidente SwiftWren"), width='stretch')
        callout("Ponto de partida", "A investigação começa pelo desfecho: postagem por arquivo, seguida por duas exclusões quase imediatas.", "evidence")

    elif story_step == 2:
        st.header("2. O comportamento é raro?")
        callout("Pergunta investigativa", "Esse padrão aparece com frequência no SaidIt ou representa uma exceção?", "question")
        left, right = st.columns(2)
        with left:
            st.plotly_chart(baseline_post_chart(baseline), width='stretch')
        with right:
            st.plotly_chart(anomaly_funnel_chart(funnel), width='stretch')
        callout("Primeira descoberta", "Das 108 publicações no SaidIt, apenas três usam content_source e as mesmas três apresentam duas exclusões rápidas.", "finding")

    elif story_step == 3:
        st.header("3. Onde o incidente ocorre no sistema?")
        callout("Pergunta investigativa", "Quais componentes conectam as delegações, os arquivos e a publicação?", "question")
        st.plotly_chart(system_network_summary(df, case_names), width='stretch')
        callout("Segunda descoberta", "As cadeias percorrem agentes distintos, mas convergem para John Windward, o sistema de arquivos e o SaidIt.", "finding")
        with st.expander("Explorar a rede completa"):
            st.plotly_chart(system_network(df, highlight_entities=["system:saidit", "system:file_system", "Agent/person:john_windward"]), width='stretch')

    elif story_step == 4:
        st.header("4. Reconstruindo a cadeia")
        callout("Pergunta investigativa", "Qual sequência de eventos produziu a publicação?", "question")
        seq = final_sequence(case_events(df, "SwiftWren"), n_delegations=show_final_n)
        st.plotly_chart(exact_sequence(seq, "Sequência cronológica final — SwiftWren"), width='stretch')
        st.plotly_chart(final_outcome_chart(seq, "Ampliação dos segundos finais — SwiftWren"), width='stretch')
        callout("Terceira descoberta", "A última delegação chega a John Windward dois segundos antes do post. Os dois arquivos são apagados nos dois segundos seguintes.", "finding")
        table = seq[["ordem", "datetime_local_desafio", "short_name", "descricao", "delta_s"]].copy()
        table["datetime_local_desafio"] = table["datetime_local_desafio"].dt.strftime("%d/%m/%Y %H:%M:%S")
        with st.expander("Ver registros que sustentam a sequência"):
            st.dataframe(table, width='stretch', hide_index=True)

    elif story_step == 5:
        st.header("5. De onde veio o conteúdo?")
        callout("Pergunta investigativa", "Qual é a origem operacional e processual do conteúdo publicado?", "question")
        prov = provenance_table(df, "SwiftWren")
        st.plotly_chart(provenance_flow("SwiftWren", prov), width='stretch')
        st.table(pd.DataFrame([
            {"Nível": "Origem operacional", "Evidência": "SwiftWren.txt usado como content_source"},
            {"Nível": "Origem processual", "Evidência": "SwiftWren_further_instructions.md propagado na cadeia"},
            {"Nível": "Desfecho", "Evidência": "publicação seguida por exclusão dos dois arquivos"},
        ]))
        callout("Quarta descoberta", "É possível reconstruir o processo de produção e publicação, mas o texto literal dos arquivos não foi preservado.", "finding")
        callout("Limitação", "A interpretação descreve o comportamento observado; ela não recupera nem inventa o conteúdo exato da mensagem.", "limit")

    elif story_step == 6:
        st.header("6. SwiftWren foi um caso isolado?")
        callout("Pergunta investigativa", "Existem ocorrências anteriores com o mesmo mecanismo?", "question")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(comparison_bars(summary, "qtd_delegacoes", "Delegações por caso"), width='stretch')
        with c2:
            st.plotly_chart(comparison_bars(summary, "duracao_horas", "Duração da cadeia (horas)"), width='stretch')
        cumulative = cumulative_events(df, posts)
        st.plotly_chart(t0_small_multiples(cumulative), width='stretch')
        st.dataframe(similarity_table(df, posts), width='stretch', hide_index=True)
        callout("Quinta descoberta", "HiddenOrca, MellowOtter e SwiftWren diferem em escala, mas compartilham o mesmo desfecho operacional.", "finding")
        st.markdown("**Padrão:** content_source, John Windward, fórum general e duas exclusões.  \n**Tendência:** cadeias maiores acumulam mais eventos e duração.  \n**Outlier:** SwiftWren concentra 186 delegações e cerca de 188 horas.")

    elif story_step == 7:
        st.header("7. Onde intervir?")
        callout("Pergunta investigativa", "Qual ponto único interrompe os três casos sem bloquear toda a comunicação entre agentes?", "question")
        st.plotly_chart(intervention_convergence(posts), width='stretch')
        c1, c2, c3 = st.columns(3)
        c1.metric("Casos cobertos", f"{len(posts)} de {len(posts)}")
        c2.metric("Pontos modificados", "1")
        c3.metric("Posts comuns sinalizados", int(baseline[(~baseline["usa_content_source"]) & baseline["sinalizado_regra"]].shape[0]))
        st.markdown("**Regra proposta:** postagem por agente + content_source + cadeia de delegações + arquivo de instruções + exclusões rápidas.")
        callout("Decisão", "Posicionar a validação em system:saidit antes da publicação, preservando as delegações internas.", "decision")

    else:
        st.header("8. Respostas finais do Mini-Challenge")
        answers = [
            ("Como o post foi produzido?", "Uma cadeia de 186 delegações terminou com Chloe Ballast encaminhando a tarefa a John Windward. Dois segundos depois, John publicou SwiftWren.txt no SaidIt; os dois arquivos associados foram excluídos nos dois segundos seguintes."),
            ("Qual a origem e o significado provável?", "A origem operacional é SwiftWren.txt e a origem processual é SwiftWren_further_instructions.md. O conteúdo literal não foi preservado; os registros indicam uma mensagem preparada externamente, propagada entre agentes e removida após a publicação."),
            ("Isso já aconteceu antes?", "Sim. HiddenOrca e MellowOtter apresentam o mesmo mecanismo final, embora com cadeias menores."),
            ("Onde intervir?", "No SaidIt, imediatamente antes da publicação, aplicando validação à combinação de sinais observada."),
        ]
        for question, answer in answers:
            callout(question, answer, "decision")
        st.plotly_chart(intervention_convergence(posts), width='stretch')

elif app_mode == "Exploração livre":
    tabs = st.tabs(["Baseline e sistema", "Cadeia exata", "Origem", "Casos anteriores", "Intervenção", "Evidências"])

    with tabs[0]:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Eventos", f"{len(df):,}".replace(",", "."))
        c2.metric("Tipos de evento", df["short_name"].nunique())
        c3.metric("Posts SaidIt", int(df["short_name"].eq("saidit_post").sum()))
        c4.metric("Posts com content_source", int(len(posts)))
        c5.metric("Casos sinalizados", int(baseline["sinalizado_regra"].sum()))
        left, right = st.columns(2)
        with left:
            st.plotly_chart(baseline_post_chart(baseline), width='stretch')
        with right:
            st.plotly_chart(deletion_latency_chart(baseline), width='stretch')
        st.plotly_chart(anomaly_funnel_chart(funnel), width='stretch')
        st.plotly_chart(system_network_summary(df, case_names), width='stretch')
        with st.expander("Rede completa"):
            st.plotly_chart(system_network(df, highlight_entities=["system:saidit", "system:file_system", "Agent/person:john_windward"]), width='stretch')
        st.plotly_chart(event_distribution(df), width='stretch')

    with tabs[1]:
        seq = final_sequence(selected_events, n_delegations=show_final_n)
        st.plotly_chart(exact_sequence(seq, f"Sequência cronológica final — {selected_case}"), width='stretch')
        st.plotly_chart(final_outcome_chart(seq, f"Ampliação do desfecho — {selected_case}"), width='stretch')
        st.plotly_chart(timeline(filtered_events, f"Linha do tempo filtrada — {selected_case}"), width='stretch')
        st.plotly_chart(relative_timeline(relative_events[relative_events["segundos_relativos_post"].between(-1800, 30)], f"Eventos próximos ao post — {selected_case}"), width='stretch')

    with tabs[2]:
        prov = provenance_table(df, selected_case)
        st.plotly_chart(provenance_flow(selected_case, prov), width='stretch')
        st.dataframe(prov, width='stretch', hide_index=True, height=420)

    with tabs[3]:
        summary_view = summary.copy()
        summary_view["data_post"] = pd.to_datetime(summary_view["data_post"]).dt.strftime("%d/%m/%Y %H:%M")
        st.dataframe(summary_view, width='stretch', hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(comparison_bars(summary, "qtd_delegacoes", "Delegações por caso"), width='stretch')
        with c2:
            st.plotly_chart(comparison_bars(summary, "duracao_horas", "Duração da cadeia (horas)"), width='stretch')
        cumulative = cumulative_events(df, posts)
        st.plotly_chart(cumulative_chart(cumulative), width='stretch')
        st.plotly_chart(t0_small_multiples(cumulative), width='stretch')
        st.dataframe(similarity_table(df, posts), width='stretch', hide_index=True)
        all_case_events = pd.concat([case_events(df, c) for c in case_names], ignore_index=True)
        st.plotly_chart(agent_event_heatmap(agent_event_matrix(all_case_events)), width='stretch')

    with tabs[4]:
        st.plotly_chart(intervention_convergence(posts), width='stretch')
        st.plotly_chart(anomaly_funnel_chart(funnel), width='stretch')
        st.dataframe(baseline.sort_values(["sinalizado_regra", "usa_content_source"], ascending=False), width='stretch', hide_index=True)

    with tabs[5]:
        display_columns = ["id", "datetime_local_desafio", "short_name", "source_entity", "target_agent_clean", "path", "content_source", "forum", "details_str"]
        st.dataframe(filtered_events[display_columns], width='stretch', hide_index=True, height=620)
        st.download_button("Baixar evidências filtradas em CSV", filtered_events[display_columns].to_csv(index=False).encode("utf-8"), file_name=f"{selected_case}_evidencias.csv", mime="text/csv")

else:
    st.header("Figuras estáticas para a apresentação")
    st.markdown("As figuras abaixo foram exportadas em PNG de alta resolução e reunidas em um PDF.")
    export_dir = Path(__file__).parent / "exports" / "figures"
    files = [
        ("01_baseline.png", "Baseline das publicações"),
        ("02_funil_raridade.png", "Funil de raridade"),
        ("03_rede_resumida.png", "Visão resumida do sistema"),
        ("04_cadeia_final_swiftwren.png", "Cadeia final do SwiftWren"),
        ("05_proveniencia_swiftwren.png", "Proveniência operacional e processual"),
        ("06_comparacao_quantitativa.png", "Comparação quantitativa"),
        ("07_desfecho_t0.png", "Desfecho alinhado em t = 0"),
        ("08_intervencao.png", "Convergência para a intervenção"),
    ]
    if export_dir.exists():
        for filename, caption in files:
            path = export_dir / filename
            if path.exists():
                st.image(str(path), caption=caption, width='stretch')
                st.download_button(f"Baixar {caption} (PNG)", path.read_bytes(), file_name=filename, mime="image/png", key=f"download_{filename}")
        pdf_path = export_dir / "figuras_para_slides.pdf"
        if pdf_path.exists():
            st.download_button("Baixar todas as figuras em PDF", pdf_path.read_bytes(), file_name=pdf_path.name, mime="application/pdf")
    else:
        st.info("Execute `python scripts/export_figures.py` para gerar as figuras.")

st.divider()
st.caption("Autores: Esdras Silva (202411140033) e João Pedro Silva (202411140020). Horários UTC−7 são representação analítica; timestamps UTC originais foram preservados.")
