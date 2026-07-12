# VAST Challenge 2026 — Mini-Challenge 2

Solução desenvolvida para o TP3 de Visualização da Informação.

## Autores

- **Esdras Silva** — matrícula 202411140033
- **João Pedro Silva** — matrícula 202411140020

## Objetivo

Investigar como uma publicação baseada no arquivo `SwiftWren.txt` chegou ao SaidIt, reconstruir sua cadeia de eventos, identificar a origem provável do conteúdo, comparar ocorrências anteriores e propor um único ponto de intervenção.

## Estrutura do repositório

```text
vast_mc2_final_repo/
├── app.py
├── requirements.txt
├── smoke_test.py
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── analysis.py
│   └── charts.py
├── notebooks/
│   └── MC2_analise_academica_final.ipynb
├── data/
│   └── VAST_Challenge_2026_MC2.zip
├── presentation/
│   ├── slides_vast_mc2_storytelling_final.pdf
│   └── slides_vast_mc2_storytelling_final_source.zip
├── exports/
│   └── figures/
├── scripts/
│   └── export_figures.py
├── docs/
│   ├── metodologia_storytelling.md
│   ├── roteiro_execucao.md
│   └── checklist_requisitos.md
└── .streamlit/
    └── config.toml
```

## Storytelling da solução

A investigação segue oito etapas:

1. incidente;
2. raridade;
3. visão do sistema;
4. reconstrução da cadeia;
5. origem do conteúdo;
6. comparação histórica;
7. intervenção;
8. respostas finais.

O Streamlit possui um modo **História guiada** para conduzir a apresentação e um modo **Exploração livre** para filtros e inspeção detalhada.

## Instalação

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

```bash
python -m pip install -r requirements.txt
```

## Execução do Streamlit

```bash
streamlit run app.py
```

## Execução do notebook

```bash
jupyter notebook notebooks/MC2_analise_academica_final.ipynb
```

## Regeneração das figuras

```bash
python scripts/export_figures.py
```

As figuras em `exports/figures/` são utilizadas na apresentação.

## Validação

```bash
python smoke_test.py
```

## Observação temporal

Os timestamps UTC originais são preservados. A representação UTC−7 é utilizada apenas para compatibilidade com o horário analítico adotado na investigação.
