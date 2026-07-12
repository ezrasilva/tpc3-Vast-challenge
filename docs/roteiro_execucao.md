# Execução da solução

## Ambiente

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

Instalação:

```bash
python -m pip install -r requirements.txt
```

## Dashboard

```bash
streamlit run app.py
```

Use o modo **História guiada** para a gravação do vídeo e o modo **Exploração livre** para demonstrar filtros e inspeção dos registros.

## Notebook

```bash
jupyter notebook notebooks/MC2_analise_academica_final.ipynb
```

## Exportação das figuras

```bash
python scripts/export_figures.py
```

## Teste básico

```bash
python smoke_test.py
```
