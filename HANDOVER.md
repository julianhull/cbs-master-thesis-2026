# Thesis Handover — Data & Replication Guide

**Title:** Multimodal Peer Identification for Relative Valuation:
Fusing Financial Ratios and LLM-Condensed 10-K Disclosures

**Authors:** Julian Hüllstrunk & Moritz Landau
**Programme:** MSc Finance & Strategic Management, Copenhagen Business School
**Year:** 2026

**GitHub repository:** https://github.com/julianhull/cbs-master-thesis-2026

---

## What You Receive

The handover ZIP (`thesis-handover.zip`) contains:

```
thesis-handover/
├── HANDOVER.md                  ← this document
├── code/                        ← full replication repository (mirrors GitHub)
│   ├── config.py                ← all parameters and file paths
│   ├── requirements.txt
│   ├── notebooks/               ← N0–N11 Jupyter notebooks (execution order below)
│   ├── src/                     ← shared Python modules
│   └── data/results/            ← all computed outputs (peer lists, CSVs, LaTeX tables)
├── figures/                     ← all 48 thesis figures as PDFs
└── processed_data/              ← intermediate data (see section below)
    ├── panel_clean.parquet      ← cleaned financial panel (20,883 firm-years)
    ├── financials_normalized.parquet  ← z-normalised financial ratios (64 features)
    ├── multiples.parquet        ← EV/Sales, EV/Assets, MktCap/SEQ per firm-year
    ├── text_embeddings.parquet  ← 768-dim FinBERT embeddings (13,559 firm-years)
    └── Business_Summaries/      ← Gemini 400-word 10-K summaries per year (CSV)
```

**Not included:** Raw Compustat data (proprietary licence — see below) and raw
SEC 10-K text (646 MB — available from SEC EDGAR).

---

## Data Sources and Access

### 1. Financial Panel — Compustat via WRDS

The raw financial data requires a WRDS subscription (CBS has institutional access).

**Database:** Compustat North America — Fundamentals Annual (`comp.funda`)

**Query filters applied in N1 (`data_prep`):**
- Currency: USD only
- Data format: `datafmt = 'STD'`, `indfmt = 'INDL'`, `consol = 'C'`
- Fiscal years: 2020–2024
- Market cap filter: ≥ $50M (penny/micro-cap removal)
- Exchange: NYSE, NASDAQ, AMEX only
- Deduplicated on `(gvkey, datadate)` — keeps first record per firm-year

**Key variables pulled:** `gvkey, tic, conm, datadate, fyear, sic, at, sale, ebitda,
ceq, csho, prcc_f, dvt, capx, xrd, dltt, dlc, act, lct, rect, invt, ppent, dp,
ib, oibdp, xsga, cogs` (plus derived ratios in N2).

**FF49 industry classification:** Applied via `sic` using the Fama–French 49-industry
mapping. The mapping file is stored in `data/results/ff49_reference_mapping.csv`.

The raw pull is stored locally at `data/raw/financial/perfect_compustat_pull_v2.csv`
(not distributed — reproduced via WRDS query above).

---

### 2. 10-K Filings — SEC EDGAR

Raw 10-K text was scraped from SEC EDGAR using the `edgartools` Python library
(see N3 for the extraction pipeline). The scraping covers fiscal years 2020–2024
for all firms in the financial panel.

- **Raw text files (not in ZIP — 646 MB):** Available from SEC EDGAR directly at
  https://www.sec.gov/cgi-bin/browse-edgar or via `edgartools`
- **Pre-processed text (included in ZIP):** `processed_data/Business_Summaries/`
  contains the Gemini-condensed 400-word summaries used as model input

To re-scrape: run N3 (`text_pipeline`) with a valid Google Gemini API key set as
`GOOGLE_API_KEY` in the environment. Estimated cost: ~$5–10 USD at current API rates.
The Gemini 2.5 Flash model was used (`gemini-2.5-flash-preview-04-17`).

---

### 3. Processed Data (included in ZIP)

All intermediate artefacts needed to reproduce N4–N11 results without re-running
N1–N3 are included in `processed_data/`. Sizes:

| File | Size | Description |
|------|------|-------------|
| `panel_clean.parquet` | 28 MB | Full financial panel with FF49 codes |
| `financials_normalized.parquet` | 11 MB | Z-normalised ratios, 64 features |
| `multiples.parquet` | 2 MB | Valuation multiples (log-transformed) |
| `text_embeddings.parquet` | 58 MB | FinBERT sentence embeddings (768-dim) |
| `Business_Summaries/` | 39 MB | Gemini summaries, one CSV per year |

Copy these files into `code/data/processed/` before running the notebooks.

---

### 4. Computed Results (in GitHub and ZIP)

`data/results/` is fully committed to GitHub and included in the ZIP.
It contains all peer lists, evaluation outputs, and LaTeX tables:

| File | Description |
|------|-------------|
| `peers_m0/m1/m2/m3.parquet` | Peer lists for all four models |
| `results_main.csv` | Primary MdAPE results table |
| `selected_features.json` | The 64 financial features selected in N2 |
| `alpha_optimal.json` | Fusion weight α = 0.3 (optimised on 2020–2022) |
| `h4_final/h4_tables.tex` | Ready-to-compile LaTeX tables for H4 results |
| `peer_coherence.csv` | Structural diagnostics (hit rate, cosine similarity) |

---

## Reproducing the Results

### Requirements

```bash
pip install -r requirements.txt
```

Python 3.10+. Key packages: `pandas`, `numpy`, `scikit-learn`, `faiss-cpu`,
`transformers`, `torch`, `google-generativeai`, `edgartools`, `matplotlib`,
`seaborn`, `scipy`, `statsmodels`.

### Execution Order

Copy `processed_data/` contents to `code/data/processed/`, then:

| Notebook | Input | Output | Runtime |
|----------|-------|--------|---------|
| N1 data_prep | Compustat CSV | `panel_clean.parquet` | ~5 min |
| N2 financials | panel_clean | `financials_normalized.parquet` | ~5 min |
| N3 text_pipeline | Raw 10-K text | Gemini summaries + gemini_ready CSVs | ~2 h |
| N4 multiples | panel_clean | `multiples.parquet` | ~2 min |
| N5 m0_baseline | panel_clean | `peers_m0.parquet` | ~2 min |
| N6 knn_financial | financials_norm | `peers_m1.parquet` | ~5 min |
| N7 embeddings | Gemini summaries | `text_embeddings.parquet` | ~45 min |
| N8 knn_text | text_embeddings | `peers_m2.parquet` | ~15 min |
| N9 fusion | peers_m1, peers_m2 | `peers_m3.parquet`, `alpha_optimal.json` | ~5 min |
| N10 evaluation | all peer files | all result CSVs + figures | ~20 min |
| N11 h4 | N10 outputs | H4 tables + figures | ~10 min |
| N0 eda_complete | panel_clean | EDA figures | ~10 min |

N4–N6 can run in parallel after N1–N2. N7b (ablation) is optional.

### Skipping data preparation

If you have the processed data from the ZIP, you can skip N1–N3 entirely and
start from N4. All results in the thesis were produced by running N4–N11 on the
processed data provided.

---

## Key Parameters

All parameters are centralised in `config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `YEARS` | 2020–2024 | Evaluation window |
| `K_MAIN` | 10 | Primary peer count |
| `K_ROBUSTNESS` | [5, 10, 15, 20] | Robustness grid |
| `RANDOM_SEED` | 42 | All stochastic steps |
| `CI_LEVEL` | 0.95 | Bootstrap confidence intervals |
| `ALPHA_GRID` | 0.0–1.0 step 0.05 | Fusion weight search grid |
| Best alpha (M3) | 0.3 | 70% financial, 30% text |

---

## Figures

All 48 thesis figures are included as PDFs in `figures/`. They are regenerated
automatically when the notebooks are run — no manual steps required.
Figures are named by the notebook that produces them:
`eda_*` (N0), `n2_*` (N2), `n4_*` (N4) ... `n10_*` (N10), `h4_*` (N11).

---

## Contact

Julian Hüllstrunk — julian.huellstrunk@gmail.com
Moritz Landau
GitHub: https://github.com/julianhull/cbs-master-thesis-2026
