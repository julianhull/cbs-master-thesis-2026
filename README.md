# Peer Identification via Financial Ratios, LLM-Condensed Text & Late Fusion
## CBS Master's Thesis — Replication Repository

> Does integrating LLM-condensed 10-K textual similarity with financial ratio
> similarity improve peer identification accuracy relative to FF49 industry
> classification, measured by relative valuation error (MdAPE on EV/Sales)?

---

## Quick Start

```bash
git clone <repo>
cd thesis
pip install -r requirements.txt
```

Place raw data files:
```
data/raw/financial/perfect_compustat_pull_v2.csv
data/raw/textual/Raw_Text/sec_text_2020.csv ... sec_text_2024.csv
```

Run in order:
```bash
jupyter notebook notebooks/N1_data_prep.ipynb        # ~5 min
jupyter notebook notebooks/N2_financials.ipynb        # ~5 min
jupyter notebook notebooks/N3_text_pipeline.ipynb     # ~2h  (Gemini API)
jupyter notebook notebooks/N4_multiples.ipynb         # ~2 min
jupyter notebook notebooks/N5_m0_baseline.ipynb       # ~2 min  (restricted to eval sample)
jupyter notebook notebooks/N6_knn_financial.ipynb     # ~5 min  (restricted to eval sample)
jupyter notebook notebooks/N7_embeddings.ipynb        # ~45 min (GPU recommended)
jupyter notebook notebooks/N7b_ablation.ipynb         # optional — embedding model ablation
jupyter notebook notebooks/N8_knn_text.ipynb          # ~15 min
jupyter notebook notebooks/N9_fusion.ipynb            # ~5 min
jupyter notebook notebooks/N10_evaluation.ipynb       # ~20 min — all H1/H2/H3 results + robustness
jupyter notebook notebooks/N11_h4.ipynb               # ~10 min — H4 disclosure-regime analysis
jupyter notebook notebooks/N0_eda_complete.ipynb      # ~10 min — all EDA figures
```

N3, N4, N5, and N6 can all run in parallel once N1 + N2 are complete.
N7 requires N3. N7b requires N7. N8 requires N7. N9 requires N6 + N8.
N10 requires N4 + N5 + N6 + N8 + N9. N11 requires N10.

**Important:** N5 and N6 restrict the peer universe to the evaluation sample
(firms with valid Gemini summaries) to ensure all four models are compared on
an identical candidate pool. See NOTEBOOKS.md for details.

---

## Models

| Model | Method | Feature space | Tests |
|-------|--------|--------------|-------|
| M0 | FF49 industry membership (baseline) | — | — |
| M1 | Financial ratio kNN (cosine) | 64-dim after correlation pruning | H1 |
| M2 | FinBERT text embedding kNN (cosine) | 768-dim | H2 |
| M3 | Late fusion of M1 + M2 (alpha=0.3 weighted ranks) | — | H3, H4 |

Primary metric: **MdAPE on ln(EV/Sales)** (k=10, bootstrapped 95% CI)
Secondary metrics: ln(EV/Assets), ln(Market Cap/SEQ)

---

## Repo Structure

```
thesis/
├── config.py                   ← single source of truth (parameters, paths, schema)
├── requirements.txt
├── README.md
├── NOTEBOOKS.md                ← full notebook reference + execution guide
│
├── data/
│   ├── raw/
│   │   ├── financial/          ← Compustat CSV (not tracked by git)
│   │   └── textual/
│   │       ├── Raw_Text/       ← scraped 10-K text per year
│   │       └── gemini_ready/   ← intermediate cleaned text
│   ├── processed/
│   │   └── Business_Summaries/ ← Gemini 400-word summaries per year
│   └── results/                ← peer lists + evaluation outputs
│
├── notebooks/
│   ├── N0_eda_complete.ipynb
│   ├── N1_data_prep.ipynb
│   ├── N2_financials.ipynb
│   ├── N3_text_pipeline.ipynb
│   ├── N4_multiples.ipynb
│   ├── N5_m0_baseline.ipynb
│   ├── N6_knn_financial.ipynb
│   ├── N7_embeddings.ipynb
│   ├── N7b_ablation.ipynb          ← embedding model ablation (supplementary)
│   ├── N8_knn_text.ipynb
│   ├── N9_fusion.ipynb
│   ├── N10_evaluation.ipynb
│   └── N11_h4.ipynb                ← H4 disclosure-regime analysis
│
├── src/
│   ├── evaluation.py           ← MdAPE, MAPE, bootstrap CI
│   ├── similarity.py           ← cosine kNN, peer coherence diagnostics
│   └── fusion.py               ← weighted rank fusion + RRF
│
└── figures/                    ← all thesis figures (auto-generated)
```

---

## Key Design Decisions

- **Peer schema** identical across M0–M3: `focal_tic | focal_fyear | peer_tic | rank | similarity_score | model` → N10 evaluation loop is model-agnostic
- **Uniform peer universe** — M0, M1, M2, and M3 all draw peers exclusively from the 13,559-firm evaluation sample. This ensures performance differences reflect similarity criterion quality rather than candidate pool composition.
- **Multiples (N4) built separately** from peer inputs — no leakage possible; only joined in N10
- **Feature selection in N2** — 91 candidate ratios → 64 after Spearman correlation pruning (|r| > 0.90); list persisted to `selected_features.json`
- **M1 crosses FF49 boundaries 64.6% of the time** — financial ratios identify economically similar firms beyond industry classification
- **M2 crosses FF49 boundaries 44.3% of the time** — text embeddings on the restricted evaluation sample show more within-industry clustering than the full universe
- **Alpha optimised on 2020–2022 only** — 2023–2024 never touched during tuning; best_alpha=0.3 (70% financial, 30% text)
- **Evaluation sample restricted to 13,559 firm-years** — firms with valid Gemini summary across all five years; all four models compared on identical set in N10
- **FF49 excluded from M1 feature vector** — clean attribution of M1 vs M0 performance
- **Winsorisation** at (1%, 99%) per fyear cross-section: applied to input ratios (N2) and multiples (N4) separately
- **NaN imputation** with 0 post z-normalisation — equals cross-sectional mean; standard academic treatment (Hou, Xue & Zhang 2020; Geertsema & Lu 2023)
- **FF49 Industry 36 (Computer Software)** explicitly included — missing from earlier implementations causes all software firms to misclassify as Business Services
- **Fiscal year heterogeneity** — December year-ends dominant; fyear cross-sections used as-is following Bhojraj & Lee (2002); December-only robustness check in N10 confirms rankings hold
- **peers_m3_rrf.parquet** saved alongside peers_m3.parquet — RRF robustness check in N10

---

## Sample Output (as of current run)

| Step | Firm-years | Firms |
|------|-----------|-------|
| Raw Compustat | 169,724 | 21,674 |
| After USD filter | 141,563 | 18,111 |
| After deduplication | 127,948 | 18,111 |
| After penny/micro-cap filter | 84,513 | 12,681 |
| After quality filter | 54,682 | 8,063 |
| After year filter (2020–2024) | 20,883 | 5,699 |
| Evaluation sample (valid Gemini summary) | 13,559 | 3,494 |

Financial features: 91 candidates → 64 selected after correlation pruning
M1 same-industry rate: 35.4% (crosses FF49 boundaries 64.6% of the time)
M2 same-industry rate: 55.7% (crosses FF49 boundaries 44.3% of the time)
Optimal fusion alpha (M3): 0.3 — 70% financial, 30% text

## Main Results (k=10, ln(EV/Sales), bootstrapped 95% CI, n=13,559)

| Model | MdAPE | 95% CI |
|-------|-------|--------|
| M0 FF49 baseline | 54.79% | [53.50%, 56.20%] |
| M1 Financial kNN | 43.75% | [42.57%, 44.57%] |
| M2 Text kNN (FinBERT) | 51.89% | [50.86%, 53.08%] |
| M3 Late Fusion | **41.13%** | [40.22%, 41.99%] |

H1 (M1 vs M0): Δ=+20.2%, p<0.0001 *** **SUPPORTED**
H2 (M2 vs M0): Δ=+5.3%,  p<0.0001 *** **SUPPORTED**
H3 (M3 vs M1): Δ=+6.0%,  p<0.0001 *** **SUPPORTED**
Total improvement M0→M3: +24.9%

### k-Sensitivity (ln(EV/Sales), MdAPE %)

| Model | k=5 | k=10 | k=15 | k=20 |
|-------|-----|------|------|------|
| M0 FF49 Baseline | 54.79 | 54.79 | 54.79 | 54.79 |
| M1 Financial kNN | 44.62 | 43.75 | 43.87 | 42.96 |
| M2 Text kNN      | 53.76 | 51.89 | 51.65 | 51.36 |
| M3 Late Fusion   | **43.52** | **41.13** | **41.60** | **41.25** |

M3 strictly dominates M1 at every k. Fusion gap (M1−M3) peaks at k=10 (2.6pp),
narrowing at k=5 (1.1pp) and k=20 (0.5pp).

---

## Pipeline Status

| Notebook | Status | Output |
|----------|--------|--------|
| N0 EDA | ✅ Complete | all EDA figures |
| N1 data prep | ✅ Complete | panel_clean.parquet |
| N2 financials | ✅ Complete | financials_normalized.parquet |
| N3 text pipeline | ✅ Complete | business_summaries_{year}.csv |
| N4 multiples | ✅ Complete | multiples.parquet |
| N5 M0 baseline | ✅ Complete | peers_m0.parquet (eval sample) |
| N6 kNN financial | ✅ Complete | peers_m1.parquet (eval sample) |
| N7 embeddings | ✅ Complete | text_embeddings.parquet |
| N7b ablation | ✅ Complete | ablation comparison figures (supplementary) |
| N8 kNN text | ✅ Complete | peers_m2.parquet |
| N9 fusion | ✅ Complete | peers_m3.parquet, peers_m3_rrf.parquet, alpha_optimal.json |
| N10 evaluation | ✅ Complete | results_main.csv + all thesis figures + n10_robustness_checks.csv |
| N11 H4 | ✅ Complete | h4_final/ CSVs, h4_tables.tex, h4_*.pdf, h4_firm_level_analysis.csv |

---

## Data Sources

| Dataset | Source | Location |
|---------|--------|----------|
| Financial panel | Compustat via WRDS | `data/raw/financial/perfect_compustat_pull_v2.csv` |
| 10-K raw text | SEC EDGAR via edgartools | `data/raw/textual/Raw_Text/sec_text_{year}.csv` |
| Business summaries | Gemini 2.5 Flash (400-word condensation) | `data/processed/Business_Summaries/business_summaries_{year}.csv` |

---

## Citation

> [Authors], "[Title]", CBS Master's Thesis, 2025.
> Core framework: Bhojraj & Lee (2002), Hoberg & Phillips (2016), Geertsema & Lu (2023)
