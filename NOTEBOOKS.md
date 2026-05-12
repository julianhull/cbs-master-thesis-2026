# Thesis Notebook Reference
# Peer Identification via Financial Ratios, LLM-Condensed Text & Late Fusion

================================================================================
UNIVERSAL HEADER — paste as first 3 cells in EVERY notebook
================================================================================

# Cell 1 — imports & config
import sys
from pathlib import Path

notebook_dir = Path('__file__').parent if '__file__' in dir() else Path.cwd()
repo_root = next(
    (p for p in [notebook_dir, *notebook_dir.parents] if (p / 'config.py').exists()),
    None
)
if repo_root is None:
    raise FileNotFoundError('config.py not found — check repo structure')
sys.path.insert(0, str(repo_root))

from config import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
print('Config loaded.')

# Cell 2 — declare I/O (makes dependencies explicit)
# INPUTS  = [...]
# OUTPUTS = [...]

# Cell 3 — notebook purpose
# PURPOSE : ...
# RUNTIME : ...
# DEPENDS : ...

================================================================================
EXECUTION ORDER & DEPENDENCIES
================================================================================

  N1_data_prep          ~5 min    panel_clean.parquet                ✅
  N2_financials         ~5 min    financials_normalized.parquet      ✅
                                  selected_features.json
  N3_text_pipeline      ~2h       business_summaries_{year}.csv      ✅
  N4_multiples          ~2 min    multiples.parquet                  ✅
  N5_m0_baseline        ~2 min    peers_m0.parquet                   ✅  (eval sample restricted)
  N6_knn_financial      ~5 min    peers_m1.parquet                   ✅  (eval sample restricted)
  N7_embeddings         ~45 min   text_embeddings.parquet            ✅
  N8_knn_text           ~15 min   peers_m2.parquet                   ✅
  N9_fusion             ~5 min    peers_m3.parquet                   ✅
                                  peers_m3_rrf.parquet
                                  alpha_optimal.json
  N10_evaluation        ~20 min   results_main.csv + all figures     ✅
  N0_eda_complete       ~10 min   all EDA figures                    ✅

  Parallelisation:
    N3, N4, N5, N6 can all run simultaneously once N1 + N2 are done
    N7 requires N3 complete
    N8 requires N7 complete
    N9 requires N6 + N8 complete
    N10 requires N4 + N5 + N6 + N8 + N9 complete
    N0 requires N10 complete (uses results_main.csv)

  IMPORTANT: N5 and N6 restrict the peer universe to the evaluation sample
  (firms with valid Gemini summaries). This ensures all four models compete
  on an identical candidate pool. See N5/N6 for implementation.

================================================================================
N1_data_prep.ipynb  ✅ COMPLETE
================================================================================
PURPOSE : Load Compustat panel, apply financial filters, engineer 70+ Geertsema &
          Lu ratios, assign FF49 industry codes, save panel_clean.parquet
RUNTIME : ~5 min
DEPENDS : perfect_compustat_pull_v2.csv
OUTPUTS : data/processed/panel_clean.parquet

KEY STEPS:
1.  Load COMPUSTAT_FILE, lowercase columns
2.  USD filter (curcd == 'USD')
3.  Sort by gvkey/fyear/datadate, drop_duplicates keep='last'
4.  Build base variables: be, ocf, ibadj
5.  Generate lags (l_*) and deltas (d_*) for 26 base variables
6.  Engineer 70+ Geertsema & Lu ratios across 6 groups:
      (1) Profitability & Margins
      (2) Turnover & Efficiency
      (3) Leverage & Capital Structure
      (4) Liquidity & Cash Flow
      (5) Coverage & Rates
      (6) Growth & Deltas
7.  Assign FF49 49-industry codes (full Ken French verbatim mapping)
    NOTE: Industry 36 (Softw) explicitly included — was missing in earlier versions
8.  df.copy() to defragment before aliases
9.  Compute market_cap, ev
10. Filter: prcc_f >= MIN_PRICE, market_cap >= MIN_MKTCAP
11. Filter: ev > 0, seq > 0, at > 0, sale > 0
12. Filter: fyear in YEARS
13. Save PANEL_CLEAN (ln_m2b/ln_v2a/ln_v2s NOT saved here — computed in N4 only)

SAMPLE OUTPUT: 20,883 firm-years | 5,699 firms | 209 columns
ATTRITION:
  Raw                    : 169,724 firm-years
  After USD filter       : 141,563
  After dedup            : 127,948
  After penny/micro-cap  :  84,513
  After quality filter   :  54,682
  After year filter      :  20,883

================================================================================
N2_financials.ipynb  ✅ COMPLETE
================================================================================
PURPOSE : Identify financial ratio columns, correlation pruning, winsorise,
          z-normalise, impute NaN, save financials_normalized.parquet
RUNTIME : ~5 min
DEPENDS : panel_clean.parquet (N1)
OUTPUTS : data/processed/financials_normalized.parquet
          data/results/selected_features.json

KEY STEPS:
1.  Load PANEL_CLEAN
2.  Identify candidate ratio columns (_wr, _an suffix only)
    Exclude: raw items, lags, deltas, identifiers, targets
3.  Drop ratios with >MISSING_THRESHOLD (80%) missing values
4.  Compute Spearman correlation matrix (rank-based, pre-winsorisation)
5.  Greedy correlation pruning at CORRELATION_THRESHOLD (0.90):
    drop feature with higher NaN rate from each correlated pair
6.  Save selected feature list to selected_features.json
7.  Winsorise at WINSOR_BOUNDS per fyear (explicit loop — no groupby apply)
8.  Z-normalise per fyear (explicit loop)
9.  Impute remaining NaN with 0 (= cross-sectional mean post z-score)
    Academic standard: Hou, Xue & Zhang (2020); Geertsema & Lu (2023)
10. Save FINANCIALS_NORM

SAMPLE OUTPUT: 20,883 firm-years | 91 candidates → 64 selected features
  Dropped coverage (>80% NaN)  : 0
  Dropped correlation (|r|>0.9): 27
  Final feature count          : 64

================================================================================
N3_text_pipeline.ipynb  ✅ COMPLETE
================================================================================
PURPOSE : Coverage check, filter to panel firms, scrape missing via EDGAR,
          clean text, generate Gemini 2.5 Flash summaries per firm-year
RUNTIME : ~20 min scraping + ~2h Gemini API (checkpointed throughout)
DEPENDS : sec_text_{year}.csv in data/raw/textual/Raw_Text/
          panel_clean.parquet (N1)
OUTPUTS : data/processed/Business_Summaries/business_summaries_{year}.csv

KEY STEPS:
1.  Load panel_clean, build per-year ticker sets (ground truth)
2.  Coverage check: raw text vs panel per year
    Drop firms not in panel; keep longest text per ticker
3.  EDGAR scraper for missing firm-years (parallel, 8 workers, checkpointed):
    - FILING_WINDOWS: 2020→{2020,2021}, 2021→{2021,2022}, ..., 2024→{2024,2025}
    - Method 1: structured obj() → Item 1
    - Method 2: raw text regex → ITEM 1 BUSINESS
    - Method 3: positional fallback
4.  Merge recovered text, final dedup (longest per ticker)
5.  Gemini 2.5 Flash summarisation (checkpointed every 50 rows per year):
    - 400-word strategic summary per firm-year
    - System prompt: business model, products, market, competition
    - INSUFFICIENT_DATA sentinel for empty filings
6.  Diagnostics: coverage table, missing list, word count plots

INITIAL COVERAGE (before scraping):
  2020: 60.2% | 2021: 60.2% | 2022: 67.6% | 2023: 70.6% | 2024: 70.5%

FINAL COVERAGE (after scraping + summarisation):
  Total: 13,559 valid summaries across 2020–2024 | 3,494 unique firms
  Coverage: 64.9% of 20,883-firm financial panel
  Coverage ranges from 58.9% of panel firms in 2020 to 72.2% in 2024.

  NOTE: Per-year summary counts at this stage include firms later dropped in N7
  (summaries shorter than 50 chars filtered out during embedding). The definitive
  per-year evaluation sample counts are in N7 below.

NOTE: Many missing firms are foreign ADRs filing 20-F (not 10-K) —
      these fail quickly in the scraper and are accepted as missing.

================================================================================
N4_multiples.ipynb  ✅ COMPLETE
================================================================================
PURPOSE : Construct EV/Sales, EV/Assets, MktCap/SEQ validation targets
RUNTIME : ~2 min
DEPENDS : panel_clean.parquet (N1)
OUTPUTS : data/processed/multiples.parquet

KEY STEPS:
1.  Load PANEL_CLEAN (subset of columns)
2.  Recompute market_cap = csho * prcc_f
3.  Recompute ev = market_cap + dltt + dlc - che
4.  Compute raw multiples:
      ev_sales       = ev / sale          → PRIMARY_MULTIPLE (ln_v2s)
      ev_assets      = ev / at            → SECONDARY (ln_v2a)
      market_cap_seq = market_cap / seq   → SECONDARY (ln_m2b)
      ev_ebitda      = ev / ebitda        → robustness only (76.1% coverage)
5.  Winsorise at WINSOR_BOUNDS per fyear (explicit loop)
6.  Log-transform → ln_v2s, ln_v2a, ln_m2b, ln_ev_ebitda
7.  Save MULTIPLES

CRITICAL: MULTIPLES kept strictly separate from peer inputs (N6, N8, N9)
          Only joined in N10 evaluation — no leakage possible.

LONGITUDINAL NOTE: Valuation compression observed 2021→2022 across all
          three multiples (rate hike effect), then stabilisation 2023–2024.

================================================================================
N5_m0_baseline.ipynb  ✅ COMPLETE
================================================================================
PURPOSE : Build M0 FF49 baseline peer lists restricted to evaluation sample
RUNTIME : ~2 min
DEPENDS : panel_clean.parquet (N1), business_summaries_{year}.csv (N3)
OUTPUTS : data/results/peers_m0.parquet

KEY STEPS:
1.  Load PANEL_CLEAN (tic, fyear, ff49_num, industry only)
2.  Build eval_tickers set from SUMMARIES_FILES (valid Gemini summaries only)
3.  Restrict panel to evaluation sample only (13,559 firm-years)
4.  For each fyear, group firms by FF49 industry
5.  Assign all other eval-sample firms in same industry as peers
6.  Equal weight: rank=1, similarity_score=1.0 for all peers
7.  Save PEERS_M0

DESIGN DECISIONS:
  - Peer universe restricted to evaluation sample — identical candidate pool
    across all four models (M0/M1/M2/M3)
  - Same fyear only (consistent with M1/M2/M3)
  - All industry firms = peers (variable k by industry size)
  - Equal weight — no ranking within industry
  - Follows Bhojraj & Lee (2002) exactly

================================================================================
N6_knn_financial.ipynb  ✅ COMPLETE
================================================================================
PURPOSE : Build M1 peer lists via cosine kNN in 64-dim financial ratio space,
          restricted to evaluation sample
RUNTIME : ~5 min
DEPENDS : financials_normalized.parquet (N2), selected_features.json (N2),
          business_summaries_{year}.csv (N3)
OUTPUTS : data/results/peers_m1.parquet

KEY STEPS:
1.  Load FINANCIALS_NORM + selected_features.json
2.  Build eval_tickers set from SUMMARIES_FILES
3.  Restrict df to evaluation sample only
4.  For each fyear:
    a. Subset + dropna on tic
    b. Build feature matrix Z (n_firms × 64)
    c. Compute full cosine similarity matrix
    d. Extract top-k=20 peers per firm (excludes self)
5.  Concat all years → save PEERS_M1

KEY RESULT:
  M1 same-industry rate (k=10): 35.4%
  → M1 crosses FF49 boundaries 64.6% of the time
  → Consistent across all 5 years (34.2%–37.0%)
  Total records: 271,180 (k_max=20 across 5 years)

================================================================================
N7_embeddings.ipynb  ✅ COMPLETE
================================================================================
PURPOSE : Embed Gemini summaries with FinBERT → 768-dim vectors per firm-year
RUNTIME : ~45 min (GPU recommended; CPU ~3-4h)
DEPENDS : business_summaries_{year}.csv (N3)
OUTPUTS : data/processed/text_embeddings.parquet

KEY STEPS:
1.  Load all SUMMARIES_FILES, concat → df [tic, fyear, business_description]
2.  Drop rows where summary is null, invalid, or len < 50 chars
3.  Load FinBERT tokenizer + model (ProsusAI/finbert)
4.  Tokenize max_length=512, truncation=True, padding=True
5.  Forward pass → mean pooling over non-padding tokens (NOT [CLS])
6.  L2-normalise embeddings (required before cosine similarity)
7.  Checkpoint year-by-year → concat → save EMBEDDINGS

GPU TIP: device = "cuda" if torch.cuda.is_available() else "cpu"
         Process in batches of 64

SAMPLE OUTPUT: 13,559 firm-years embedded (evaluation sample only by construction)
  Coverage per year (definitive evaluation sample counts):
    2020: 2,446 | 2021: 2,781 | 2022: 2,717 | 2023: 2,742 | 2024: 2,873
  L2 norm check: mean=1.000000, std=0.000000 ✓
  NaN values: 0 ✓

================================================================================
N8_knn_text.ipynb  ✅ COMPLETE
================================================================================
PURPOSE : Build M2 peer lists via cosine kNN in FinBERT embedding space
RUNTIME : ~15 min
DEPENDS : text_embeddings.parquet (N7)
OUTPUTS : data/results/peers_m2.parquet

KEY STEPS:
1.  Load EMBEDDINGS (already restricted to eval sample by construction)
2.  For each fyear:
    a. Subset to that year
    b. Build embedding matrix E (n_firms × 768) — already L2-normalised
    c. Cosine similarity → top-k=20 peers per firm
3.  Concat all years → save PEERS_M2

KEY RESULT:
  M2 same-industry rate (k=10): 55.7%
  → M2 crosses FF49 boundaries 44.3% of the time
  Avg top-1 similarity: ~0.981 (very high — FinBERT summaries cluster tightly)
  Total records: 271,180 (k_max=20 across 5 years)

  Coverage per year:
    2020: 2,446 firms → 48,920 records
    2021: 2,781 firms → 55,620 records
    2022: 2,717 firms → 54,340 records
    2023: 2,742 firms → 54,840 records
    2024: 2,873 firms → 57,460 records

================================================================================
N9_fusion.ipynb  ✅ COMPLETE
================================================================================
PURPOSE : Optimise alpha on VALIDATION_YEARS, build M3 peers for all years
RUNTIME : ~5 min
DEPENDS : peers_m1.parquet (N6), peers_m2.parquet (N8), multiples.parquet (N4)
OUTPUTS : data/results/peers_m3.parquet
          data/results/peers_m3_rrf.parquet   (RRF robustness)
          data/results/alpha_optimal.json

KEY STEPS:
1.  Load PEERS_M1, PEERS_M2, MULTIPLES
2.  Coverage: M1=13,559 focal firm-years, M2=13,559, overlap=13,559 (100%)
3.  Alpha optimisation on VALIDATION_YEARS = [2020, 2021, 2022] only:
      for alpha in ALPHA_GRID (0.0 to 1.0, step 0.1):
          fused = weighted_rank_fusion(m1, m2, alpha, k=20)
          mdape = evaluate(fused, multiples, k=K_MAIN)
      best_alpha = argmin(mdape)
4.  Save best_alpha to ALPHA_OPTIMAL
5.  Build M3 peers for ALL years using best_alpha:
      Weighted Rank Fusion  → save PEERS_M3         (271,180 records)
      Reciprocal Rank Fusion → save peers_m3_rrf    (271,180 records)
6.  Plot alpha sensitivity curve → figures/n9_alpha_sensitivity.pdf

KEY RESULT:
  best_alpha = 0.3  (val MdAPE = 40.67% on ln_v2s, k=10, n=7,944 val firm-years)
  → α=0.3 means 30% weight on text (M2), 70% on financial (M1)
  → M3 (WR) outperforms M3 (RRF): 40.67% vs 41.75% on validation set

ALPHA GRID (validation set, k=10, ln_v2s):
  α=0.0 (pure M1): 43.23%
  α=0.1:           42.24%
  α=0.2:           41.13%
  α=0.3:           40.67%  ← optimal
  α=0.4:           41.17%
  α=0.5:           41.62%
  α=1.0 (pure M2): 51.08%

CRITICAL: TEST_YEARS = [2023, 2024] never touched during alpha optimisation

================================================================================
N10_evaluation.ipynb  ✅ COMPLETE
================================================================================
PURPOSE : All thesis results — MdAPE tables, CI, figures
RUNTIME : ~20 min
DEPENDS : peers_m0/m1/m2/m3.parquet, peers_m3_rrf.parquet, multiples.parquet (N4),
          panel_clean.parquet (N1), business_summaries_{year}.csv (N3)
OUTPUTS : data/results/results_main.csv + all thesis figures

EVALUATION SAMPLE:
  All models restricted to 13,559 firm-years with valid Gemini summary
  (64.9% of full 20,883-firm panel) — ensures apples-to-apples comparison

STRUCTURE:
  10.1 Main results table — Model × k → MdAPE [95% CI], n
  10.2 Annual breakdown — MdAPE by year per model
  10.3 Fusion robustness — WR vs RRF across k values
  10.4 H1: M1 vs M0  ✅
  10.5 H2: M2 vs M0  ✅
  10.6 H3: M3 vs best single modality  ✅
  10.7 H4: Sector stratification  ✅
  10.8 FF49 same-industry hit rate — all models × all k
  10.9 December fiscal year-end robustness  ✅
  10.10 Case studies  ✅

PRIMARY METRIC: MdAPE on ln_v2s (EV/Sales), k=10, bootstrapped 95% CI

MAIN RESULTS (k=10, evaluation sample n=13,559):

  ln(EV/Sales):
    M0_FF49       54.79%  [53.50%, 56.20%]   n=13,558
    M1_Financial  43.75%  [42.57%, 44.57%]   n=13,559  ← best single
    M2_Text       51.89%  [50.86%, 53.08%]   n=13,559
    M3_Fusion     41.13%  [40.22%, 41.99%]   n=13,559  ← best overall

  ln(EV/Assets):
    M0_FF49       82.31%  [80.89%, 83.53%]
    M1_Financial  68.60%  [66.92%, 70.02%]
    M2_Text       76.53%  [75.47%, 77.81%]
    M3_Fusion     66.29%  [64.90%, 67.52%]

  ln(MktCap/SEQ):
    M0_FF49       60.06%  [59.18%, 61.02%]
    M1_Financial  50.43%  [49.33%, 51.50%]
    M2_Text       55.87%  [54.66%, 57.17%]
    M3_Fusion     49.04%  [48.08%, 50.18%]

HYPOTHESIS RESULTS (Wilcoxon signed-rank, k=10, ln_v2s):
  H1 M1 vs M0:  Δ=+20.2%  p<0.0001 ***  SUPPORTED
  H2 M2 vs M0:  Δ=+5.3%   p<0.0001 ***  SUPPORTED
  H3 M3 vs M1:  Δ=+6.0%   p<0.0001 ***  SUPPORTED
  All hypotheses supported across all three multiples.

H4 SECTOR STRATIFICATION (ln_v2s, k=10):
  Innovation (n≈4,365):
    M0=45.29%  M1=37.81%  M2=43.39%  M3=36.25%
    → M3 best; text adds modest value in innovation sectors
  Traditional (n≈1,708, after excluding 6 industries with <100 firm-years):
    M0=83.36%  M1=74.27%  M2=93.30%  M3=71.54%
    → M2 badly underperforms M0 in traditional sectors
    → M3 recovers via low alpha weight on text (α=0.3)
    → M3 is the best model in both strata

PEER COMPLEMENTARITY (Jaccard overlap M1 vs M2, k=10):
  Full sample:   mean J=0.030, median J=0.000, 63.6% with J=0
  Innovation:    mean J=0.019, median J=0.000, 71.8% with J=0
  Traditional:   mean J=0.042, median J=0.000, 56.9% with J=0
  → M1 and M2 share zero peers in majority of firm-years → strong fusion rationale

K-SENSITIVITY RESULTS (MdAPE %, all models × all k, ln_v2s):
  NOTE: peers_m3.parquet contains ranks 1..20 (271,180 rows); N10 slices
  correctly at each k. A prior stale run erroneously showed M3=M1 at k≠10;
  the corrected figures below confirm strict M3 dominance at every k.

            k=5     k=10    k=15    k=20
  M0_FF49  54.79   54.79   54.79   54.79
  M1       44.62   43.75   43.87   42.96
  M2       53.76   51.89   51.65   51.36
  M3       43.52   41.13   41.60   41.25

  ln(EV/Assets):
            k=5     k=10    k=15    k=20
  M0_FF49  82.31   82.31   82.31   82.31
  M1       69.76   68.60   69.04   68.98
  M2       78.73   76.53   76.76   76.81
  M3       69.20   66.29   66.94   67.19

  ln(MktCap/SEQ):
            k=5     k=10    k=15    k=20
  M0_FF49  60.06   60.06   60.06   60.06
  M1       53.47   50.43   50.35   49.64
  M2       59.48   55.87   55.90   55.76
  M3       52.77   49.04   48.63   48.31

  Fusion gap (M1−M3) on ln_v2s: k=5→1.1pp | k=10→2.6pp | k=15→2.3pp | k=20→0.5pp
  → Gap peaks at k=10; fusion benefit is structural, not a k=10 artefact.

ROBUSTNESS:
  December FYE only (ln_v2s, k=10):
    M0=51.95%  M1=40.40%  M2=48.49%  M3=38.50%  → rankings preserved
  Total improvement M0→M3: +24.9%

================================================================================
N0_eda_complete.ipynb  ✅ COMPLETE
================================================================================
PURPOSE : All EDA and results figures for thesis
RUNTIME : ~10 min
DEPENDS : panel_clean.parquet, multiples.parquet, peers_m0/m1/m2/m3.parquet,
          results_main.csv, business_summaries_{year}.csv (N3)
OUTPUTS : all figures in figures/eda_*.pdf

KEY CHANGES vs earlier version:
  - All peer lists and multiples restricted to evaluation sample (13,559 firm-years)
  - Attrition table updated with evaluation sample row
  - H4 figure updated to show all four models (M0/M1/M2/M3)
  - Text corpus cells (EDA.8–12) require summary files in environment

================================================================================
MODELS SUMMARY
================================================================================

  M0  FF49 industry membership — all same-industry firms, equal weight
      Peer universe: evaluation sample only | variable k by industry size
  M1  Financial ratio kNN — cosine, 64-dim (after N2 pruning)
      Same-industry rate: 35.4% → crosses boundaries 64.6% of time
  M2  FinBERT text embedding kNN — cosine, 768-dim, mean pooling
      Same-industry rate: 55.7% → crosses boundaries 44.3% of time
      High top-1 similarity (~0.981) — summaries cluster tightly
  M3  Late fusion M1 + M2 — alpha-weighted rank fusion
      best_alpha = 0.3  (70% financial, 30% text)
      Optimised on validation years [2020–2022] only

PEER SCHEMA (identical across M0-M3):
  focal_tic | focal_fyear | peer_tic | rank | similarity_score | model

================================================================================
OPEN QUESTIONS / TO DISCUSS
================================================================================

1. CHINESE ADR CLUSTERING (N6 spot check)
   Financial ratios may match firms on characteristics correlated with foreign
   listing structure rather than business similarity. Accept as known limitation
   and flag in thesis — excluded firms would need SIC/exchange filter.

2. H4 INTERPRETATION — TRADITIONAL SECTORS
   M2 (93.30%) badly underperforms M0 (83.36%) in traditional sectors.
   Discuss in thesis as evidence that textual similarity captures strategic
   positioning rather than operational fundamentals — FinBERT is less suited
   to industries where 10-K language is formulaic and undifferentiated.
   M3 recovers by downweighting text (α=0.3), finishing best in both strata.

================================================================================
HOW TO GIVE CLAUDE CONTEXT EFFICIENTLY
================================================================================

  NOTEBOOK : N{x}_{name}
  ISSUE    : [one sentence]
  CELL     : [paste only the failing cell]
  CONFIG   : [paste only the relevant config.py section]
