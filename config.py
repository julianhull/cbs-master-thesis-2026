# =============================================================================
# config.py — Single source of truth for the entire thesis pipeline
# Peer Identification via Financial Ratios, LLM-Condensed Text & Late Fusion
# =============================================================================

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent
DATA_RAW     = ROOT / "data/raw"
DATA_PROC    = ROOT / "data/processed"
DATA_RESULTS = ROOT / "data/results"
FIGURES      = ROOT / "figures"

for p in [DATA_RAW, DATA_PROC, DATA_RESULTS, FIGURES]:
    p.mkdir(parents=True, exist_ok=True)

# ── Raw file names ─────────────────────────────────────────────────────────────
COMPUSTAT_FILE = DATA_RAW / "financial/perfect_compustat_pull_v2.csv"

# ── Textual data paths ────────────────────────────────────────────────────────
SEC_TEXT_FILES   = {yr: DATA_RAW / f"textual/Raw_Text/sec_text_{yr}.csv" for yr in range(2020, 2025)}
GEMINI_READY_DIR = DATA_RAW / "textual/gemini_ready"
SUMMARIES_DIR    = DATA_PROC / "Business_Summaries"
SUMMARIES_FILES  = {yr: SUMMARIES_DIR / f"business_summaries_{yr}.csv" for yr in range(2020, 2025)}

# ── Processed artefacts (auto-generated — never edit manually) ────────────────
PANEL_CLEAN      = DATA_PROC / "panel_clean.parquet"
FINANCIALS_NORM  = DATA_PROC / "financials_normalized.parquet"
MULTIPLES        = DATA_PROC / "multiples.parquet"
EMBEDDINGS       = DATA_PROC / "text_embeddings.parquet"
PEERS_M0         = DATA_RESULTS / "peers_m0.parquet"
PEERS_M1         = DATA_RESULTS / "peers_m1.parquet"
PEERS_M2         = DATA_RESULTS / "peers_m2.parquet"
PEERS_M3         = DATA_RESULTS / "peers_m3.parquet"
PEERS_M3_RRF     = DATA_RESULTS / "peers_m3_rrf.parquet"
RESULTS_MAIN     = DATA_RESULTS / "results_main.csv"
ALPHA_OPTIMAL    = DATA_RESULTS / "alpha_optimal.json"

# ── Sample ────────────────────────────────────────────────────────────────────
YEARS            = [2020, 2021, 2022, 2023, 2024]
VALIDATION_YEARS = [2020, 2021, 2022]
TEST_YEARS       = [2023, 2024]

# ── Financial filters (following Geertsema & Lu 2023) ────────────────────────
MIN_PRICE  = 1.0
MIN_MKTCAP = 50.0

# ── Financial feature vector (M1) ─────────────────────────────────────────────
FINANCIAL_FEATURES     = []   # populated by N2 at runtime
SELECTED_FEATURES_FILE = DATA_RESULTS / "selected_features.json"
CORRELATION_THRESHOLD  = 0.90
MISSING_THRESHOLD      = 0.80

# ── Multiples (validation targets — kept strictly separate from peer inputs) ──
PRIMARY_MULTIPLE    = "ev_sales"
SECONDARY_MULTIPLES = ["ev_assets", "market_cap_seq"]
WINSOR_BOUNDS       = (0.01, 0.99)

# ── kNN models ────────────────────────────────────────────────────────────────
K_MAIN            = 10
K_ROBUSTNESS      = [5, 10, 15, 20]
SIMILARITY_METRIC = "cosine"

# ── Late fusion (M3) ──────────────────────────────────────────────────────────
# best_alpha = 0.3  (70% financial / 30% text)
# Validation MdAPE at α=0.3: 40.67% (ln_v2s, k=10, n=7,944 val firm-years)
# RRF baseline at equal weights: 41.75% — WR with tuned α wins by 1.08 pp
ALPHA_GRID     = [round(x * 0.1, 1) for x in range(11)]
FUSION_METHOD  = "weighted_rank"
RRF_K_CONSTANT = 60

# ── Text embedding (M2) ───────────────────────────────────────────────────────
EMBEDDING_MODEL   = "ProsusAI/finbert"
EMBEDDING_DIM     = 768
EMBEDDING_POOLING = "mean"          # mean pooling over non-padding tokens (NOT CLS)
SUMMARY_COL       = "business_description"
TIC_COL           = "tic"
FYEAR_COL         = "fyear"

# ── Evaluation ────────────────────────────────────────────────────────────────
BOOTSTRAP_ITERS = 1000
CI_LEVEL        = 0.95
RANDOM_SEED     = 42

# ── Peer file schema ──────────────────────────────────────────────────────────
PEER_SCHEMA = ["focal_tic", "focal_fyear", "peer_tic", "rank", "similarity_score", "model"]

# ── Plotting ──────────────────────────────────────────────────────────────────
FIGURE_DPI    = 300
FIGURE_FORMAT = "pdf"
MODEL_COLORS  = {
    "M0_FF49":      "#4C72B0",
    "M1_Financial": "#DD8452",
    "M2_Text":      "#55A868",
    "M3_Fusion":    "#C44E52",
}
MODEL_LABELS  = {
    "M0_FF49":      "M0: FF49 Baseline",
    "M1_Financial": "M1: Financial kNN",
    "M2_Text":      "M2: Text kNN (FinBERT)",
    "M3_Fusion":    "M3: Late Fusion",
}
