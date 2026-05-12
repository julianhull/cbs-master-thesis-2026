# Thesis Repo — Cleanup & GitHub Status Summary

**Date written:** 2026-05-12  
**Repo:** `https://github.com/julianhull/cbs-master-thesis-2026.git`  
**Branch:** `main`  
**Last commit:** `6ab8e2a` — N11 H4

---

## What this document is

A record of every notebook cleanup decision made across two Claude Code sessions
(~2026-05-11 to 2026-05-12). Use it to verify the GitHub repo is in the expected
state when you return.

---

## Pipeline overview

```
N0  EDA
N1  Data prep & feature engineering        → panel_clean.parquet
N2  Financial normalization & feature sel  → finn.parquet / finn_rob.parquet
N3  Text pipeline (scrape → Gemini summ.)  → business_summaries_{year}.csv
N4  Multiples construction                 → multiples.parquet
N5  FF49 baseline peer lists (M0)          → peers_m0.parquet
N6  kNN financial peer lists (M1)          → peers_m1.parquet
N7  FinBERT embeddings                     → text_embeddings.parquet
N7b Embedding model ablation              (supplementary, no downstream dep)
N8  kNN text peer lists (M2)              → peers_m2.parquet
N9  Late fusion (M3) + alpha sweep        → peers_m3.parquet
N10 Full evaluation (M0–M3) + robustness  → data/results/results_main.csv etc.
N11 H4 disclosure-regime analysis         → data/results/h4_final/
```

Key config constants (from `config.py`):

| Constant | Value |
|---|---|
| `YEARS` | [2020, 2021, 2022, 2023, 2024] |
| `K_MAIN` | 10 |
| `K_ROBUSTNESS` | [5, 10, 15, 20] |
| `BOOTSTRAP_ITERS` | 1000 |
| `RANDOM_SEED` | 42 |
| `EMBEDDING_MODEL` | ProsusAI/finbert |
| `ALPHA_GRID` | [0.0, 0.1, …, 1.0] |
| `VALIDATION_YEARS` | [2020, 2021, 2022] |
| `TEST_YEARS` | [2023, 2024] |

---

## Notebook status after cleanup

All notebooks listed below are in `notebooks/` on `main`. All outputs cleared
(execution_count = null, outputs = []) except **N1** (see note).

| Notebook | Cells | Commit | What changed |
|---|---|---|---|
| N0_eda_complete | 24 | `8d5ee31`–`b817dea` | Removed pip/subprocess, shrunk headers, fixed aliases, cleared outputs |
| N1_data_prep | 25 | `047dd6c` | Deduplicated validation cell, fixed misleading FF49 print, bumped faiss-cpu — **⚠ still has 19 output cells** (see note below) |
| N2_financials | 28 | `f788a86`–`ee8d175` | Removed Colab setup cell, unified `CORR_THRESHOLD` → config constant, stripped `Cell N` numbering |
| N3_text_pipeline | 17 | `6eec245` | Removed hardcoded API key, dropped test cell, standard header |
| N4_multiples | 20 | `3b9fe6a` | Standard header, stripped `Cell N` numbering, fixed Next label |
| N5_m0_baseline | 19 | `f965ff1` | Standard header, removed redundant mkdir |
| N6_knn_financial | 19 | `dcc7bc0` | Removed pip install, dropped docstring, stripped `Cell N` numbering |
| N7_embeddings | 12 | `7f33672` | Standard header, removed pip/subprocess/mkdir, deleted 2 dev cells |
| N7b_ablation | 22 | `fd5a976` + `fedf514` | Full rewrite (standard header, fix scatter title, simplify `build_peers`); **lazy import fix** for `sentence_transformers` — import moved inside `embed_st()` so Cell 1 runs without the library installed |
| N8_knn_text | 11 | `c264a35` + `6f6c074` | Standard header, removed pip/subprocess/dup import; source format normalised (strings → arrays) |
| N9_fusion | 23 | `6f6c074` | Standard header, fixed OUTPUTS/mkdir/docstrings/dev-comments, added missing section headers, stripped `Cell N` prefixes, removed duplicate `import matplotlib.gridspec`, added `### 4b · Per-Year Alpha Sensitivity` and `### 6 · Early Fusion Ablation` headers |
| N10_evaluation | 79 | `3a96d35` | Standard header, fixed all markdown/section headers (cascade-renumbered sections 7→12), deleted 14 dev/scratch cells, **integrated N13 robustness checks as Sections 9.3–9.6**, clear outputs |
| N11_h4 | 35 | `6ab8e2a` | Created from three `_to_review` sources (see below); standard header, Phases A/B/C + alpha sensitivity extension + firm-level section, clear outputs |

### N1 output note

N1 was the very first notebook touched and still has `outputs:19`. It's safe to re-run it and clear outputs before final submission — the logic is correct, outputs just weren't cleared in that early commit.

---

## Cleaning standards applied (universal)

Every notebook was brought to the same standard:

- **Header cell:** `#### N{x} — Title` (not `#` or `##`)
- **Sections:** `###` for main sections, `####` for subsections (never `##`)
- **Setup cell:** Standard robust repo-root pattern — no Colab, no subprocess, no pip install:
  ```python
  import sys
  from pathlib import Path
  notebook_dir = Path('__file__').parent if '__file__' in dir() else Path.cwd()
  repo_root = next(
      (p for p in [notebook_dir, *notebook_dir.parents] if (p / 'config.py').exists()), None
  )
  if repo_root is None:
      raise FileNotFoundError('config.py not found — check repo structure')
  sys.path.insert(0, str(repo_root))
  from config import *
  ```
- **I/O declaration cell:** Every notebook has an `INPUTS = [...]` / `OUTPUTS = [...]` cell immediately after setup
- **No `# Cell N —` prefixes** on code cells
- **No multi-line docstrings** on functions
- **No redundant `mkdir`** calls — `config.py` creates all directories at import
- **No hardcoded paths, API keys, or dev breadcrumbs**
- **All outputs cleared** before commit
- **`config.py` is the single source of truth** for all paths and constants

---

## What was merged from `_to_review/`

### N13 → N10 (Sections 9.3–9.6)

`_to_review/N13_discussion_robustness_part (1).ipynb` contained four robustness sections.
All four were adapted and inserted into N10 as Sections 9.3–9.6:

| Section | Content |
|---|---|
| 9.3 · Constrained Peer Selection | Within-industry kNN using `build_constrained_peers_vectorised` (numpy matrix ops) |
| 9.4 · M0 Variants at k=10 | Different FF49 aggregation levels |
| 9.5 · Alpha\* Stability Across k | Fusion weight sensitivity at k∈{5,10,15,20} |
| 9.6 · Financial Sector Exclusion | Robustness to dropping SIC 6000–6999 |

Variable names were adapted for N10's namespace (`panel_focal` → `p_focal_rob`,
`eval_index` → `eval_idx_rob = summary_tickers`, etc.). A bridge cell loads the
correct data objects.

**N13 itself is NOT in `notebooks/` — its content is fully absorbed into N10.**

### N11_h4_with_alpha + H4_firm_level → N11_h4

`N11_h4_with_alpha.ipynb` (28 cells) was the primary source. Key content:

- **Cells 0–11:** Phases A, B, C (R&D disclosure regime stratification)
- **Cells 12–19:** Alpha sensitivity by stratum (the "with_alpha" addition)

`H4_firm_level (1).ipynb` (9 cells) contributed **Section 8** (firm-level Δ analysis).

**Dropped from `_with_alpha`:**
- Cell 6: CSV read-back debug cell
- Cell 20: Hardcoded bootstrap bar chart (stale numbers)
- Cells 21–22: Null-tic debug + sector count dev cells
- Cells 23–27: Per-sector alpha extension (too granular for thesis)

**`N11_h4.ipynb` in `_to_review/` is superseded — the canonical version is `notebooks/N11_h4.ipynb`.**

---

## N11 structure (35 cells)

```
#### N11 — H4: Disclosure-Regime Heterogeneity...
### 0 · Setup
### 1 · Load Data
### 2 · R&D Disclosure Stratification
### 3 · Phase A — Main Results by Stratum
### 4 · Phase B — Peer-List Overlap Mechanism (Jaccard)
### 5 · Phase C — Robustness
    #### 5.1 · Year-by-Year Stability
    #### 5.2 · Multi-Multiple Consistency
    #### 5.3 · Size-Tercile Robustness
### 6 · Synthesis Figure (4-panel)
### 7 · Alpha Sensitivity by Stratum
    [alpha intro, helpers, pooled sweep, per-(stratum,year) sweep,
     bootstrap α*, figure, interpretation]
### 8 · Firm-Level Fusion Benefit
### 9 · Save & LaTeX Tables
```

Key findings baked in:
- H4 partially supported (3/4 criteria): A1 ✓, A2 ✓, B ✓, C1 ✗ (3/5 years for Δ M2)
- R&D-Zero firms have ~2× higher Jaccard(M1,M2) = consensus-reinforcement mechanism
- Alpha* shifts: R&D-Active α* ≠ global 0.3 (shifts toward more text weight in some years)
- Firm-level Δ: out-of-sample R² ≈ 0 (fusion benefit is diffuse, not firm-sortable)

---

## N10 structure (79 cells)

```
#### N10 — Full Evaluation (M0–M3)
### 0 · Setup
### 1 · Load Data
### 2 · Evaluation Functions
### 3 · Main Results Grid
    #### 3.1 · Main Results Table (k=10)
    #### 3.2 · Fusion Method Robustness — Weighted Rank vs RRF
### 4 · Hypothesis Tests (H1, H2, H3)
    #### 4.1 · H1 — Financial kNN Beats FF49
    #### 4.2 · H2 — Text kNN Beats FF49
    #### 4.3 · H3 — Fusion Beats Best Single Modality
### 5 · Annual Breakdown
    #### 5.1 · Year-by-Year MdAPE
    #### 5.2 · Industry-Level Breakdown
    #### 5.3 · Peer Coherence Diagnostics
    #### 5.4 · Failure Analysis
### 6 · Economic Significance
### 7 · Peer Set Complementarity — Jaccard Overlap
### 8 · FF49 Same-Industry Hit Rate
### 9 · Robustness Checks
    #### 9.1 · December FYE Subsample
    #### 9.2 · Validation vs Test Year Split
    #### 9.3 · Constrained Peer Selection        ← from N13
    #### 9.4 · M0 Variants at k=10               ← from N13
    #### 9.5 · Alpha* Stability Across k          ← from N13
    #### 9.6 · Financial Sector Exclusion         ← from N13
### 10 · Case Studies
    #### 10.1 · Implied Valuations and APE
    #### 10.2 · t-SNE Visualisation of Feature Spaces
### 11 · Narrative Diffusion Analysis
### 12 · Save
```

---

## `_to_review/` — current state

All decisions resolved. Files remaining:

| File | Status |
|---|---|
| `N11_h4.ipynb` | **Superseded** — canonical version is `notebooks/N11_h4.ipynb` |
| `N11_h4_with_alpha.ipynb` | **Superseded** — all useful content absorbed into `notebooks/N11_h4.ipynb` |
| `H4_firm_level (1).ipynb` | **Absorbed** — Section 8 of N11 is derived from this |
| `N13_discussion_robustness_part (1).ipynb` | **Absorbed** — Sections 9.3–9.6 of N10 |
| `README.md` | Stale (pre-cleanup decisions) — can be deleted or left as-is |

**None of the `_to_review/` files need to be in `notebooks/`. They are source archives.**

---

## Specific fixes applied (beyond standard cleaning)

| Notebook | Issue | Fix |
|---|---|---|
| N7b | `ModuleNotFoundError: No module named 'sentence_transformers'` on Cell 1 | Moved `from sentence_transformers import SentenceTransformer` inside `embed_st()` body (lazy import) |
| N9 | Duplicate `import matplotlib.gridspec as gridspec` | Removed duplicate |
| N9 | Missing section headers for Per-Year Alpha Sensitivity and Early Fusion Ablation | Added `### 4b ·` and `### 6 ·` markdown cells |
| N10 | 14 dev/scratch/debug cells (Colab lookups, testing zones, re-rank filters) | Deleted all 14 |
| N10 | Sections mis-numbered after N13 merge + Economic Significance insertion | Cascade-renumbered sections 7→12 |
| N11 | Colab setup (drive mount, subprocess, pip installs) | Full header replacement with standard pattern |
| N11 | Dev cells (CSV read-back, null-tic debug, sector count) | Dropped |

---

## Commit log (newest first)

```
6ab8e2a  Clean N11: standard header, phases A/B/C + alpha sensitivity extension + firm-level section
3a96d35  Clean N10: standard header, fix all markdown/section headers, delete 14 dev cells, integrate N13
fedf514  Fix N7b: lazy-import SentenceTransformer inside embed_st()
6f6c074  Clean N9: standard header, fix OUTPUTS/mkdir/docstrings; normalize N8 source; delete N7b Colab file
c264a35  Clean N8: standard header, remove pip/subprocess/mkdir/dup import
fd5a976  Clean N7b: standard header, fix OUTPUTS/scatter title, simplify build_peers
7f33672  Clean N7: standard header, remove pip/subprocess/mkdir, delete dev cells
b817dea  Clean N0: clear outputs
cbd8072  Clean N0: shrink top title cell to H4
be56b9e  Clean N0: shrink markdown headers, trim bridge section
8d5ee31  Clean N0: remove pip/subprocess, add json import, fix aliases, strip dev comments
dcc7bc0  Clean N6: remove pip install, drop docstring, strip Cell N numbering
f965ff1  Clean N5: standard header, strip Cell N numbering
3b9fe6a  Clean N4: standard header, strip Cell N numbering
6eec245  Clean N3: remove hardcoded API key, standard header
ee8d175  Clean N2: clear outputs
f788a86  Clean N2: remove Colab setup cell, unify CORR_THRESHOLD
047dd6c  Clean N1: dedupe validation cell, fix FF49 print, bump faiss-cpu
d1dd530  Initial commit: thesis repo pre-cleanup baseline
```

---

## What remains before final submission

1. **N1 outputs** — re-run N1 and clear before final push (or just clear the 19 output cells manually)
2. **`_to_review/` cleanup** — optionally delete the now-superseded files from `_to_review/` to keep the repo clean
3. **End-to-end run** — run the full pipeline N1→N11 in order on a clean machine to confirm no broken imports or missing parquet files
4. **README.md at repo root** — check whether a root-level README exists and is up to date (the current one in `_to_review/` is stale)

---

*Generated by Claude Sonnet 4.6 on 2026-05-12*
