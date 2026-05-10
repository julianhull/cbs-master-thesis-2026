# src/evaluation.py
# MdAPE, MAPE, bootstrap CI — identical logic applied to all four models

import numpy as np
import pandas as pd
from typing import Tuple

def compute_mdape(actual: np.ndarray, implied: np.ndarray) -> float:
    """Median Absolute Percentage Error — primary metric."""
    return float(np.median(np.abs((actual - implied) / actual)))

def compute_mape(actual: np.ndarray, implied: np.ndarray) -> float:
    """Mean Absolute Percentage Error — secondary diagnostic."""
    return float(np.mean(np.abs((actual - implied) / actual)))

def bootstrap_ci(actual: np.ndarray, implied: np.ndarray,
                 n_iter: int = 1000, ci: float = 0.95,
                 seed: int = 42) -> Tuple[float, float]:
    """Bootstrap confidence interval around MdAPE."""
    rng = np.random.default_rng(seed)
    boot = []
    n = len(actual)
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        boot.append(compute_mdape(actual[idx], implied[idx]))
    lo = (1 - ci) / 2
    return float(np.quantile(boot, lo)), float(np.quantile(boot, 1 - lo))

def evaluate_model(peers_df: pd.DataFrame, multiples_df: pd.DataFrame,
                   multiple_col: str = "ln_ev_sales",
                   value_driver: str = "sales",
                   k: int = 10) -> pd.DataFrame:
    
    top_k = peers_df[peers_df["rank"] <= k].copy()

    top_k = top_k.merge(
        multiples_df[["tic", "fyear", multiple_col]].rename(
            columns={"tic": "peer_tic", "fyear": "focal_fyear",
                     multiple_col: "peer_multiple"}
        ),
        on=["peer_tic", "focal_fyear"], how="left"
    )

    peer_medians = (
        top_k.groupby(["focal_tic", "focal_fyear"])["peer_multiple"]
        .median().reset_index()
        .rename(columns={"peer_multiple": "peer_median_multiple"})
    )

    result = peer_medians.merge(
        multiples_df[["tic", "fyear", multiple_col,
                       value_driver, "ev_actual"]].rename(
            columns={"tic": "focal_tic", "fyear": "focal_fyear"}
        ),
        on=["focal_tic", "focal_fyear"], how="inner"
    )

    # Exponentiate back from log space before multiplying by value driver
    result["implied_ev"] = (
        np.exp(result["peer_median_multiple"]) * result[value_driver]
    )
    result["abs_pct_error"] = np.abs(
        (result["ev_actual"] - result["implied_ev"]) / result["ev_actual"]
    )
    return result

def summarise_results(eval_df: pd.DataFrame, model_name: str,
                      n_iter: int = 1000) -> dict:
    """Return MdAPE, MAPE and 95% CI for a model."""
    actual  = eval_df["ev_actual"].values
    implied = eval_df["implied_ev"].values
    lo, hi  = bootstrap_ci(actual, implied, n_iter=n_iter)
    return {
        "model":  model_name,
        "n":      len(eval_df),
        "MdAPE":  compute_mdape(actual, implied),
        "MAPE":   compute_mape(actual, implied),
        "CI_lo":  lo,
        "CI_hi":  hi,
    }
