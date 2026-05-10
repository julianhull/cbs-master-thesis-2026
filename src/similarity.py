# src/similarity.py
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def build_peer_list(feature_matrix, tickers, fyear,
                    k_max=20, model=""):
    sim = cosine_similarity(feature_matrix)
    np.fill_diagonal(sim, -1)

    rows = []
    for i, focal in enumerate(tickers):
        ranked_idx = np.argsort(sim[i])[::-1][:k_max]
        for rank, j in enumerate(ranked_idx, start=1):
            rows.append({
                "focal_tic":        focal,
                "focal_fyear":      fyear,
                "peer_tic":         tickers[j],
                "rank":             rank,
                "similarity_score": float(sim[i, j]),
                "model":            model,
            })
    return pd.DataFrame(rows), sim


def peer_focal_similarity(peers_df: pd.DataFrame) -> pd.Series:
    """Average peer-to-focal cosine similarity per focal firm (Eq. S_focal)."""
    return peers_df.groupby(["focal_tic", "focal_fyear"])["similarity_score"].mean()


def peer_coherence(peers_df, sim_matrix,   # ← accept precomputed
                   tickers, k=10):
    tic_idx = {t: i for i, t in enumerate(tickers)}
    sim = cosine_similarity(feature_matrix)
    rows = []
    for focal, grp in peers_df[peers_df["rank"] <= k].groupby(
            ["focal_tic", "focal_fyear"]):
        peers = grp["peer_tic"].tolist()
        scores = []
        for a in range(len(peers)):
            for b in range(a + 1, len(peers)):
                ia, ib = tic_idx.get(peers[a]), tic_idx.get(peers[b])
                if ia is not None and ib is not None:
                    scores.append(sim[ia, ib])
        rows.append({
            "focal_tic":   focal[0],
            "focal_fyear": focal[1],
            "coherence":   float(np.mean(scores)) if scores else np.nan,
        })
    return pd.DataFrame(rows)


# src/fusion.py
def weighted_rank_fusion(peers_m1: pd.DataFrame, peers_m2: pd.DataFrame,
                          alpha: float, k_max: int = 20) -> pd.DataFrame:
    """
    Late fusion: score = alpha * text_sim + (1-alpha) * financial_sim
    Normalise ranks to [0,1] so scores are comparable.
    """
    m1 = peers_m1[["focal_tic", "focal_fyear", "peer_tic", "rank"]].copy()
    m2 = peers_m2[["focal_tic", "focal_fyear", "peer_tic", "rank"]].copy()

    # Rank score = 1 - (rank-1)/k_max  →  rank 1 = 1.0, rank k_max = ~0
    m1["fin_score"]  = 1 - (m1["rank"] - 1) / k_max
    m2["text_score"] = 1 - (m2["rank"] - 1) / k_max

    merged = m1.merge(m2, on=["focal_tic", "focal_fyear", "peer_tic"], how="outer")
    merged["fin_score"]  = merged["fin_score"].fillna(0)
    merged["text_score"] = merged["text_score"].fillna(0)
    merged["fusion_score"] = alpha * merged["text_score"] + (1 - alpha) * merged["fin_score"]

    merged = merged.sort_values(
        ["focal_tic", "focal_fyear", "fusion_score"], ascending=[True, True, False]
    )
    merged["rank"] = merged.groupby(["focal_tic", "focal_fyear"]).cumcount() + 1
    merged = merged[merged["rank"] <= k_max].copy()
    merged["similarity_score"] = merged["fusion_score"]
    merged["model"] = f"M3_Fusion_alpha{alpha}"
    return merged[["focal_tic", "focal_fyear", "peer_tic", "rank", "similarity_score", "model"]]


def reciprocal_rank_fusion(peers_m1: pd.DataFrame, peers_m2: pd.DataFrame,
                            k_const: int = 60, k_max: int = 20) -> pd.DataFrame:
    """Alternative fusion: RRF score = 1/(k+rank_fin) + 1/(k+rank_text)."""
    m1 = peers_m1[["focal_tic", "focal_fyear", "peer_tic", "rank"]].rename(columns={"rank": "rank_fin"})
    m2 = peers_m2[["focal_tic", "focal_fyear", "peer_tic", "rank"]].rename(columns={"rank": "rank_text"})

    merged = m1.merge(m2, on=["focal_tic", "focal_fyear", "peer_tic"], how="outer")
    merged["rank_fin"]  = merged["rank_fin"].fillna(k_max + 1)
    merged["rank_text"] = merged["rank_text"].fillna(k_max + 1)
    merged["rrf_score"] = (1 / (k_const + merged["rank_fin"]) +
                           1 / (k_const + merged["rank_text"]))

    merged = merged.sort_values(
        ["focal_tic", "focal_fyear", "rrf_score"], ascending=[True, True, False]
    )
    merged["rank"] = merged.groupby(["focal_tic", "focal_fyear"]).cumcount() + 1
    merged = merged[merged["rank"] <= k_max].copy()
    merged["similarity_score"] = merged["rrf_score"]
    merged["model"] = "M3_Fusion_RRF"
    return merged[["focal_tic", "focal_fyear", "peer_tic", "rank", "similarity_score", "model"]]
