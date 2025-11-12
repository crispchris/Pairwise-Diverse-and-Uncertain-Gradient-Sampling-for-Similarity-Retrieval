# retrieval_metrics.py

import numpy as np
import torch
import random
from tqdm.auto import tqdm
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score


# -------------------------------------------------------------------------
# Core utility functions
# -------------------------------------------------------------------------

def compare_events(x_base, y_base, x_query, y_query):
    """
    Compute Hungarian alignment cost averaged over all timesteps.

    Args:
        x_base, y_base: (n_entities, T_base) base event coordinates
        x_query, y_query: (n_entities, T_query) query event coordinates
    Returns:
        float: average minimal assignment cost per timestep
    """
    num_frames = min(x_base.shape[1], x_query.shape[1])
    total_cost = 0.0

    for t in range(num_frames):
        # Stack x/y positions into (n_entities, 2)
        pos1 = np.column_stack((x_base[:, t], y_base[:, t]))
        pos2 = np.column_stack((x_query[:, t], y_query[:, t]))

        # Hungarian algorithm (minimum-cost bipartite matching)
        cost_matrix = cdist(pos1, pos2)
        rows, cols = linear_sum_assignment(cost_matrix)
        total_cost += cost_matrix[rows, cols].sum()

    return total_cost / num_frames if num_frames > 0 else 0.0


def recall_at_k(true_rel, ranked, k):
    """Recall@k = correct retrieved / total relevant"""
    retrieved_k = ranked[:k]
    hits = len(set(retrieved_k) & set(true_rel))
    return hits / len(true_rel) if true_rel else 0.0


def precision_at_k(true_rel, ranked, k):
    """Precision@k = correct retrieved / k"""
    retrieved_k = ranked[:k]
    hits = len(set(retrieved_k) & set(true_rel))
    return hits / k if k > 0 else 0.0


def topk_recall_and_precision(true_rel, ranked, Ks=(1, 5, 10)):
    """Return dict of Recall@K and Precision@K for each K."""
    metrics = {}
    for k in Ks:
        metrics[f"Recall-at-{k}"] = recall_at_k(true_rel, ranked, k)
        metrics[f"Precision-at-{k}"] = precision_at_k(true_rel, ranked, k)
    return metrics


def average_precision(true_rel, ranked, k):
    """Compute AP@k (average precision up to rank k)."""
    hits, sum_prec = 0, 0.0
    for i, idx in enumerate(ranked[:k], start=1):
        if idx in true_rel:
            hits += 1
            sum_prec += hits / i
    return sum_prec / len(true_rel) if true_rel else 0.0


# -------------------------------------------------------------------------
# Data reconstruction
# -------------------------------------------------------------------------

def reconstruct_xy(vec, sport):
    """
    Reconstruct per-player x/y matrices from flattened vector.

    Args:
        vec (ndarray): flattened trajectory [2*n_players*T]
        sport (str): 'football' or 'basketball'
    Returns:
        (x_matrix, y_matrix)
    """
    if sport == "football":
        n_players, T = 23, 50
    else:
        n_players, T = 11, 150

    mat = vec.reshape(2 * n_players, T)
    x_mat, y_mat = mat[:n_players], mat[n_players:]
    return x_mat, y_mat


# -------------------------------------------------------------------------
# Retrieval evaluation
# -------------------------------------------------------------------------

def evaluate_retrieval(model, dataset, device, sport="football",
                       Ks=(1, 5, 10), relevance_top_m=5,
                       max_queries=50, max_gallery=200,
                       msrcc_topN=50, seed=42):
    """
    Evaluate retrieval metrics (Recall, Precision, mAP, nDCG, MSRCC).

    For each query trajectory:
      - Compute embedding distances to all others
      - Compute ground-truth alignment cost via Hungarian matching
      - Measure correlation and ranking quality

    Returns:
        dict[str, float]: averaged metrics across queries
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=128,
                                         shuffle=False, drop_last=True)

    all_embs, all_vecs = [], []
    model.eval()

    # ---- build embeddings ----
    with torch.no_grad():
        for (x1, x2), _ in tqdm(loader, desc="Building embeddings"):
            for x in [x1, x2]:
                x = x.to(device)
                emb, _ = model(x, x)  # trick: forward twice, keep first output
                all_embs.append(emb.cpu().numpy())
                all_vecs.append(x.cpu().numpy())

    all_embs = np.concatenate(all_embs)
    all_vecs = np.concatenate(all_vecs)
    N = all_embs.shape[0]

    # ---- subsample queries ----
    rng = random.Random(seed)
    query_indices = rng.sample(range(N), min(max_queries, N))

    # Initialize metric containers
    metrics = {f"{m}-at-{k}": [] for m in ["Recall", "Precision", "mAP", "nDCG"] for k in Ks}
    metrics["MSRCC"] = []

    # ---- main query loop ----
    for q_idx in tqdm(query_indices, desc="Queries"):
        q_emb = all_embs[q_idx:q_idx + 1]

        # Candidate selection
        candidates = list(range(N))
        candidates.remove(q_idx)
        cand_subset = rng.sample(candidates, min(max_gallery, len(candidates)))
        cand_embs, cand_vecs = all_embs[cand_subset], all_vecs[cand_subset]

        # Predicted distances in embedding space
        d_pred = np.linalg.norm(cand_embs - q_emb, axis=1)

        # Ground-truth Hungarian alignment costs
        true_costs = []
        for loc, g_idx in enumerate(cand_subset):
            xq, yq = reconstruct_xy(all_vecs[q_idx], sport)
            xg, yg = reconstruct_xy(cand_vecs[loc], sport)
            cost = compare_events(xq, yq, xg, yg)
            true_costs.append((g_idx, cost))

        # Sort by cost (lower = more similar)
        true_costs.sort(key=lambda x: x[1])
        true_relevant = [idx for idx, _ in true_costs[:relevance_top_m]]

        # ---- retrieval metrics ----
        ranked_local = np.argsort(d_pred)
        ranked_idx = [cand_subset[i] for i in ranked_local]

        for k in Ks:
            rec = recall_at_k(true_relevant, ranked_idx, k)
            prec = precision_at_k(true_relevant, ranked_idx, k)
            ap = average_precision(true_relevant, ranked_idx, k)

            # Build relevance vector for nDCG
            rel = np.zeros(len(cand_subset))
            for idx in true_relevant:
                if idx in cand_subset:
                    rel[cand_subset.index(idx)] = 1
            ndcg = ndcg_score([rel], [-d_pred])

            # Store
            metrics[f"Recall-at-{k}"].append(rec)
            metrics[f"Precision-at-{k}"].append(prec)
            metrics[f"mAP-at-{k}"].append(ap)
            metrics[f"nDCG-at-{k}"].append(ndcg)

        # ---- MSRCC (Spearman rank correlation between GT & predicted distances) ----
        nn_gt = [idx for idx, _ in true_costs[:msrcc_topN]]
        gt_d = [d for _, d in true_costs[:msrcc_topN]]
        pred_d = [d_pred[cand_subset.index(idx)] for idx in nn_gt]
        if len(gt_d) > 1:
            r, _ = spearmanr(gt_d, pred_d)
            metrics["MSRCC"].append(r)

    # ---- aggregate over queries ----
    return {m: float(np.nanmean(vals)) for m, vals in metrics.items()}
