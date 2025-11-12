# active_selection.py
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from cost_fns import keypoint_hungarian_cost

# -------------------------------------------------------------------------
# Active learning selection strategies
# -------------------------------------------------------------------------

def active_select_random(batch, model=None, device=None, final_k=128, **_):
    """Uniformly sample `final_k` examples from a batch."""
    (x1, x2), y = batch
    if len(y) <= final_k:
        return (x1, x2), y
    idx = torch.randperm(len(y))[:final_k]
    return (x1[idx], x2[idx]), y[idx]


def active_select_pairdug_gt(batch, model, device, final_k=128):
    """
    PairDUG selection using ground-truth distances.
    Clusters approximate gradient embeddings for diversity.
    """
    (x1, x2), y = batch
    x1, x2, y = x1.to(device), x2.to(device), y.to(device)
    model.eval()

    emb1, emb2 = model(x1, x2)
    dist = torch.nn.functional.pairwise_distance(emb1, emb2)
    delta = 2 * (dist.detach() - y.detach())  # grad wrt distance

    # Approximate per-sample gradient embedding
    grads = torch.cat([emb1, emb2], dim=1) * delta.unsqueeze(1)
    grads = grads.detach().cpu().numpy()

    # Cluster for diversity
    n_clusters = min(final_k, len(grads))
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++",
                    n_init=1, random_state=42).fit(grads)
    chosen, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, grads)
    return (x1.cpu()[chosen], x2.cpu()[chosen]), y.cpu()[chosen]


def active_select_pairdug_fast(batch, model, device, final_k=128, sport="football"):
    """
    Faster PairDUG variant:
    recomputes approximate Hungarian costs and uses pseudo-gradients.
    """
    (x1, x2), _ = batch
    x1, x2 = x1.to(device), x2.to(device)
    model.eval()

    # --- approximate costs ---
    x1_np, x2_np = x1.cpu().numpy(), x2.cpu().numpy()
    costs = torch.tensor([keypoint_hungarian_cost(a, b, n_keypoints=20)
                          for a, b in zip(x1_np, x2_np)], dtype=torch.float32)

    emb1, emb2 = model(x1, x2)
    dist = torch.nn.functional.pairwise_distance(emb1, emb2)
    delta = 2 * (dist.detach() - costs.to(device))

    grads = torch.cat([emb1, emb2], dim=1) * delta.unsqueeze(1)
    grads = grads.detach().cpu().numpy()

    n_clusters = min(final_k, len(grads))
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++",
                    n_init=1, random_state=42).fit(grads)
    chosen, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, grads)
    (_, _), dists = batch
    return (x1.cpu()[chosen], x2.cpu()[chosen]), dists.cpu()[chosen]


# -------------------------------------------------------------------------
# Dispatcher
# -------------------------------------------------------------------------

STRATEGY_MAP = {
    "random": active_select_random,
    "pairdug_gt": active_select_pairdug_gt,
    "pairdug_fast": active_select_pairdug_fast,
    "full": lambda batch, *args, **kwargs: batch
}

def get_active_selection_fn(name):
    """Return the active selection function given a strategy name."""
    if name not in STRATEGY_MAP:
        raise ValueError(f"Unknown active selection strategy: {name}")
    return STRATEGY_MAP[name]
