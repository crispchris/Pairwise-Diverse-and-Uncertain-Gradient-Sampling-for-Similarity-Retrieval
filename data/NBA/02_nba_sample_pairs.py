"""
Sample pairs of NBA SportVU events and compute Hungarian alignment costs.
- Loads pre-built train/val/test tensors (see preprocessing script).
- Randomly samples pairs of events for each split.
- Computes average Hungarian assignment cost across timesteps.
- Saves results as .npy arrays with shape (n_pairs, 3): [i, j, cost].
"""

import os, random, torch, numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

import multiprocessing as mp
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)


# -----------------------------
# Hungarian cost functions
# -----------------------------
def hungarian_cost(ev1, ev2):
    """
    Hungarian alignment cost between two events.
    Each event is (2,11,150): 11 entities, 2D trajectory over 150 timesteps.
    """
    n_entities = ev1.shape[1]
    T = min(ev1.shape[2], ev2.shape[2])  # usually 150
    cost_matrix = np.zeros((n_entities, n_entities), dtype=np.float32)

    # Build cost matrix: average L2 distance over time
    for i in range(n_entities):
        traj1 = ev1[:, i, :T].T  # (T,2)
        for j in range(n_entities):
            traj2 = ev2[:, j, :T].T  # (T,2)
            dists = np.linalg.norm(traj1 - traj2, axis=1)  # (T,)
            cost_matrix[i, j] = dists.mean()

    # Hungarian assignment
    r, c = linear_sum_assignment(cost_matrix)
    return cost_matrix[r, c].sum()

# -----------------------------
# Parallel sampling
# -----------------------------
def worker(dataset_np, pairs):
    """Compute costs for a chunk of (i,j) pairs."""
    out = []
    for (i, j) in pairs:
        cost = hungarian_cost(dataset_np[i], dataset_np[j])
        out.append((i, j, cost))
    return out

def sample_pairs(dataset, n_pairs, n_workers=6, seed=42, n_chunks=10):
    rng = random.Random(seed)
    N = len(dataset)

    pairs = [(rng.randrange(N), rng.randrange(N)) for _ in range(n_pairs)]
    uniq = sorted(set(i for ij in pairs for i in ij))
    idx_map = {old: new for new, old in enumerate(uniq)}

    subset = dataset[uniq]           # still a torch.Tensor
    dataset_np = subset.numpy()      # only for Hungarian cost

    remapped_pairs = [(idx_map[i], idx_map[j]) for (i, j) in pairs]

    chunk_size = (n_pairs + n_chunks - 1) // n_chunks
    chunks = [remapped_pairs[k*chunk_size:(k+1)*chunk_size]
              for k in range(n_chunks) if remapped_pairs[k*chunk_size:(k+1)*chunk_size]]

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        futs = [exe.submit(worker, dataset_np, ch) for ch in chunks]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Computing pairs"):
            results.extend(fut.result())

    mapped_back = [(uniq[i], uniq[j], cost) for (i, j, cost) in results]

    # Convert directly into a torch tensor
    return torch.tensor(mapped_back, dtype=torch.float32)


if __name__ == "__main__":
    splits = {
        "train": (10_000_000, "nba_sportvu_train.pt"),
        "val":   (500_000,   "nba_sportvu_val.pt"),
        "test":  (500_000,   "nba_sportvu_test.pt"),
    }

    for name, (n_pairs, fpath) in splits.items():
        if not os.path.exists(fpath):
            print(f"Missing dataset file: {fpath}, skipping {name}")
            continue

        print(f"\n=== Processing {name.upper()} ({n_pairs} pairs) ===")
        dataset = torch.load(fpath, map_location="cpu")
        print(f"Loaded {name} dataset: {dataset.shape}")

        pairs_and_costs = sample_pairs(dataset, n_pairs, n_workers=6)
        out_file = f"pairs_{name}_10M.pt"
        torch.save(pairs_and_costs, out_file)

        print(f"Saved {name} pairs -> {out_file}, shape={pairs_and_costs.shape}")
