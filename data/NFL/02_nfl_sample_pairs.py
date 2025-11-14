from scipy.optimize import linear_sum_assignment
import torch, os, numpy as np, random
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# -----------------------------
# Hungarian cost functions
# -----------------------------
def hungarian_cost(ev1, ev2):
    """Each event is (2,23,60)."""
    n_entities, T = ev1.shape[1], min(ev1.shape[2], ev2.shape[2])
    cost_matrix = np.zeros((n_entities, n_entities), dtype=np.float32)
    for i in range(n_entities):
        traj1 = ev1[:,i,:T].T
        for j in range(n_entities):
            traj2 = ev2[:,j,:T].T
            cost_matrix[i,j] = np.linalg.norm(traj1-traj2,axis=1).mean()
    r,c = linear_sum_assignment(cost_matrix)
    return cost_matrix[r,c].sum()


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

if __name__=="__main__":
    splits = {
        "train": (100_000,"nfl_ngs_train.pt"),
        "val":   (100_000, "nfl_ngs_val.pt"),
        "test":  (100_000, "nfl_ngs_test.pt"),
    }
    for name,(n_pairs,fpath) in splits.items():
        if not os.path.exists(fpath): continue
        dataset = torch.load(fpath,map_location="cpu")
        pairs = sample_pairs(dataset,n_pairs,n_workers=6)
        torch.save(pairs,f"pairs_nfl_{name}_100k.pt")
