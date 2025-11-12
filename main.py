import os, random, numpy as np, torch, mlflow, hydra
from tqdm.auto import tqdm
from omegaconf import DictConfig
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from model import SiameseNetwork, mean_absolute_percentage_error
from data_module import get_dataloaders
from retrieval_metrics import evaluate_retrieval
from gradient_metrics import compute_per_example_gradients, gradient_metrics
from active_selection import get_active_selection_fn


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------

def set_seed(seed: int):
    """Ensure reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------------------------------------------------------
# Active learning selection strategies
# -------------------------------------------------------------------------

def active_select_random(batch, final_k=128):
    """Uniformly sample `final_k` examples from a batch."""
    (x1, x2), y = batch
    if len(y) <= final_k: return (x1, x2), y
    idx = torch.randperm(len(y))[:final_k]
    return (x1[idx], x2[idx]), y[idx]


def active_select_pairdug_gt(batch, model, device, final_k=128):
    """
    PairDUG selection
    """
    (x1, x2), y = batch
    x1, x2, y = x1.to(device), x2.to(device), y.to(device)
    model.eval()

    emb1, emb2 = model(x1, x2)
    dist = torch.nn.functional.pairwise_distance(emb1, emb2)
    delta = 2 * (dist.detach() - y.detach())  # grad wrt distance

    # Approximate gradient embedding per sample
    grads = torch.cat([emb1, emb2], dim=1) * delta.unsqueeze(1)
    grads = grads.detach().cpu().numpy()

    # Cluster gradient embeddings for diversity
    n_clusters = min(final_k, len(grads))
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++",
                    n_init=1, random_state=42).fit(grads)
    chosen, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, grads)
    return (x1.cpu()[chosen], x2.cpu()[chosen]), y.cpu()[chosen]


def active_select_pairdug_fast(batch, model, device, final_k=128):
    """
    Faster PairDUG variant: recompute approximate Hungarian costs
    and derive pseudo-gradients for selection.
    """
    (x1, x2), _ = batch
    x1, x2 = x1.to(device), x2.to(device)
    model.eval()

    # --- recompute approximate costs ---
    def recompute_batch_cost(x1, x2, sport="football"):
        from cost_fns import keypoint_hungarian_cost
        x1, x2 = x1.cpu().numpy(), x2.cpu().numpy()
        return torch.tensor([keypoint_hungarian_cost(a, b, n_keypoints=20)
                             for a, b in zip(x1, x2)], dtype=torch.float32)

    costs = recompute_batch_cost(x1, x2)
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
# Training and evaluation loop
# -------------------------------------------------------------------------
def run_epoch(dataloader, model, optimizer, device,
              mode="train", active_selection_strategy="random",
              tau=None, cfg={}, global_step=0):
    """
    Run one epoch of training/validation/testing.

    Handles active data selection, gradient logging, and optional retrieval eval.
    """
    running = {"loss": [], "mape": []}
    model.train() if mode == "train" else model.eval()

    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader),
                                 desc=f"{mode.capitalize()} Epoch", leave=False):
        if batch_idx == len(dataloader) - 5: continue  # skip tail to avoid small batches

        # --- Active learning subsampling ---
        if mode == "train":
            select_fn = get_active_selection_fn(cfg.active_strategy)
            batch = select_fn(batch, model, device, cfg.eval_batch_size)

        (x1, x2), y = batch
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)

        with torch.set_grad_enabled(mode == "train"):
            out1, out2 = model(x1, x2)
            loss = model.criterion(out1, out2, y)
            dist = torch.nn.functional.pairwise_distance(out1, out2)
            mape = mean_absolute_percentage_error(y, dist)

            # --- Gradient metrics logging ---
            if mode == "train" and cfg.gradient_metrics:
                per_ex_grads = compute_per_example_gradients(model, model.criterion, out1, out2, y)
                grad_stats = gradient_metrics(per_ex_grads)
                for k, v in grad_stats.items():
                    mlflow.log_metric(f"grad/{k}", v, step=global_step)

            # --- Optimization ---
            if mode == "train":
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                global_step += 1

        running["loss"].append(loss.item()); running["mape"].append(mape)

    # --- Retrieval eval on test mode ---
    if mode == "test":
        from data_module import PreprocessedDataset
        test_dataset = PreprocessedDataset(cfg.test_pairs, cfg.test_events)
        results = evaluate_retrieval(model, test_dataset, device, sport=cfg.sport,
                                     max_queries=cfg.max_queries, max_gallery=cfg.max_gallery)
        for k, v in results.items():
            mlflow.log_metric(f"retrieval/{k}", v)
            print(f"{k}: {v:.4f}")

    return {k: np.mean(v) for k, v in running.items()}


# -------------------------------------------------------------------------
# Experiment entry point
# -------------------------------------------------------------------------

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main experiment orchestration with MLflow logging and early stopping."""
    print("⚙️ Running experiment with config:\n", cfg)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    mlflow.set_experiment("PairDUG")

    # --- Dataset path resolution ---
    sport_cfg = {"football": cfg.football, "basketball": cfg.basketball}.get(cfg.sport)
    if sport_cfg is None:
        raise ValueError(f"Unsupported sport '{cfg.sport}'")

    for split in ["train", "val", "test"]:
        setattr(cfg, f"{split}_pairs", os.path.abspath(getattr(sport_cfg, f"{split}_pairs")))
        setattr(cfg, f"{split}_events", os.path.abspath(getattr(sport_cfg, f"{split}_events")))

    with mlflow.start_run(run_name=f"{cfg.active_strategy}_{cfg.sport}"):
        mlflow.log_params({
            "sport": cfg.sport, "active_strategy": cfg.active_strategy,
            "seed": cfg.seed, "batch_size": cfg.batch_size,
            "eval_batch_size": cfg.eval_batch_size, "epochs": cfg.epochs,
            "lr": cfg.lr, "weight_decay": cfg.weight_decay,
            "dropout_rate": cfg.model.dropout_rate,
            "embedding_dim": cfg.model.embedding_dim,
            "device": device.type
        })

        # --- Data and model ---
        train_dl, val_dl, test_dl = get_dataloaders(
            cfg.train_pairs, cfg.val_pairs, cfg.test_pairs,
            cfg.train_events, cfg.val_events, cfg.test_events,
            batch_size=cfg.batch_size
        )

        model = SiameseNetwork(
            sport=cfg.sport,
            with_dropout=(cfg.active_strategy in ["dropout", "entropy"]),
            embedding_dim=cfg.model.embedding_dim,
            dropout_rate=cfg.model.dropout_rate,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        tau = 316.71 if cfg.sport == "basketball" else 414.25

        # --- Early stopping setup ---
        es = cfg.early_stopping
        best_val = float("inf") if es.mode == "min" else -float("inf")
        patience_counter, best_state = 0, None
        global_step = 0
        # --- Training loop ---
        for epoch in range(cfg.epochs):
            train_m = run_epoch(train_dl, model, optimizer, device, "train",
                                cfg.active_strategy, tau, cfg, global_step)
            val_m = run_epoch(val_dl, model, optimizer, device, "val", cfg=cfg, global_step=global_step)

            # Early stopping
            if es.enabled:
                val_metric = val_m[es.monitor]
                improved = (val_metric < best_val) if es.mode == "min" else (val_metric > best_val)
                if improved:
                    best_val, patience_counter, best_state = val_metric, 0, model.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= es.patience:
                        print(f"⏹️ Early stopping at epoch {epoch+1} on {es.monitor}")
                        mlflow.log_metric("best_epoch", epoch+1)
                        break

            mlflow.log_metrics({f"train/{k}": v for k, v in train_m.items()}, step=epoch)
            mlflow.log_metrics({f"val/{k}": v for k, v in val_m.items()}, step=epoch)
            print(f"[Epoch {epoch+1}/{cfg.epochs}] Train loss={train_m['loss']:.4f} | "
                  f"Val loss={val_m['loss']:.4f}")

        # --- Final test evaluation ---
        test_m = run_epoch(test_dl, model, optimizer, device, "test", cfg=cfg, global_step=global_step)
        mlflow.log_metrics({f"test/{k}": v for k, v in test_m.items()}, step=epoch)
        print(f"[Epoch {epoch+1}/{cfg.epochs}] Test loss={test_m['loss']:.4f}")


if __name__ == "__main__":
    main()
