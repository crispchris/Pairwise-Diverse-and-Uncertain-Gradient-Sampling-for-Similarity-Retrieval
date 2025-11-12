# gradient_metrics.py
import torch

def compute_per_example_gradients(model, loss_fn, out1, out2, targets):
    """
    Compute per-example gradients wrt model parameters (flattened).
    Returns: grads [N, P] where N=batch size, P=flattened parameter dim.
    """
    batch_size = targets.size(0)
    grads = []

    for i in range(batch_size):
        loss_i = model.criterion(out1[i:i+1], out2[i:i+1], targets[i:i+1])
        grad_params = torch.autograd.grad(
            loss_i, model.parameters(), retain_graph=True, create_graph=False
        )
        flat_grad = torch.cat([g.view(-1) for g in grad_params])
        grads.append(flat_grad)
    return torch.stack(grads)  # [N, P]

def gradient_metrics(per_example_grads):
    """
    Compute gradient diagnostics on GPU when available.
    per_example_grads: [N, P] torch.Tensor, already on device.
    Returns: dict of Python scalars.
    """
    device = per_example_grads.device

    norms = torch.norm(per_example_grads, dim=1)              # [N], stays on GPU
    mean_norm = norms.mean()
    cv_norm = norms.std() / (mean_norm + 1e-8)

    # cosine similarity via Gram matrix (O(N^2 * P))
    X_norm = torch.nn.functional.normalize(per_example_grads, p=2, dim=1)
    sim_matrix = X_norm @ X_norm.T                            # [N, N] on GPU

    # upper triangular (cosine distances)
    tri_idx = torch.triu_indices(len(norms), len(norms), offset=1, device=device)
    upper_tri = sim_matrix[tri_idx[0], tri_idx[1]]
    grad_diversity = (1 - upper_tri).mean()

    # SVD for rank (O(N*P^2) or O(P*N^2))
    u, s, v = torch.svd(per_example_grads)  # runs on GPU if input is there
    rank = (s > 1e-6).sum()

    snr = mean_norm / (norms.std() + 1e-8)

    # Move only final scalars back to CPU
    return {
        "grad_mean_norm": mean_norm.item(),
        "grad_cv": cv_norm.item(),
        "grad_diversity": grad_diversity.item(),
        "grad_rank": rank.item(),
        "grad_snr": snr.item(),
    }
