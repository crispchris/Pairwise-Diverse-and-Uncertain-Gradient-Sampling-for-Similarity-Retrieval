# cost_fns.py
import numpy as np
from scipy.optimize import linear_sum_assignment

def pairwise_cost_matrix(ev1, ev2, T=None):
    """
    Calculates the pairwise cost matrix between two events.

    Parameters:
    - ev1 (np.ndarray): The first event array of shape (n1, T, d).
    - ev2 (np.ndarray): The second event array of shape (n2, T, d).
    - T (int, optional): The number of time steps to consider. If None, defaults to the minimum dimension of either event array.

    Returns:
    - np.ndarray: A matrix of pairwise costs between all pairs of events.

    Note:
    - The cost is calculated as the Euclidean distance between corresponding time steps of the two events.
    - The resulting matrix has dimensions (n1, n2).
    - If T is None, it defaults to the minimum dimension of either event array.
    """
    T = min(ev1.shape[2], ev2.shape[2]) if T is None else T
    X = ev1[:, :, :T].transpose(1, 2, 0)  # (n1, T, d)
    Y = ev2[:, :, :T].transpose(1, 2, 0)  # (n2, T, d)
    diff = X[:, None, :, :] - Y[None, :, :, :]
    return np.linalg.norm(diff, axis=-1).mean(axis=-1)

def hungarian_cost(ev1, ev2):
    """
    Hungarian algorithm to find the minimum cost assignment between two sets of elements.

    The Hungarian algorithm is used to find a minimum-cost perfect matching in bipartite graphs.
    It works by solving the problem of finding the maximum weighted independent set in a graph,
    which corresponds to the minimum weighted dependent set in the bipartite graph's complement.

    Args:
        ev1 (Sequence[Any]): The first set of elements represented as a sequence of items.
        ev2 (Sequence[Any]): The second set of elements represented as a sequence of items.

    Returns:
        int: The total cost of the minimum cost assignment.

    Raises:
        ValueError: If the input sequences are not of equal length or contain duplicate items.
    """
    C = pairwise_cost_matrix(ev1, ev2)
    r, c = linear_sum_assignment(C)
    return C[r, c].sum()

def keypoint_hungarian_cost(ev1, ev2, n_keypoints=30):
    """
    Compute the Hungarian cost between two energy fields for keypoint matching.

    Parameters:
    - ev1: np.ndarray of shape (N1, N2, T) or (T, N1, N2). The first dimension represents the number of time points, the second and third dimensions represent the spatial dimensions, and T is the total number of frames. This should contain feature values at different times for each pixel.
    - ev2: np.ndarray of shape (N1, N2, T) or (T, N1, N2). Similar to `ev1`, but contains feature values at different times for each pixel.
    - n_keypoints: int (default=30). The number of keypoints used in the matching. It is ignored if either dimension of `ev1` or `ev2` has less than this number of frames.

    Returns:
    - cost: np.ndarray of shape (N1, N2) representing the Hungarian cost between the two energy fields.
    """
    T = min(ev1.shape[2], ev2.shape[2])
    idx = np.linspace(0, T - 1, n_keypoints, dtype=int)
    ev1_sub = ev1[:, :, idx]
    ev2_sub = ev2[:, :, idx]
    return hungarian_cost(ev1_sub, ev2_sub)
