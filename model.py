import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from scipy.stats import spearmanr


def mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-8):
    """
    Compute Mean Absolute Percentage Error (MAPE) between predictions and targets.

    Args:
        y_true (Tensor): Ground truth values.
        y_pred (Tensor): Predicted values.
        epsilon (float): Small value to avoid division by zero.

    Returns:
        float: MAPE in percentage.
    """
    # Normalize both tensors by the maximum target value for scale invariance
    max_y_true = torch.max(y_true)
    y_true_norm = torch.clamp(y_true / max_y_true, min=epsilon)
    y_pred_norm = y_pred / max_y_true

    # Compute average absolute percentage difference
    mape = torch.mean(torch.abs((y_true_norm - y_pred_norm) / y_true_norm)) * 100
    return mape.item()


class EmbeddingLoss(nn.Module):
    """
    Custom loss for Siamese embedding learning.

    L = (||f(x1) - f(x2)||_2 - d(x1,x2))^2
        + λ * (||f(x1)||_2 + ||f(x2)||_2 + ||θ||_2)

    Encourages embeddings to preserve the target distance while remaining compact
    and regularized by model parameters.
    """
    def __init__(self, parameters, lambda_reg=0.1):
        super().__init__()
        self.params = list(parameters)
        self.lambda_reg = lambda_reg

    def forward(self, out1, out2, targets):
        # Predicted distance between embeddings
        d_pred = F.pairwise_distance(out1, out2)
        mse_term = F.mse_loss(d_pred, targets)

        # Embedding norm regularization (keep embeddings bounded)
        reg_embed = out1.norm(p=2, dim=1).mean() + out2.norm(p=2, dim=1).mean()

        # Weight regularization (L2 over all parameters)
        reg_weights = torch.sum(torch.stack([p.norm(2) for p in self.params]))

        # Total loss
        return mse_term + self.lambda_reg * (reg_embed + reg_weights)


class SiameseNetwork(nn.Module):
    """
    Simple fully-connected Siamese network for trajectory embeddings.

    Args:
        sport (str): 'basketball' or 'football' to set input dimension.
        with_dropout (bool): Whether to include dropout layers.
        embedding_dim (int): Size of the final embedding vector.
        dropout_rate (float): Dropout probability.
    """
    def __init__(self, sport="basketball", with_dropout=False,
                 embedding_dim=64, dropout_rate=0.3):
        super().__init__()

        # Input dimension depends on the sport
        self.input_dim = 3300 if sport == "basketball" else 2300
        self.flatten = nn.Flatten()

        # Build MLP encoder
        layers = [
            nn.Linear(self.input_dim, 256), nn.ReLU(),
            nn.Dropout(dropout_rate) if with_dropout else nn.Identity(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(dropout_rate) if with_dropout else nn.Identity(),
            nn.Linear(128, embedding_dim), nn.ReLU(),
            nn.Dropout(dropout_rate) if with_dropout else nn.Identity()
        ]
        self.fc = nn.Sequential(*layers)

        # Initialize custom loss
        self.criterion = EmbeddingLoss(self.parameters())

    def forward(self, x1, x2):
        """
        Forward pass for a pair of inputs.

        Returns:
            (Tensor, Tensor): Embeddings for x1 and x2.
        """
        x1 = self.flatten(x1)
        x2 = self.flatten(x2)
        out1 = self.fc(x1)
        out2 = self.fc(x2)
        return out1, out2
