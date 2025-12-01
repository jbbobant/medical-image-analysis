import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import Optional

def plot_prediction(
    ct_slice: torch.Tensor, 
    mask_slice: torch.Tensor, 
    pred_slice: torch.Tensor, 
    threshold: float = 0.5
) -> plt.Figure:
    """
    Creates a matplotlib figure comparing input, ground truth, and prediction.
    Expects 2D tensors (H, W).
    """
    # Convert to numpy and detatch from graph
    ct = ct_slice.cpu().detach().numpy()
    mask = mask_slice.cpu().detach().numpy()
    pred = torch.sigmoid(pred_slice).cpu().detach().numpy()
    pred_binary = (pred > threshold).astype(np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: CT with Ground Truth
    axes[0].imshow(ct, cmap="bone")
    masked_gt = np.ma.masked_where(mask == 0, mask)
    axes[0].imshow(masked_gt, alpha=0.5, cmap="autumn")
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    # Plot 2: CT with Prediction
    axes[1].imshow(ct, cmap="bone")
    masked_pred = np.ma.masked_where(pred_binary == 0, pred_binary)
    axes[1].imshow(masked_pred, alpha=0.5, cmap="autumn")
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    # Plot 3: Raw Heatmap
    im = axes[2].imshow(pred, cmap="jet", vmin=0, vmax=1)
    axes[2].set_title("Prediction Heatmap")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    return fig