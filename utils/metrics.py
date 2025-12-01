import torch
import torch.nn as nn

class DiceScore(nn.Module):
    def __init__(self, threshold: float = 0.5, epsilon: float = 1e-6):
        super().__init__()
        self.threshold = threshold
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw output from model (before Sigmoid)
            targets: Binary ground truth masks
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        
        # Flatten label and prediction tensors
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        # Binarize predictions
        preds = (probs > self.threshold).float()
        
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()
        
        dice = (2. * intersection + self.epsilon) / (union + self.epsilon)
        
        return dice