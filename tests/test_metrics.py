import torch
import pytest
from utils.metrics import DiceScore

def test_dice_score_perfect_match():
    metric = DiceScore()
    
    # Create fake logits (high positive for 1s, high negative for 0s)
    targets = torch.tensor([1, 0, 1, 0]).float()
    logits = torch.tensor([10.0, -10.0, 10.0, -10.0]) # Sigmoid(10) ~= 1
    
    score = metric(logits, targets)
    assert score > 0.99

def test_dice_score_no_overlap():
    metric = DiceScore()
    
    targets = torch.tensor([1, 1, 1]).float()
    logits = torch.tensor([-10.0, -10.0, -10.0]) # Sigmoid(-10) ~= 0
    
    score = metric(logits, targets)
    # Epsilon handles division by zero, intersection is 0
    assert score < 0.01