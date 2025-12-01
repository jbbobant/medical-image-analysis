import torch
import pytorch_lightning as pl
from src.engine.segmentor import TumorSegmentationTask

def test_training_step_flow():
    """Verifies that a forward pass and loss calculation works."""
    model = TumorSegmentationTask()
    
    # Fake Batch: (Batch Size, Channels, H, W)
    ct = torch.randn(2, 1, 64, 64)
    mask = torch.randint(0, 2, (2, 1, 64, 64)).float()
    batch = (ct, mask)
    
    # Manual call to training_step
    loss = model.training_step(batch, batch_idx=0)
    
    assert loss is not None
    assert loss.requires_grad # Ensure gradients are attached for backprop

def test_optimizer_config():
    model = TumorSegmentationTask(lr=0.01)
    optimizer = model.configure_optimizers()
    
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]['lr'] == 0.01