import torch
import pytest
from src.models.unet import UNet

def test_unet_output_shape():
    """Verifies that the model outputs the same spatial resolution as input."""
    model = UNet(in_channels=1, out_channels=1)
    
    # Batch size 2, 1 Channel, 256x256
    dummy_input = torch.randn(2, 1, 256, 256)
    
    output = model(dummy_input)
    
    assert output.shape == (2, 1, 256, 256)

def test_unet_integration_backward():
    """Smoke test: check if gradients can flow."""
    model = UNet()
    input_t = torch.randn(1, 1, 64, 64) # Smaller size for speed
    target = torch.randn(1, 1, 64, 64)
    
    output = model(input_t)
    loss = torch.nn.MSELoss()(output, target)
    loss.backward()
    
    # Check if a specific layer has gradients
    assert model.layer1.step[0].weight.grad is not None