import pytest
import torch
import numpy as np
from pathlib import Path
from src.data.lung_dataset import LungDataset
from src.data.transforms import ImageAugmentor
from src.data.data_module import LungDataModule

@pytest.fixture
def mock_processed_data(tmp_path):
    """Creates synthetic processed .npy files structure."""
    train_dir = tmp_path / "processed" / "train" / "0"
    (train_dir / "data").mkdir(parents=True)
    (train_dir / "masks").mkdir(parents=True)

    # Create 10 dummy slices
    for i in range(10):
        # Image: float random
        img = np.random.rand(256, 256).astype(np.float32)
        # Mask: binary 0 or 1
        mask = np.random.randint(0, 2, (256, 256)).astype(np.float32)
        
        np.save(train_dir / "data" / f"{i}.npy", img)
        np.save(train_dir / "masks" / f"{i}.npy", mask)

    return tmp_path / "processed"

def test_dataset_loading(mock_processed_data):
    train_path = mock_processed_data / "train"
    ds = LungDataset(train_path, transform=None)
    
    assert len(ds) == 10
    
    img, mask = ds[0]
    # Check PyTorch conversion and Channel dim adding
    assert torch.is_tensor(img)
    assert img.shape == (1, 256, 256)
    assert mask.shape == (1, 256, 256)

def test_augmentation_shapes(mock_processed_data):
    """Ensure augmentation keeps shapes consistent."""
    train_path = mock_processed_data / "train"
    augmentor = ImageAugmentor(is_train=True)
    ds = LungDataset(train_path, transform=augmentor)
    
    img, mask = ds[0]
    assert img.shape == (1, 256, 256)
    assert mask.shape == (1, 256, 256)

def test_datamodule_sampler(mock_processed_data):
    """Verifies that the datamodule computes weights without crashing."""
    dm = LungDataModule(data_dir=str(mock_processed_data), batch_size=2)
    dm.setup(stage="fit")
    
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    
    imgs, masks = batch
    assert imgs.shape == (2, 1, 256, 256)
    
    # Check if sampler was created
    assert dm.sampler is not None
    assert isinstance(dm.sampler, torch.utils.data.WeightedRandomSampler)