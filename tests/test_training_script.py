import pytest
import pytorch_lightning as pl
from src.data.data_module import LungDataModule
from src.engine.segmentor import TumorSegmentationTask

def test_fast_dev_run(tmp_path):
    """
    Runs 1 batch of training and validation to ensure no crashes.
    Simulates what happens inside train.py.
    """
    # 1. Setup Mock Data
    data_dir = tmp_path / "processed"
    train_dir = data_dir / "train" / "0"
    val_dir = data_dir / "val" / "0"
    
    # Create directories
    (train_dir / "data").mkdir(parents=True)
    (train_dir / "masks").mkdir(parents=True)
    (val_dir / "data").mkdir(parents=True)
    (val_dir / "masks").mkdir(parents=True)
    
    # Create fake files
    import numpy as np
    np.save(train_dir / "data" / "0.npy", np.random.rand(256, 256))
    np.save(train_dir / "masks" / "0.npy", np.random.randint(0, 2, (256, 256)))
    np.save(val_dir / "data" / "0.npy", np.random.rand(256, 256))
    np.save(val_dir / "masks" / "0.npy", np.random.randint(0, 2, (256, 256)))

    # 2. Init Modules
    dm = LungDataModule(data_dir=str(data_dir), batch_size=2)
    model = TumorSegmentationTask()
    
    # 3. Trainer with fast_dev_run
    trainer = pl.Trainer(
        fast_dev_run=True, # Runs 1 batch then stops
        accelerator="cpu",
        logger=False,
        checkpoint_callback=False
    )
    
    # 4. Run
    try:
        trainer.fit(model, datamodule=dm)
    except Exception as e:
        pytest.fail(f"Training crashed: {e}")