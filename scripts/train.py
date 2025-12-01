import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path

from src.data.data_module import LungDataModule
from src.engine.segmentor import TumorSegmentationTask

def parse_args():
    parser = argparse.ArgumentParser(description="Train Lung Cancer Segmentation Model")
    
    # Data params
    parser.add_argument("--data_dir", type=str, required=True, help="Path to processed data (containing train/val)")
    parser.add_argument("--output_dir", type=str, default="./artifacts", help="Where to save logs and checkpoints")
    
    # Training params
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use (0 for CPU)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Setup Data
    data_module = LungDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # 2. Setup Model
    model = TumorSegmentationTask(lr=args.lr)
    
    # 3. Setup Callbacks & Logger
    logger = TensorBoardLogger(save_dir=args.output_dir, name="unet_logs")
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_dice",
        mode="max",
        filename="best-checkpoint-{epoch:02d}-{val_dice:.2f}",
        save_top_k=3
    )
    
    early_stopping = EarlyStopping(
        monitor="val_dice",
        mode="max",
        patience=10
    )
    
    # 4. Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else "auto",
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=10
    )
    
    # 5. Train
    print("Starting training...")
    trainer.fit(model, datamodule=data_module)
    print(f"Training complete. Best model saved at: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    main()