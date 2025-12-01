from typing import Any, List, Optional
import torch
import pytorch_lightning as pl
from torch.optim import Adam
import matplotlib.pyplot as plt

from src.models.unet import UNet
from utils.metrics import DiceScore
from utils.visualization import plot_prediction

class TumorSegmentationTask(pl.LightningModule):
    def __init__(self, lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        
        # Initialize Architecture
        self.model = UNet(in_channels=1, out_channels=1)
        
        # Loss and Metrics
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.dice_metric = DiceScore()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        ct, mask = batch
        mask = mask.float()
        logits = self(ct)
        
        loss = self.loss_fn(logits, mask)
        
        # Logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Optional: Log train dice occasionally (expensive)
        if batch_idx % 100 == 0:
            dice = self.dice_metric(logits, mask)
            self.log("train_dice", dice, on_step=True, on_epoch=False)
            
        return loss

    def validation_step(self, batch, batch_idx):
        ct, mask = batch
        mask = mask.float()
        logits = self(ct)
        
        loss = self.loss_fn(logits, mask)
        dice = self.dice_metric(logits, mask)
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_dice", dice, on_step=False, on_epoch=True, prog_bar=True)

        # Log Images for the first batch of every validation epoch
        if batch_idx == 0:
            self._log_images(ct, mask, logits)
            
        return loss

    def _log_images(self, ct, mask, logits):
        """Logs a single example slice to TensorBoard."""
        # Take the first image in the batch
        # Dimensions: (Batch, Channel, H, W) -> (H, W)
        ct_slice = ct[0, 0]
        mask_slice = mask[0, 0]
        pred_slice = logits[0, 0]
        
        fig = plot_prediction(ct_slice, mask_slice, pred_slice)
        
        # Log to TensorBoard
        # Note: self.logger might be None during unit tests
        if self.logger:
            self.logger.experiment.add_figure(
                "Validation/Predictions", 
                fig, 
                global_step=self.global_step
            )
            
        plt.close(fig) # Memory cleanup

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)