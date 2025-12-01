from pathlib import Path
from typing import Optional, List
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
from tqdm import tqdm

from .lung_dataset import LungDataset
from .transforms import ImageAugmentor

class LungDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        data_dir: str, 
        batch_size: int = 16, 
        num_workers: int = 4
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset: Optional[LungDataset] = None
        self.val_dataset: Optional[LungDataset] = None
        self.sampler: Optional[WeightedRandomSampler] = None

    def setup(self, stage: Optional[str] = None):
        """
        Load datasets and compute sampler weights for class imbalance.
        """
        train_path = self.data_dir / "train"
        val_path = self.data_dir / "val"

        # Initialize Datasets
        self.train_dataset = LungDataset(
            root_dir=train_path, 
            transform=ImageAugmentor(is_train=True)
        )
        
        self.val_dataset = LungDataset(
            root_dir=val_path, 
            transform=ImageAugmentor(is_train=False)
        )

        # --- Imbalance Handling ---
        # Only compute weights for training
        if stage == "fit" or stage is None:
            print("Computing class weights for sampler...")
            self.sampler = self._compute_sampler(self.train_dataset)

    def _compute_sampler(self, dataset: LungDataset) -> WeightedRandomSampler:
        """
        Iterates over the dataset to find tumor vs background slices 
        and assigns weights.
        """
        targets = []
        # We need to iterate the dataset (fast-load without transforms)
        for path in tqdm(dataset.files, desc="Scanning dataset for imbalance"):
            mask_path = dataset._get_mask_path(path)
            mask = np.load(str(mask_path))
            
            # Check if tumor exists in slice (1 = tumor, 0 = background)
            has_tumor = 1 if np.any(mask > 0) else 0
            targets.append(has_tumor)

        targets = np.array(targets)
        
        # Calculate counts
        count_tumor = np.sum(targets == 1)
        count_bg = len(targets) - count_tumor
        
        # Avoid division by zero
        if count_tumor == 0: 
            print("Warning: No tumors found in training set.")
            return None

        # Weights: Inverse frequency
        weight_bg = 1.0 / count_bg
        weight_tumor = 1.0 / count_tumor
        
        # Assign weight to each sample
        sample_weights = np.where(targets == 1, weight_tumor, weight_bg)
        
        # Create sampler
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(sample_weights),
            replacement=True
        )
        return sampler

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            sampler=self.sampler, # The magic happens here
            shuffle=False, # Must be False when using Sampler
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )