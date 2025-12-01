from pathlib import Path
from typing import List, Tuple, Callable, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

class LungDataset(Dataset):
    def __init__(
        self, 
        root_dir: Path, 
        transform: Optional[Callable] = None
    ):
        """
        Args:
            root_dir: Path to 'processed/train' or 'processed/val'
            transform: Instance of ImageAugmentor or compatible callable
        """
        self.root_dir = root_dir
        self.transform = transform
        self.files = self._load_files()

    def _load_files(self) -> List[Path]:
        """Recursively finds all data slice .npy files."""
        # Structure is root/subject_id/data/*.npy
        return list(self.root_dir.glob("**/data/*.npy"))

    def _get_mask_path(self, slice_path: Path) -> Path:
        """Replaces 'data' with 'masks' in the path."""
        # path/to/0/data/100.npy -> path/to/0/masks/100.npy
        return slice_path.parent.parent / "masks" / slice_path.name

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        slice_path = self.files[idx]
        mask_path = self._get_mask_path(slice_path)

        # Load numpy arrays
        # Note: Preprocessing saved them as (256, 256) floats/ints
        img = np.load(str(slice_path)).astype(np.float32)
        mask = np.load(str(mask_path)).astype(np.float32)

        # Apply augmentations (expects numpy)
        if self.transform:
            img, mask = self.transform(img, mask)

        # Convert to Tensor and add Channel dimension
        # Input: (H, W) -> Output: (1, H, W)
        img_t = torch.from_numpy(img).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        return img_t, mask_t