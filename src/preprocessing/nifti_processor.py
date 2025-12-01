import os
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm

class NiftiPreprocessor:
    def __init__(
        self, 
        target_size: Tuple[int, int] = (256, 256), 
        clip_slices_min: int = 30, 
        normalization_factor: float = 3071.0
    ):
        """
        Args:
            target_size: Output resolution (height, width)
            clip_slices_min: Number of slices to skip from the beginning (abdomen crop)
            normalization_factor: divisor for CT standardization
        """
        self.target_size = target_size
        self.clip_slices_min = clip_slices_min
        self.normalization_factor = normalization_factor

    def load_nifti(self, path: Path) -> np.ndarray:
        return nib.load(str(path)).get_fdata()

    def process_volume(self, ct_path: Path, mask_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Loads and preprocesses a single subject volume."""
        ct_data = self.load_nifti(ct_path)
        mask_data = self.load_nifti(mask_path)

        # 1. Crop volume (z-axis is usually the last one in nibabel: x, y, z)
        # Check orientation usually required, assuming z is last based on notebook context
        ct_data = ct_data[:, :, self.clip_slices_min:]
        mask_data = mask_data[:, :, self.clip_slices_min:]

        # 2. Normalize
        ct_data = ct_data / self.normalization_factor

        return ct_data, mask_data

    def save_slices(
        self, 
        ct_volume: np.ndarray, 
        mask_volume: np.ndarray, 
        subject_id: str, 
        output_dir: Path
    ) -> None:
        """Resizes and saves individual 2D slices as .npy files."""
        
        # Directories for data and masks
        slice_dir = output_dir / subject_id / "data"
        mask_dir = output_dir / subject_id / "masks"
        slice_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        # Iterate over z-axis
        for i in range(ct_volume.shape[-1]):
            slice_img = ct_volume[:, :, i]
            mask_img = mask_volume[:, :, i]

            # Resize
            slice_resized = cv2.resize(slice_img, self.target_size)
            # Use Nearest Neighbor for masks to preserve class integers (0/1)
            mask_resized = cv2.resize(mask_img, self.target_size, interpolation=cv2.INTER_NEAREST)

            # Save
            np.save(slice_dir / f"{i}.npy", slice_resized)
            np.save(mask_dir / f"{i}.npy", mask_resized)

    def run(self, input_root: Path, output_root: Path, val_split_count: int = 6) -> None:
        """
        Main execution pipeline.
        
        Args:
            input_root: Folder containing imagesTr and labelsTr
            output_root: Folder to save processed/train and processed/val
            val_split_count: Number of subjects to hold out for validation (from the end)
        """
        images_dir = input_root / "imagesTr"
        labels_dir = input_root / "labelsTr"
        
        # Get all lung files
        all_files = sorted(list(images_dir.glob("lung_*.nii.gz")))
        total_files = len(all_files)
        train_cutoff = total_files - val_split_count

        print(f"Processing {total_files} volumes. Split: {train_cutoff} Train, {val_split_count} Val.")

        for idx, ct_path in enumerate(tqdm(all_files)):
            # Infer label path based on naming convention
            label_filename = ct_path.name
            mask_path = labels_dir / label_filename

            if not mask_path.exists():
                print(f"Warning: Mask not found for {ct_path.name}, skipping.")
                continue

            # Determine split
            is_train = idx < train_cutoff
            split_dir = "train" if is_train else "val"
            save_path = output_root / split_dir

            # Process
            ct_vol, mask_vol = self.process_volume(ct_path, mask_path)
            self.save_slices(ct_vol, mask_vol, str(idx), save_path)