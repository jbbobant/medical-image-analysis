import torch
import numpy as np
import imgaug
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from typing import Tuple, Optional

class ImageAugmentor:
    def __init__(self, is_train: bool = True):
        self.is_train = is_train
        
        # Define the augmentation sequence
        if self.is_train:
            self.seq = iaa.Sequential([
                iaa.Affine(
                    translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)},
                    scale=(0.85, 1.15),
                    rotate=(-45, 45)
                ),
                iaa.ElasticTransformation(alpha=(0, 5.0), sigma=0.25)
            ])
        else:
            self.seq = None

    def __call__(self, slice_img: np.ndarray, mask_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            slice_img: Shape (H, W) or (H, W, C)
            mask_img: Shape (H, W) or (H, W, C)
        Returns:
            Transformed arrays
        """
        # If Validation/Test, just return copies
        if not self.is_train or self.seq is None:
            return slice_img, mask_img

        # Fix for PyTorch DataLoader multiprocessing RNG issues
        # We perform the deterministic seeding per call based on torch's generator
        random_seed = torch.randint(0, 1000000, (1,)).item()
        imgaug.seed(random_seed)

        # Wrap mask for imgaug
        segmap = SegmentationMapsOnImage(mask_img, shape=mask_img.shape)
        
        # Apply augmentation
        slice_aug, mask_aug_obj = self.seq(image=slice_img, segmentation_maps=segmap)
        
        # Unpack mask
        mask_aug = mask_aug_obj.get_arr()
        
        return slice_aug, mask_aug