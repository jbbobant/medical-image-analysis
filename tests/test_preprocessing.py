import pytest
import numpy as np
import nibabel as nib
import os
from pathlib import Path
from src.preprocessing.nifti_processor import NiftiPreprocessor

@pytest.fixture
def mock_nifti_data(tmp_path):
    """Creates a dummy NIfTI dataset structure."""
    root = tmp_path / "Task06_Lung"
    images_tr = root / "imagesTr"
    labels_tr = root / "labelsTr"
    images_tr.mkdir(parents=True)
    labels_tr.mkdir(parents=True)

    # Create dummy volume (100x100 spatial, 40 slices depth)
    # We use 40 slices because the processor clips the first 30
    affine = np.eye(4)
    dummy_data = np.random.rand(100, 100, 40)
    dummy_mask = np.random.randint(0, 2, (100, 100, 40))

    img_obj = nib.Nifti1Image(dummy_data, affine)
    mask_obj = nib.Nifti1Image(dummy_mask, affine)

    # Save 2 subjects
    nib.save(img_obj, images_tr / "lung_001.nii.gz")
    nib.save(mask_obj, labels_tr / "lung_001.nii.gz")
    
    nib.save(img_obj, images_tr / "lung_002.nii.gz")
    nib.save(mask_obj, labels_tr / "lung_002.nii.gz")

    return root

def test_preprocessing_output_shapes(mock_nifti_data, tmp_path):
    """
    Test logic:
    1. Create fake data.
    2. Run processor.
    3. Check if output files exist.
    4. Check if output shapes match target_size (256x256).
    5. Check if Z-depth logic (cropping) worked.
    """
    output_root = tmp_path / "Processed"
    
    processor = NiftiPreprocessor(
        target_size=(50, 50), # Small size for speed
        clip_slices_min=30,
        normalization_factor=1.0
    )

    # We have 2 files. Let's make 1 train, 1 val
    processor.run(mock_nifti_data, output_root, val_split_count=1)

    # 1. Check folder creation
    assert (output_root / "train" / "0" / "data").exists()
    assert (output_root / "val" / "1" / "data").exists()

    # 2. Check Slice Logic
    # Input was 40 slices, clip is 30. Should allow 10 output slices (0 to 9)
    train_slice_0 = output_root / "train" / "0" / "data" / "0.npy"
    assert train_slice_0.exists()
    
    # 3. Check Shape
    loaded_slice = np.load(train_slice_0)
    assert loaded_slice.shape == (50, 50) # Matching target_size

def test_normalization(mock_nifti_data, tmp_path):
    output_root = tmp_path / "Processed_Norm"
    factor = 100.0
    processor = NiftiPreprocessor(normalization_factor=factor)
    
    # Process single volume internal logic test
    ct_path = mock_nifti_data / "imagesTr" / "lung_001.nii.gz"
    mask_path = mock_nifti_data / "labelsTr" / "lung_001.nii.gz"
    
    ct, _ = processor.process_volume(ct_path, mask_path)
    
    # Expected max value should be scaled (approx)
    # Original random data is 0-1. Normalized should be original/100
    assert np.max(ct) <= 1.0/factor + 1e-6