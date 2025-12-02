#  Volumetric Lung Tumor Segmentation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red)
![Lightning](https://img.shields.io/badge/PyTorch_Lightning-1.4.9-purple)
![License](https://img.shields.io/badge/License-MIT-green)

**LungNet** is a deep learning framework designed to automate the detection and segmentation of lung tumors from 3D Computed Tomography (CT) scans. By leveraging a slice-by-slice 2D U-Net architecture, this project translates volumetric medical imaging data into precise, pixel-wise tumor masks, aiding in early diagnosis and radiomic analysis.

##  Project Overview & Medical Relevance

Manual segmentation of lung nodules is a labor-intensive and highly subjective task for radiologists. 

* **The Challenge:** Lung tumors vary wildly in size, shape, and texture. Furthermore, a full-body CT scan contains hundreds of slices, but tumors may only appear in a small fraction of them (severe class imbalance).
* **The Solution:** This repository implements a robust pipeline that treats 3D NIfTI volumes as a sequence of 2D slices. It employs "smart" sampling to handle the rarity of tumor pixels and utilizes heavy augmentation to generalize across patient variability.

##  Technical Architecture

The system is built on **PyTorch Lightning** for modularity and scalability.

### 1. Model Architecture: Classic U-Net
We utilize a U-Net architecture, the gold standard for biomedical image segmentation.
* **Encoder:** Captures context via a contracting path using `DoubleConv` blocks (Conv2d -> ReLU -> Conv2d -> ReLU) and Max Pooling.
* **Decoder:** Enables precise localization using an expanding path with bilinear upsampling and skip connections.
* **Output:** A 1x1 convolution produces a pixel-wise probability map.

### 2. Handling Class Imbalance
In medical datasets, "background" (healthy tissue) vastly outnumbers "tumor" pixels. Standard training would result in a model that simply predicts "background" everywhere.
* **WeightedRandomSampler:** We scan the dataset prior to training to identify slices containing tumors. The `LungDataModule` calculates inverse frequency weights and over-samples slices with positive tumor labels during training.
* **Loss Function:** We use `BCEWithLogitsLoss` for numerical stability combined with Dice Score monitoring.

### 3. Data Pipeline & Preprocessing
Raw CT data (NIfTI format) requires rigorous preprocessing before entering the neural network:
* **Z-Axis Clipping:** The first 30 slices (often containing the abdomen or artifacts) are clipped to focus on the lung region.
* **Normalization:** Pixel intensities are normalized by a factor of 3071.0 to standardize Hounsfield Units (HU) across scanners.
* **Augmentation:** To prevent overfitting, we use `imgaug` for real-time affine transformations (rotation, scaling, translation) and elastic deformations, but only during the training phase.

##  Repository Structure

```bash
├── src
│   ├── data
│   │   ├── data_module.py       # LightningDataModule with WeightedSampler
│   │   ├── lung_dataset.py      # Custom Torch Dataset for .npy slices
│   │   └── transforms.py        # ImgAug augmentations (Affine, Elastic)
│   ├── engine
│   │   └── segmentor.py         # LightningModule (Train/Val loop, Logging)
│   ├── models
│   │   └── unet.py              # PyTorch U-Net implementation
│   └── preprocessing
│       └── nifti_processor.py   # NIfTI to 2D .npy slice converter
├── scripts
│   ├── inference.py             # Inference on raw .nii.gz volumes
│   └── train.py                 # Main training entry point
├── utils
│   ├── metrics.py               # Dice Score implementation
│   └── visualization.py         # Matplotlib plotting for TensorBoard
├── requirements.txt             # Dependencies
└── LICENSE                      # MIT License
```


##  Getting Started

### Prerequisites
Ensure you have Python 3.10+ installed. Install dependencies:

```bash
pip install -r requirements.txt
```
## 1. Data Preparation
The system expects NIfTI files (.nii.gz). The preprocessing pipeline converts these 3D volumes into 2D numpy arrays for efficient loading.

```Python

from src.preprocessing.nifti_processor import NiftiPreprocessor
from pathlib import Path
```
# Example usage

```python 
processor = NiftiPreprocessor(target_size=(256, 256), clip_slices_min=30)
processor.run(
    input_root=Path("./raw_data"), 
    output_root=Path("./processed_data"), 
    val_split_count=6
)
```

## 2. Training
Run the training script pointing to your processed data directory.

```ash
python scripts/train.py \
    --data_dir ./processed_data \
    --batch_size 16 \
    --max_epochs 50 \
    --gpus 1
```
This will automatically handle train/val splitting, logging to TensorBoard, and saving the best model checkpoint based on the Dice Score.

## 3. Inference
To generate a segmentation mask for a new patient CT scan:

```Bash

python scripts/inference.py \
    --input_nifti path/to/patient_ct.nii.gz \
    --checkpoint artifacts/unet_logs/version_0/checkpoints/best.ckpt \
    --output_dir predictions \
    --device cuda
```

The script will reconstruct the 3D volume from 2D predictions and save it as a NIfTI file (_pred.nii.gz).

###  Testing

The project includes a comprehensive test suite using pytest.

```Bash

# Run all tests
pytest tests/
```
Tests cover:

Preprocessing: Verifies Z-depth clipping and normalization logic.

Model: Checks input/output tensor shapes and gradient flow.

Overfitting Check: test_fast_dev_run simulates a single batch pass to ensure no runtime crashes.

### Results & Visualization
During training, the TumorSegmentationTask logs sample predictions to TensorBoard. This includes:

Original CT Slice (Bone cmap)

Ground Truth Overlay (Autumn cmap)

Model Prediction (Heatmap)

### License
This project is licensed under the MIT License - see the LICENSE file for details.
