import argparse
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import cv2

from src.engine.segmentor import TumorSegmentationTask
from src.preprocessing.nifti_processor import NiftiPreprocessor

def parse_args():
    parser = argparse.ArgumentParser(description="Run Inference on a CT Volume")
    parser.add_argument("--input_nifti", type=str, required=True, help="Path to raw .nii.gz CT file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument("--output_dir", type=str, default="./predictions", help="Where to save the result")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def load_model(checkpoint_path: str, device: str):
    """Loads model weights from checkpoint."""
    model = TumorSegmentationTask.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()
    return model

def save_nifti(mask_volume: np.ndarray, reference_nifti_path: str, output_path: Path):
    """Saves the segmentation mask using the affine of the original input."""
    ref_img = nib.load(reference_nifti_path)
    affine = ref_img.affine
    
    # Reconstruct NIfTI
    nifti_img = nib.Nifti1Image(mask_volume.astype(np.uint8), affine)
    nib.save(nifti_img, str(output_path))

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, args.device)
    
    # 2. Preprocessing Config (Must match training!)
    # Ideally, load this from a config file saved during training
    target_size = (256, 256)
    clip_min = 30
    norm_factor = 3071.0
    
    # 3. Load & Process Volume
    print(f"Processing {args.input_nifti}...")
    nifti_data = nib.load(args.input_nifti).get_fdata()
    
    # Apply cropping
    # Note: We need to remember original shape for padding back if we want exact overlay,
    # but for this segmentation task, we'll output the cropped mask.
    cropped_data = nifti_data[:, :, clip_min:]
    normalized_data = cropped_data / norm_factor
    
    # 4. Inference Loop (Slice by Slice)
    prediction_volume = np.zeros(normalized_data.shape, dtype=np.uint8)
    
    with torch.no_grad():
        for i in tqdm(range(normalized_data.shape[-1]), desc="Inference"):
            slice_img = normalized_data[:, :, i]
            
            # Resize to model input size
            slice_resized = cv2.resize(slice_img, target_size)
            
            # Prepare tensor (1, 1, H, W)
            input_tensor = torch.from_numpy(slice_resized).float().unsqueeze(0).unsqueeze(0).to(args.device)
            
            # Predict
            logits = model(input_tensor)
            preds = torch.sigmoid(logits) > 0.5
            
            # Post-process (Resize back to original slice dimensions if necessary)
            # Here we assume output needs to match input crop resolution
            pred_mask = preds.cpu().numpy()[0, 0].astype(np.uint8)
            
            # Resize back to original slice shape (usually 512x512)
            original_h, original_w = slice_img.shape
            pred_mask_original_size = cv2.resize(
                pred_mask, 
                (original_w, original_h), 
                interpolation=cv2.INTER_NEAREST
            )
            
            prediction_volume[:, :, i] = pred_mask_original_size

    # 5. Save Result
    # Create a full volume mask (pad the clipped abdomen area with zeros)
    full_mask_shape = nifti_data.shape
    final_mask = np.zeros(full_mask_shape, dtype=np.uint8)
    final_mask[:, :, clip_min:] = prediction_volume
    
    output_filename = Path(args.input_nifti).name.replace(".nii.gz", "_pred.nii.gz")
    save_path = output_dir / output_filename
    
    save_nifti(final_mask, args.input_nifti, save_path)
    print(f"Segmentation saved to {save_path}")

if __name__ == "__main__":
    main()