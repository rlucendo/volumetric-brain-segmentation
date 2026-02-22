import os
import argparse
import torch

# --- PyTorch 2.6+ Security Bypass (Monkeypatch) ---
# We trained this checkpoint ourselves, so we trust it 100%.
# This forces torch.load to bypass the strict 'weights_only=True' check.
_original_load = torch.load
def _trusted_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)
torch.load = _trusted_load
# ---------------------------------------------------

from omegaconf import OmegaConf

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    SaveImage,
    AsDiscrete,
)
from monai.inferers import sliding_window_inference
from monai.data import Dataset, DataLoader

# Import our custom LightningModule
from src.models.seg_module import BraTSSegmentationModule

def get_inference_transforms():
    """
    Transforms for inference. 
    Strictly standardizes the input volume to match training conditions.
    """
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])

def main(config_dir: str, ckpt_path: str, input_image_path: str, output_dir: str):
    """
    End-to-end inference pipeline for a single patient volume.
    Generates and saves the 3D segmentation mask as a NIfTI file.
    """
    print(f"Loading configuration and model from: {ckpt_path}")
    
    # 1. Load model configuration
    model_cfg = OmegaConf.load(os.path.join(config_dir, "model_config.yaml"))
    
    # 2. Instantiate the model and load trained weights
    # We use Lightning's built-in checkpoint loader
    model = BraTSSegmentationModule.load_from_checkpoint(
        checkpoint_path=ckpt_path, 
        cfg=model_cfg,
        strict=False
    )
    
    # Put model in evaluation mode and move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 3. Prepare the input data
    # We create a dictionary mimicking MONAI's expected structure
    test_files = [{"image": input_image_path}]
    test_ds = Dataset(data=test_files, transform=get_inference_transforms())
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=0)

    # 4. Define the saver transform to export the prediction
    # It will automatically use the original image's metadata (Affine matrix)
    saver = SaveImage(
        output_dir=output_dir, 
        output_postfix="pred", 
        output_ext=".nii.gz",
        resample=False # Keep it in the standardized spacing for now
    )
    post_pred = AsDiscrete(argmax=True)

    print(f"Starting inference on: {input_image_path}")
    
    # 5. Execute inference
    with torch.no_grad():
        for batch_data in test_loader:
            inputs = batch_data["image"].to(device)
            
            # Use sliding window to predict the whole volume chunk by chunk
            outputs = sliding_window_inference(
                inputs=inputs, 
                roi_size=model_cfg.roi_size, 
                sw_batch_size=4, 
                predictor=model.forward,
                overlap=0.5 # 50% overlap for smoother predictions
            )
            
            # Convert logits to discrete class labels (0, 1, 2, 3)
            # outputs shape: (1, 4, H, W, D) -> (1, 1, H, W, D)
            discrete_outputs = post_pred(outputs[0]).unsqueeze(0)
            
            # Attach the prediction back to the dictionary for the saver
            batch_data["pred"] = discrete_outputs.cpu()
            
            # Save to disk using MONAI's saver (needs the meta_dict from the original image)
            # The saver automatically extracts metadata from batch_data["image_meta_dict"]
            saver(batch_data["pred"][0], meta_data=batch_data["image"][0].meta)

    print(f"Inference complete! Prediction saved in: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a 3D NIfTI volume")
    parser.add_argument("--config_dir", type=str, default="configs", help="Path to configs")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to trained .ckpt")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input .nii.gz")
    parser.add_argument("--output_dir", type=str, default="predictions", help="Output folder")
    
    args = parser.parse_args()
    main(args.config_dir, args.ckpt_path, args.input_image, args.output_dir)