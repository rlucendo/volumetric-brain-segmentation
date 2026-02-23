import os
import argparse
import torch

# PyTorch 2.6+ checkpoint security bypass
# Allows loading of trusted local checkpoints by overriding strict 'weights_only' constraints.
_original_load = torch.load
def _trusted_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)
torch.load = _trusted_load

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

# Import custom LightningModule
from src.models.seg_module import BraTSSegmentationModule

def get_inference_transforms():
    """
    Defines the data transformation pipeline for inference.
    Standardizes the input volume to strictly match the required training spatial conditions.
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
    Executes the end-to-end inference pipeline for a single patient volume.
    Generates and exports the 3D segmentation mask as a NIfTI file.
    """
    print(f"Loading configuration and model from: {ckpt_path}")
    
    # Load model configuration
    model_cfg = OmegaConf.load(os.path.join(config_dir, "model_config.yaml"))
    
    # Instantiate the model and load trained weights
    # Utilizing Lightning's built-in checkpoint loader
    model = BraTSSegmentationModule.load_from_checkpoint(
        checkpoint_path=ckpt_path, 
        cfg=model_cfg,
        strict=False
    )
    
    # Set model to evaluation mode and allocate to the appropriate hardware device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Prepare the input data
    # Construct a dictionary matching MONAI's expected input structure
    test_files = [{"image": input_image_path}]
    test_ds = Dataset(data=test_files, transform=get_inference_transforms())
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=0)

    # Define the saver transform to export the prediction
    # Retains the original image's metadata and affine transformation matrix
    saver = SaveImage(
        output_dir=output_dir, 
        output_postfix="pred", 
        output_ext=".nii.gz",
        resample=False # Maintain standardized spacing without resampling
    )
    post_pred = AsDiscrete(argmax=True)

    print(f"Starting inference on: {input_image_path}")
    
    # Execute inference
    with torch.no_grad():
        for batch_data in test_loader:
            inputs = batch_data["image"].to(device)
            
            # Apply sliding window inference to process the full volume in discrete chunks
            outputs = sliding_window_inference(
                inputs=inputs, 
                roi_size=model_cfg.roi_size, 
                sw_batch_size=4, 
                predictor=model.forward,
                overlap=0.5 # 50% overlap to ensure smooth transitions between patches
            )
            
            # Convert raw logits to discrete class labels (0, 1, 2, 3)
            # Reshape outputs: (1, 4, H, W, D) -> (1, 1, H, W, D)
            discrete_outputs = post_pred(outputs[0]).unsqueeze(0)
            
            # Append the prediction tensor back to the data dictionary for the saver transform
            batch_data["pred"] = discrete_outputs.cpu()
            
            # Export the prediction to disk using MONAI's SaveImage transform
            # Metadata is automatically extracted from the source image dictionary
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