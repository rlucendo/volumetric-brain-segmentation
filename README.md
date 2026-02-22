# NeuroSeg-3D: Volumetric 3D Brain Tumor Segmentation (BraTS)

This repository contains an end-to-end MLOps pipeline for 3D brain tumor segmentation using the Medical Segmentation Decathlon dataset (Task01_BrainTumour). 

Designed as a technical showcase, this project address the complex hardware and infrastructural challenges of volumetric medical imaging (NIfTI), multi-modal MRI sequences, and clinical-grade evaluation.

## Key Features & Architectural Decisions

* **3D U-Net Architecture:** Implemented from scratch to learn spatial contexts across all three anatomical planes (Axial, Coronal, Sagittal) simultaneously.
* **MONAI Medical Framework:** Utilized for real-time volumetric ETL, caching, and medical-specific transformations (spacing, orientation, normalization).
* **PyTorch Lightning Orchestration:** Decoupled training logic from infrastructure. Features built-in Mixed Precision (`16-mixed`), deterministic seeding, and automatic checkpointing.
* **Weights & Biases Integration:** Cloud-based experiment tracking, real-time loss/Dice metric visualization, and automated model artifact versioning.
* **Hardware-Resilient Design:** Configured to survive Google Colab's strict `/dev/shm` memory limits and preemptions by implementing robust "Resume from Checkpoint" capabilities.
* **Clinical-Grade XAI & Evaluation:** Moved beyond Grad-CAM to implement 3D Marching Cubes rendering, Shannon Entropy Uncertainty Maps, and Normalized Voxel Confusion Matrices.

## Repository Structure

```text
volumetric-brain-segmentation/
├── configs/
│   ├── data_config.yaml      # Hyperparameters for DataLoader and caching
│   └── model_config.yaml     # Architecture dimensions and Loss/Optimizer settings
├── data/                     # Local data directory (ignored in git)
├── notebooks/
│   └── 00_end_to_end_demo.ipynb # Showcase notebook with EDA, Inference & XAI
├── scripts/
│   ├── train.py              # Main training orchestrator (supports resuming)
│   └── infer.py              # Single-volume NIfTI inference script
├── src/
│   ├── data/
│   │   └── datamodule.py     # LightningDataModule using MONAI DataLoaders
│   └── models/
│       └── seg_module.py     # LightningModule (U-Net + DiceCELoss + Optimizer)
└── requirements.txt
```

## Installation & Setup

Clone the repository and install the strict medical dependencies:

```bash
git clone https://github.com/rlucendo/volumetric-brain-segmentation.git
cd volumetric-brain-segmentation
pip install -r requirements.txt
```

*Note: For optimal performance, an NVIDIA GPU with at least 15GB VRAM (e.g., T4, L4, or A100) is highly recommended.*

## Training the Model

The training pipeline downloads the dataset automatically (if not present) and synchronizes with Weights & Biases. 

To start training from scratch:
```bash
export PYTHONPATH=. 
python scripts/train.py --config_dir configs
```

**Resuming from a preempted session:**
To bypass PyTorch 2.6+ strict `weights_only` security checks on local trusted configurations, the script safely monkeypatches the loader.
```bash
export PYTHONPATH=. 
python scripts/train.py --config_dir configs --ckpt_path checkpoints/last.ckpt
```

## Inference & Prediction

The `infer.py` script applies sliding window inference to process massive 3D volumes efficiently, outputting a clinically compliant `.nii.gz` file that preserves the original affine matrix.

```bash
export PYTHONPATH=. 
python scripts/infer.py \
    --config_dir configs \
    --ckpt_path checkpoints/last.ckpt \
    --input_image data/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz \
    --output_dir predictions/
```

## Evaluation & Clinical Visualizations

The `notebooks/NeuroSeg-3D_training_pipeline.ipynb` provides a comprehensive visual suite designed for radiological review:
1. **Multi-planar Reconstructions:** Transparent overlays on Axial, Coronal, and Sagittal slices.
2. **Interactive 3D Meshes:** Rendered using Plotly and Marching Cubes.
3. **Shannon Entropy (Uncertainty Maps):** Highlights decision boundaries where the model hesitates, crucial for clinical trust.
4. **Hausdorff Distance (HD95):** Measures the absolute boundary error in physical millimeters.

## Future Improvements
Given more compute budget and time, the following enhancements would be implemented:
* **Test-Time Augmentation (TTA):** Averaging predictions across flipped axes for smoother edges.
* **Region-Specific Sub-classing:** Segmenting Edema, Enhancing Tumor, and Necrotic Core individually rather than treating it as a "Whole Tumor" binary problem.
* **Distributed Data Parallel (DDP):** Scaling the pipeline across multiple A100 nodes.