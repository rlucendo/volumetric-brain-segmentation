# Volumetric 3D Brain Tumor Segmentation (BraTS)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![MONAI](https://img.shields.io/badge/MONAI-Medical_AI-darkgreen)
![Weights & Biases](https://img.shields.io/badge/MLOps-W&B-yellow)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](WIP)

## 1. Project summary

This repository (NeuroSeg-3D) contains a fully decoupled training and evaluation pipeline for 3D brain tumor segmentation using the Medical Segmentation Decathlon dataset (Task01_BrainTumour). 

Designed as a technical showcase, this project address the complex hardware and infrastructural challenges of volumetric medical imaging (NIfTI), multi-modal MRI sequences, and clinical-grade evaluation.

---

## 2. Dataset

The model is trained on the **Medical Segmentation Decathlon (Task01_BrainTumour)**, using multi-modal MRI sequences (FLAIR, T1w, T1gd, T2w) to capture different biological properties of the tumor. 

The original dataset annotations include:

| Class | Description | Clinical Relevance |
| :--- | :--- | :--- |
| **Label 0** | Background / Healthy Tissue | Baseline reference. |
| **Label 1** | Necrotic & Non-enhancing core (NCR/NET) | Dead tissue within the tumor mass. |
| **Label 2** | Peritumoral Edema (ED) | Swelling surrounding the tumor (visible in FLAIR). |
| **Label 3** | GD-enhancing tumor (ET) | Active tumor growth with disrupted blood-brain barrier. |

### BraTS Dataset Sample

![BraTS Dataset Sample](https://drive.google.com/uc?id=1x4aAVdZFGW3CYoOd03n3n8rfdDNgR6zb)

*Note: For this initial MVP pipeline, the model is evaluated on its ability to segment the **Whole Tumor (WT)**, merging the sub-regions to establish a robust baseline.*

---

## 3. Architecture

Volumetric processing requires strict memory management and a modular infrastructure. The codebase is fully decoupled (Data, Model, Orchestration).

### Directory Structure
```text
volumetric-brain-segmentation/
├── configs/
│   ├── data_config.yaml      # Hyperparameters for DataLoader and caching
│   └── model_config.yaml     # Architecture dimensions and Loss/Optimizer settings
├── data/                     # Local data directory (ignored in git)
├── notebooks/
│   └── NeuroSeg_3D_training_pipeline.ipynb # The lab: EDA, Visualization & XAI showcase
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

### Tech stack
* **Core:** Python, PyTorch, PyTorch Lightning.
* **Medical framework:** **MONAI** for real-time volumetric ETL and medical-specific spatial transforms (Spacing, Orientation, Normalization).
* **Experiment tracking:** **Weights & Biases (W&B)** for real-time loss/Dice metric visualization and automated model artifact versioning.
* **Model:** 3D U-Net. Configured for Mixed Precision (`16-mixed`) to reduce VRAM footprint by 50% without sacrificing clinical accuracy.

---

## 4. Methodology

Training on massive 3D NIfTI files required a hardware-resilient strategy:

1.  **Data ingestion & ETL:** Used MONAI's deterministic transforms to ensure the data is loaded, correctly oriented to standard neurological views, and intensity-normalized (Z-score).
2.  **Hardware optimization:** Bypassed Google Colab's strict `/dev/shm` memory limits by customizing the dataloader workers, ensuring stable convergence on A100 GPUs.
3.  **Fault tolerance:** Implemented a robust "resume from checkpoint" logic. If the cloud instance is preempted, the script securely monkeypatches PyTorch 2.6+ `weights_only` strictness to safely resume training from local or W&B downloaded artifacts.

---

## 5. Performance and clinical evaluation

The evaluation suite (`NeuroSeg_3D_training_pipeline.ipynb`) moves beyond standard 2D metrics to provide radiological-grade feedback.

### Quantitative Metrics
* **Dice Score:** Measures the volumetric overlap between the ground truth and prediction.
* **Hausdorff Distance 95 (HD95):** Measures the absolute maximum boundary error in physical millimeters, a crucial metric for surgical margin planning.

### Clinical Explainability (XAI)
To ensure clinical trust, standard Grad-CAM was replaced with specific volumetric visualization tools:

* **Multi-planar Radiological Overlays:** Projects the 3D segmentation mask simultaneously across the Axial, Coronal, and Sagittal planes. This allows clinicians to evaluate the spatial coherence and anatomical boundaries of the prediction exactly as they would in a standard medical viewer.

![Multi-planar Radiological Overlays](https://drive.google.com/uc?id=1zkOwfX_gDZge9rOgW4T12aY2cziJ3c8S)

* **Interactive 3D Meshes:** Employs Marching Cubes and Plotly to render the predicted tumor as an interactive 3D object for topological review.

![Interactive 3D Mesh](https://drive.google.com/uc?id=1NCSmlliDQpQT7HvCRP61N7GYW6aP0gmi)

* **Shannon Entropy Uncertainty Maps ("Ring of Fire"):** Highlights decision boundaries where the model hesitates. The model confidently predicts the core and the background, but flags the exact boundary (in milimeters) where the human surgeon should pay extra attention.

![Shannon Entropy Uncertainty Map](https://drive.google.com/uc?id=1-Jv1PKoRNXhNQfK069UhV91yzDZ8J2o1)

---

## 6. How to use and replicate

### Prerequisites
* Git & Python 3.10+
* An NVIDIA GPU with at least 15GB VRAM (e.g., T4, L4, or A100) is highly recommended.

### Setup
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/rlucendo/volumetric-brain-segmentation.git
    cd volumetric-brain-segmentation
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Scenario A: Train from scratch
The training pipeline downloads the dataset automatically (if not present) and synchronizes with Weights & Biases.

```bash
export PYTHONPATH=. 
python scripts/train.py --config_dir configs
```

### Scenario B: Resuming from a preempted session
Safely resumes from the last saved `.ckpt` file, retaining optimizer states and epoch progress.

```bash
export PYTHONPATH=. 
python scripts/train.py --config_dir configs --ckpt_path checkpoints/last.ckpt
```

### Scenario C: Production inference
Applies sliding window inference to process a massive 3D volume, outputting a clinically compliant `.nii.gz` file that preserves the original affine matrix.

```bash
export PYTHONPATH=. 
python scripts/infer.py \
    --config_dir configs \
    --ckpt_path checkpoints/last.ckpt \
    --input_image data/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz \
    --output_dir predictions/
```

---

## 7. Future roadmap

Given more compute budget and time, the following enhancements would scale this MVP to production readiness:

* [ ] **Test-Time Augmentation (TTA):** Averaging predictions across flipped axes for smoother edges and lower HD95.
* [ ] **Region-Specific Sub-classing:** Upgrading the loss function to segment Edema, Enhancing Tumor, and Necrotic Core individually rather than treating it as a binary problem.
* [ ] **Distributed Data Parallel (DDP):** Scaling the pipeline across multiple A100 nodes for faster convergence.

---

## Author

**Rubén Lucendo**  
*AI Engineer & Product Builder*

Building systems that bridge the gap between theory and business value.