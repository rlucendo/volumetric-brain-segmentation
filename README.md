# volumetric-brain-segmentation

## NeuroSeg-3D: End-to-End Volumetric Segmentation Pipeline

**A Production-Ready 3D Medical Image Segmentation Pipeline for Brain Tumor Detection (BraTS/MSD).**

This repository implements an industrial-grade Deep Learning pipeline for volumetric brain tumor segmentation using the **Medical Segmentation Decathlon (Task01_BrainTumor)** dataset. 

This project is designed focusing on resource optimization (VRAM/RAM) and the scalability of NIfTI image processing in high-performance computing environments.

---

## Key Technical Features

* **Architecture**: 3D U-Net / SegResNet optimized for multi-channel data (FLAIR, T1w, T1gd, T2w), designed to capture complex spatial dependencies in medical volumes.
* **Memory Management**: Implementation of MONAI's `CacheDataset` and **Automatic Mixed Precision (AMP)** to optimize VRAM usage, allowing training on hardware with limited resources like Google Colab.
* **Strategy**: Patch-based training with **Sliding Window Inference** for precise full-volume evaluation without loss of resolution.
* **Metrics**: Rigorous evaluation using the Dice Similarity Coefficient ($DSC$) and Hausdorff Distance, ensuring clinical relevance in the results.
* **Engineering**: Modular structure based on **PyTorch Lightning**, static typing for code robustness, and centralized configuration via **Hydra/YAML**.
