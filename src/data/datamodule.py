import os
from typing import Optional
import pytorch_lightning as pl
# from torch.utils.data import DataLoader
from monai.data import DataLoader
from omegaconf import DictConfig

from monai.apps import DecathlonDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    RandSpatialCropSamplesd,
    RandFlipd,
    Orientationd,
    Spacingd,
)

class BraTSDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        """
        Professional LightningDataModule for 3D BraTS.
        Handles data downloading, medical-grade transformations, and dataloader generation.
        """
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None

    def get_train_transforms(self):
        """
        Training data transformations.
        Includes spatial resampling, intensity normalization, and 3D patch extraction.
        """
        return Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            # Standardize to neurological orientation (RAS - Right, Anterior, Superior)
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # Resample to a uniform isometric voxel spacing
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            # Apply channel-wise Z-score normalization (e.g., across FLAIR, T1w, T1gd, T2w)
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            # Data Augmentation: Random spatial cropping
            RandSpatialCropSamplesd(
                keys=["image", "label"],
                roi_size=self.cfg.patch_size,
                num_samples=2, # Extract 2 patches per loaded volume to optimize I/O
                random_center=True,
                random_size=False
            ),
            # Data Augmentation: Random axis flipping for rotational invariance
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        ])

    def get_val_transforms(self):
        """
        Validation data transformations.
        Excludes random cropping to evaluate on the full anatomical volume.
        """
        return Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ])

    def setup(self, stage: Optional[str] = None):
        """
        Downloads (if absent) and prepares the MONAI dataset.
        Implements smart caching to prevent I/O bottlenecks during epochs.
        """
        # Self-healing infrastructure: Ensure the target directory exists 
        # before MONAI attempts to access or download files into it.
        os.makedirs(self.cfg.data_dir, exist_ok=True)
        
        if stage == "fit" or stage is None:
            # Training split (80% of the dataset)
            self.train_dataset = DecathlonDataset(
                root_dir=self.cfg.data_dir,
                task=self.cfg.task,
                transform=self.get_train_transforms(),
                section="training",
                download=True,
                cache_num=self.cfg.cache_num,
                val_frac=0.2, # Deterministically allocate 20% for internal validation
                seed=42
            )
            
            # Validation split (20% of the dataset)
            self.val_dataset = DecathlonDataset(
                root_dir=self.cfg.data_dir,
                task=self.cfg.task,
                transform=self.get_val_transforms(),
                section="validation",
                download=False, # Data already downloaded in the training block
                cache_num=self.cfg.cache_num,
                val_frac=0.2,
                seed=42
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.cfg.batch_size, 
            num_workers=self.cfg.num_workers, 
            shuffle=True,
            pin_memory=True # Accelerate CPU-to-GPU memory transfer
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.cfg.val_batch_size, 
            num_workers=self.cfg.num_workers, 
            shuffle=False,
            pin_memory=True
        )