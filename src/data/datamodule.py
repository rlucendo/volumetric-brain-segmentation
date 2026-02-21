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
        DataModule profesional para BraTS 3D.
        Maneja descarga, transformaciones médicas y generación de dataloaders.
        """
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None

    def get_train_transforms(self):
        """
        Transformaciones para entrenamiento.
        Incluye remuestreo espacial, normalización y extracción de parches 3D.
        """
        return Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            # Orientación estándar neurológica (RAS)
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # Remuestreo a un espaciado isométrico uniforme (opcional pero recomendado)
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            # Normalización Z-score por canal (FLAIR, T1, etc.)
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            # Data Augmentation: Recortes aleatorios
            RandSpatialCropSamplesd(
                keys=["image", "label"],
                roi_size=self.cfg.patch_size,
                num_samples=2, # Genera 2 parches por cada volumen cargado
                random_center=True,
                random_size=False
            ),
            # Data Augmentation: Volteo en ejes
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        ])

    def get_val_transforms(self):
        """
        Transformaciones para validación.
        Sin recortes aleatorios (evaluamos el volumen completo).
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
        Descarga (si no existe) y prepara el dataset de MONAI.
        Usa caché inteligente para no saturar el disco en cada epoch.
        """
        if stage == "fit" or stage is None:
            # Entrenamiento (80% de los datos)
            self.train_dataset = DecathlonDataset(
                root_dir=self.cfg.data_dir,
                task=self.cfg.task,
                transform=self.get_train_transforms(),
                section="training",
                download=True,
                cache_num=self.cfg.cache_num,
                val_frac=0.2, # Separa 20% para validación interna de forma determinista
                seed=42
            )
            
            # Validación (20% de los datos)
            self.val_dataset = DecathlonDataset(
                root_dir=self.cfg.data_dir,
                task=self.cfg.task,
                transform=self.get_val_transforms(),
                section="validation",
                download=False, # Ya se descargó arriba
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
            pin_memory=True # Acelera la transferencia CPU -> GPU
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.cfg.val_batch_size, 
            num_workers=self.cfg.num_workers, 
            shuffle=False,
            pin_memory=True
        )