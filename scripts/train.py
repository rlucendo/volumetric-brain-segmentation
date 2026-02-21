import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf

# Adjust these imports based on your exact repository structure
from src.data.datamodule import BraTSDataModule
from src.models.seg_module import BraTSSegmentationModule

def main(config_dir: str):
    """
    Main training orchestrator.
    Sets up configurations, data, model, loggers, and the Lightning Trainer.
    """
    # 1. Load Configurations
    data_cfg = OmegaConf.load(os.path.join(config_dir, "data_config.yaml"))
    model_cfg = OmegaConf.load(os.path.join(config_dir, "model_config.yaml"))

    # 2. Ensure Reproducibility
    pl.seed_everything(42, workers=True)

    # 3. Initialize Modules
    datamodule = BraTSDataModule(cfg=data_cfg)
    model = BraTSSegmentationModule(cfg=model_cfg)

    # 4. Setup MLOps Loggers (Weights & Biases)
    wandb_logger = WandbLogger(
        project="brats-3d-segmentation",
        name="unet-baseline",
        log_model="all" # Automatically upload checkpoints to W&B cloud
    )
    
    # Log hyperparameters to W&B
    wandb_logger.experiment.config.update({"data": OmegaConf.to_container(data_cfg)})

    # 5. Setup Callbacks
    # Save the model with the highest validation Dice score
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="brats-{epoch:02d}-{val/dice_score:.4f}",
        monitor="val/dice_score",
        mode="max",
        save_top_k=3,
        save_last=True,
    )

    # Stop training if the model stops improving to save compute resources
    early_stopping_callback = EarlyStopping(
        monitor="val/dice_score",
        patience=15,
        mode="max",
        verbose=True
    )

    # 6. Initialize Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=100,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        accelerator="gpu",
        devices=1,
        precision="16-mixed", # Crucial for 3D data: reduces VRAM usage by 50%
        log_every_n_steps=5,
        deterministic=True,
    )

    # 7. Execute Training
    print("Starting training pipeline...")
    trainer.fit(model=model, datamodule=datamodule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D Segmentation Model")
    parser.add_argument(
        "--config_dir", 
        type=str, 
        default="configs", 
        help="Directory containing YAML configuration files"
    )
    args = parser.parse_args()
    
    main(args.config_dir)