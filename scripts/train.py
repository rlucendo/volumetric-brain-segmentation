import os
import argparse
import torch
import pytorch_lightning as pl

# Enable Tensor Cores for NVIDIA A100 to maximize throughput
torch.set_float32_matmul_precision('medium')

# --- PyTorch 2.6+ Security Bypass (Monkeypatch) ---
# We trained this checkpoint ourselves, so we trust it 100%.
# This forces torch.load to bypass the strict 'weights_only=True' check,
# avoiding the whack-a-mole with nested OmegaConf internal classes.
_original_load = torch.load
def _trusted_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)
torch.load = _trusted_load
# ---------------------------------------------------

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf

from src.data.datamodule import BraTSDataModule
from src.models.seg_module import BraTSSegmentationModule

def main(config_dir: str, ckpt_path: str = None):
    """
    Main training orchestrator.
    """
    # Load Configurations
    data_cfg = OmegaConf.load(os.path.join(config_dir, "data_config.yaml"))
    model_cfg = OmegaConf.load(os.path.join(config_dir, "model_config.yaml"))

    pl.seed_everything(42, workers=True)

    datamodule = BraTSDataModule(cfg=data_cfg)
    model = BraTSSegmentationModule(cfg=model_cfg)

    # Note: We use the same W&B project/name to group the resumed runs
    wandb_logger = WandbLogger(
        project="brats-3d-segmentation",
        name="unet-baseline",
        log_model="all"
    )
    wandb_logger.experiment.config.update({"data": OmegaConf.to_container(data_cfg)})

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="brats-{epoch:02d}-{val/dice_score:.4f}",
        monitor="val/dice_score",
        mode="max",
        save_top_k=3,
        save_last=True, # This is crucial: it always saves a 'last.ckpt' file
    )

    early_stopping_callback = EarlyStopping(
        monitor="val/dice_score",
        patience=15,
        mode="max",
        verbose=True
    )

    trainer = pl.Trainer(
        max_epochs=100,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        log_every_n_steps=5,
        deterministic=False,
    )

    print(f"Starting training pipeline... Resuming from: {ckpt_path if ckpt_path else 'Scratch'}")
    
    # Pass the ckpt_path to the fit method. If it's None, it starts from scratch.
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D Segmentation Model")
    parser.add_argument(
        "--config_dir", 
        type=str, 
        default="configs", 
        help="Directory containing YAML configuration files"
    )
    # New argument to handle checkpoint resuming
    parser.add_argument(
        "--ckpt_path", 
        type=str, 
        default=None, 
        help="Path to the .ckpt file to resume training from"
    )
    args = parser.parse_args()
    
    main(args.config_dir, args.ckpt_path)