import os
import argparse
import torch
import pytorch_lightning as pl

# Configure TensorFloat-32 (TF32) matrix multiplication to maximize throughput
torch.set_float32_matmul_precision('medium')

_original_load = torch.load
def _trusted_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)
torch.load = _trusted_load

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf

from src.data.datamodule import BraTSDataModule
from src.models.seg_module import BraTSSegmentationModule

def main(config_dir: str):
    """
    Idempotent training orchestrator.
    Automatically detects the current state and decides whether to skip, resume, or train from scratch.
    """
    CHECKPOINT_DIR = "checkpoints"
    LAST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "last.ckpt")
    MARKER_FILE = os.path.join(CHECKPOINT_DIR, ".training_completed")

    print("=== [PHASE 2: STATE-AWARE MODEL TRAINING] ===")

    # --- 1. IDEMPOTENCY CHECK (State Detection) ---
    if os.path.exists(MARKER_FILE):
        print("SUCCESS: Model is already fully trained (Marker found).")
        print("Skipping training phase. You can safely proceed to evaluation or deployment.")
        return  # Exit gracefully without loading heavy libraries into GPU

    if os.path.exists(LAST_CKPT_PATH):
        print(f"STATE DETECTED: Incomplete training found at '{LAST_CKPT_PATH}'. Resuming process...")
        resume_ckpt = LAST_CKPT_PATH
    else:
        print("STATE DETECTED: No existing model found. Starting fresh training from scratch...")
        resume_ckpt = None

    # --- 2. INITIALIZE PIPELINE ---
    data_cfg = OmegaConf.load(os.path.join(config_dir, "data_config.yaml"))
    model_cfg = OmegaConf.load(os.path.join(config_dir, "model_config.yaml"))

    pl.seed_everything(42, workers=True)

    datamodule = BraTSDataModule(cfg=data_cfg)
    model = BraTSSegmentationModule(cfg=model_cfg)

    wandb_logger = WandbLogger(project="brats-3d-segmentation", name="unet-baseline", log_model="all")
    wandb_logger.experiment.config.update({"data": OmegaConf.to_container(data_cfg)})

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR, 
        filename="brats-{epoch:02d}-{val/dice_score:.4f}",
        monitor="val/dice_score", mode="max", save_top_k=3, save_last=True,
    )

    early_stopping_callback = EarlyStopping(
        monitor="val/dice_score", patience=15, mode="max", verbose=True
    )

    trainer = pl.Trainer(
        max_epochs=100, logger=wandb_logger, callbacks=[checkpoint_callback, early_stopping_callback],
        accelerator="gpu", devices=1, precision="16-mixed", log_every_n_steps=5, deterministic=False,
    )

    # --- 3. EXECUTE TRAINING ---
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=resume_ckpt)

    # --- 4. MARK AS COMPLETE ---
    # If trainer.fit() finishes without crashing (either max_epochs reached or early stopping triggered),
    # we create the marker file to ensure idempotency in future runs.
    with open(MARKER_FILE, 'w') as f:
        f.write("Training completed successfully.\n")
    
    print("\n✓ Training Phase finalized. State marker created.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="State-Aware 3D Segmentation Training")
    parser.add_argument("--config_dir", type=str, default="configs", help="Path to YAML configs")
    args = parser.parse_args()
    
    main(args.config_dir)