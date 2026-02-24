import os
import argparse
from omegaconf import OmegaConf
from src.data.datamodule import BraTSDataModule

def main(config_dir: str):
    """
    Extract, Transform, Load (ETL) standalone script.
    Responsible exclusively for data ingestion, medical transformations, 
    and caching, ensuring the dataset is ready before any GPU is allocated.
    """
    print("=== [PHASE 1: DATA INGESTION & ETL] ===")
    
    # 1. Load Data Configuration
    data_cfg = OmegaConf.load(os.path.join(config_dir, "data_config.yaml"))
    
    # 2. Instantiate DataModule
    datamodule = BraTSDataModule(cfg=data_cfg)
    
    # 3. Force the setup process (Download, Extract, Cache)
    print(f"Executing ETL pipeline for task: {data_cfg.task}...")
    datamodule.setup(stage="fit")
    
    print("✓ ETL Phase complete. Dataset is structured and cached.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ETL Pipeline")
    parser.add_argument(
        "--config_dir", 
        type=str, 
        default="configs", 
        help="Path to directory containing YAML configuration files"
    )
    args = parser.parse_args()
    
    main(args.config_dir)