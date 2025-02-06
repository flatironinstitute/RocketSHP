import os, sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from dataclasses import dataclass
from omegaconf import OmegaConf
from rocketshp.utils import configure_logger

# # Load environment variables from .env file if it exists
load_dotenv()
configure_logger(os.getenv("LOGURU_LEVEL", "INFO"))

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.debug(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

@dataclass
class PARAMETER_STRUCT:
    num_parameters: int = -1
    batch_size: int = 8
    lr: float = 0.001
    grad_norm: bool = True
    max_epochs: int = 20
    epoch_scale: int = -1
    num_data_workers: int = 31
    shuffle: bool = True
    random_seed: int = 0
    train_pct: float = 0.8
    val_pct: float = 0.1
    precision: str = "highest"
    embedding_dim: int = 1536
    output_dim: int = 1
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    square_loss: bool = False
    variance_norm: bool = False
    rmsf_alpha: float = 1.0
    ca_alpha: float = 1.0
    dyn_corr_alpha: float = 1.0
    seq_features: bool = True
    struct_features: bool = False

DEFAULT_PARAMETERS = OmegaConf.structured(PARAMETER_STRUCT())