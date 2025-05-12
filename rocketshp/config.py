import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from omegaconf import OmegaConf

from rocketshp.utils import configure_logger

# # Load environment variables from .env file if it exists
load_dotenv()
configure_logger(os.getenv("LOGURU_LEVEL", "INFO"))

# Paths
# PROJ_ROOT = Path(__file__).resolve().parents[1]
PROJ_ROOT = Path("/mnt/home/ssledzieski/Projects/rocketshp")
logger.debug(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
EVALUATION_DATA_DIR = DATA_DIR / "evaluation"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


@dataclass
class PARAMETER_STRUCT:
    num_parameters: int = -1
    batch_size: int = 8
    crop_size: int = 512
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
    ca_dist_alpha: float = 0.0
    dyn_corr_alpha: float = 0.0
    autocorr_alpha: float = 0.0
    gcc_lmi_alpha: float = 1.0
    shp_alpha: float = 1.0
    seq_features: bool = True
    struct_features: bool = False
    struct_stage: str = "quantized"
    struct_dim: int = 1


DEFAULT_PARAMETERS = OmegaConf.structured(PARAMETER_STRUCT())

PRETRAINED_MODELS = {
    "latest": "rocketshp_pretrained_20250505_v1.ckpt",
    "v0": "rocketshp_pretrained_20250215_v0.ckpt",
    "v0_seq": "rocketshp_pretrained_20250215_seqonly_v0.ckpt",
    "v1": "rocketshp_pretrained_20250505_v1.ckpt",
    "v1_seq": "rocketshp_pretrained_20250507_v1_seq.ckpt",
    "v1_mini": "rocketshp_pretrained_20250505_v1_mini.ckpt"
}