from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from dataclasses import dataclass
from omegaconf import OmegaConf

# # Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

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
    max_epochs: int = 20
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
    n_layers: int = 4
    rmsf_loss: str = "rmse"
    rmsf_alpha: float = 0.0
    ca_alpha: float = 1.0

DEFAULT_PARAMETERS = OmegaConf.structured(PARAMETER_STRUCT())

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
