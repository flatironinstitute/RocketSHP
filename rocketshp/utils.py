import sys
from loguru import logger

def seed_everything(seed: int):
    """Seed all random number generators for reproducibility."""
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def configure_logger(level: str = "INFO"):
    try:
        from tqdm import tqdm
        try:
            logger.remove(0)
        except ValueError:
            pass
        logger.add(lambda msg: tqdm.write(msg, end=""), level=level)
    except ModuleNotFoundError:
        pass
    logger.configure(handlers=[{"sink": sys.stdout, "level": level}])