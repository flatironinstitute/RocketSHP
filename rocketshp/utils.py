import sys
from collections.abc import Sequence
from contextlib import AbstractContextManager as ContextManager
from typing import TypeVar

import numpy as np
import torch
from loguru import logger

TSequence = TypeVar("TSequence", bound=Sequence)


def seed_everything(seed: int):
    """Seed all random number generators for reproducibility."""
    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
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


def fp32_autocast_context(device_type: str) -> ContextManager[torch.amp.autocast]:  # type: ignore
    """
    Returns an autocast context manager that disables downcasting by AMP.

    Args:
        device_type: The device type ('cpu' or 'cuda')

    Returns:
        An autocast context manager with the specified behavior.
    """
    if device_type == "cpu":
        return torch.amp.autocast(device_type, enabled=False)  # type: ignore
    elif device_type == "cuda":
        return torch.amp.autocast(device_type, dtype=torch.float32)  # type: ignore
    else:
        raise ValueError(f"Unsupported device type: {device_type}")


def slice_python_object_as_numpy(
    obj: TSequence, idx: int | list[int] | slice | np.ndarray
) -> TSequence:
    """
    Slice a python object (like a list, string, or tuple) as if it was a numpy object.

    Example:
        >>> obj = "ABCDE"
        >>> slice_python_object_as_numpy(obj, [1, 3, 4])
        "BDE"

        >>> obj = [1, 2, 3, 4, 5]
        >>> slice_python_object_as_numpy(obj, np.arange(5) < 3)
        [1, 2, 3]
    """
    if isinstance(idx, int):
        idx = [idx]

    if isinstance(idx, np.ndarray) and idx.dtype == bool:
        sliced_obj = [obj[i] for i in np.where(idx)[0]]
    elif isinstance(idx, slice):
        sliced_obj = obj[idx]
    else:
        sliced_obj = [obj[i] for i in idx]

    match obj, sliced_obj:
        case str(), list():
            sliced_obj = "".join(sliced_obj)
        case _:
            sliced_obj = obj.__class__(sliced_obj)  # type: ignore

    return sliced_obj  # type: ignore


def save_small_checkpoint(check_in_path: str, check_out_path: str):
    """
    Saves a small checkpoint file that can be loaded with `Model.load_from_checkpoint` for inference, but cannot be used to resume training.

    Args:
        check_in_path: The path to the original checkpoint file.
        check_out_path: The path to save the cleaned checkpoint file.
    """
    checkpoint = torch.load(check_in_path)

    checkpoint_small = {
        "state_dict": checkpoint["state_dict"],
        "hyper_parameters": checkpoint["hyper_parameters"],
        "hparams_name": checkpoint["hparams_name"],
    }

    torch.save(checkpoint_small, check_out_path)
    logger.info(f"Saved clean checkpoint to {check_out_path}")
