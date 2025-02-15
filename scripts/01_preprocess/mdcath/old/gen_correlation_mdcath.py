
import os
import subprocess as sp
import sys

import psutil

# import netchem
# import netcalc
from loguru import logger
from tqdm import tqdm

from rocketshp import config


def check_memory_usage(max_memory_percent=90):
    """
    Monitor memory usage and kill the process if it exceeds the threshold.
    
    Args:
        max_memory_percent (float): Maximum memory usage percentage allowed (default: 90%)
    """
    process = psutil.Process(os.getpid())
    memory_percent = process.memory_percent()

    if memory_percent > max_memory_percent:
        print(f"Memory usage exceeded {max_memory_percent}% (current: {memory_percent:.1f}%)")
        print("Terminating process...")
        sys.exit(1)

    return memory_percent

os.environ["LOGURU_LEVEL"] = "INFO"

MDCATH_DATA_DIR = config.RAW_DATA_DIR / "mdcath"
MDCATH_PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR / "mdcath"

mdcath_files = list(MDCATH_DATA_DIR.glob("mdcath_dataset_*.h5"))
TEMPS = [320, 348, 379, 413, 450]
REPS = [0, 1, 2, 3, 4]
LOCAL_DIST_CUTOFF = 0.75 # in nm
DO_LOCAL_ALIGN = True

for mdc_f in tqdm(mdcath_files, total=len(mdcath_files)):
    pdb_code = mdc_f.stem.split("_")[-1]

    for temp in TEMPS:
        for rep in REPS:

            # if gen correlation exists, continue
            local_suff = "local_" if DO_LOCAL_ALIGN else ""
            corr_matrix_filename = str(MDCATH_PROCESSED_DATA_DIR / pdb_code / f"{pdb_code}_{temp}_{rep}_{local_suff}_corr_matrix.pt")
            if os.path.exists(corr_matrix_filename):
                logger.info(f"Skipping {pdb_code} temp {temp} rep {rep}")
                continue

            check_memory_usage(max_memory_percent=50)

            logger.info(f"Processing {pdb_code} temp {temp} rep {rep}")
            cmd = f"python scripts/gen_corr_mdcath_inner.py {mdc_f} {temp} {rep}"
            proc = sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE)
            out, err = proc.communicate()
            if proc.returncode != 0:
                logger.error(f"Error processing {pdb_code} temp {temp} rep {rep}")
                logger.error(f"Command: {cmd}")
                logger.error(f"Output: {out}")
                logger.error(f"Error: {err}")
                continue
