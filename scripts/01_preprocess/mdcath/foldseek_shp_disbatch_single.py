import gc
import os
import shlex
import subprocess as sp
import sys
import tempfile

import torch
from loguru import logger
from tqdm import tqdm

from rocketshp import config
from rocketshp.data.mdcath import convert_to_mdtraj
from rocketshp.trajectory import frame_to_chain, normalize

MDCATH_DATA_DIR = config.RAW_DATA_DIR / "mdcath"
RESULT_DIR = config.PROCESSED_DATA_DIR / "mdcath/fs_shp"

FS_3DI_LIST = [
    "L",
    "A",
    "G",
    "V",
    "S",
    "E",
    "R",
    "T",
    "I",
    "D",
    "P",
    "K",
    "Q",
    "N",
    "F",
    "Y",
    "M",
    "H",
    "W",
    "C",
]


def foldseek_process(pdb_file, out_file, threads=30):
    cmd = f"foldseek structureto3didescriptor {pdb_file} {out_file} --threads {threads}"
    proc = sp.Popen(shlex.split(cmd), stdout=sp.PIPE, stderr=sp.PIPE)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise Exception(f"foldseek failed: {stderr.decode()}")
    else:
        logger.debug(stdout)
    with open(out_file) as f:
        results = f.readline().strip().split("\t")
        return results[2]


def seq_list_to_tensor(seq_list):
    max_len = max([len(i) for i in seq_list])
    seq_tensor = torch.zeros(len(seq_list), max_len, dtype=torch.long)
    for i, seq in enumerate(seq_list):
        exploded = [FS_3DI_LIST.index(j) for j in seq]
        seq_tensor[i] = torch.tensor(exploded)
    return seq_tensor


def compute_trajectory_shp(pdb_id, start=0, end=None, stride=100):
    mdcath_f, rep, temp = pdb_id
    pdb_code = mdcath_f.split("_")[-1].split(".")[0]
    logger.info(f"Processing {pdb_code}:{rep}:{temp}")
    # logger.info(f"Worker {os.getpid()} memory before trajectory: {process.memory_info().rss / 1024 / 1024} MB")

    # Load trajectory
    traj = convert_to_mdtraj(mdcath_f, temp, rep)
    # logger.info(f"Worker {os.getpid()} memory after load: {process.memory_info().rss / 1024 / 1024} MB")

    traj = normalize(traj, ca_only=False)
    traj = traj[start:end:stride]

    shp = []
    with torch.inference_mode():
        for frame in tqdm(traj):
            chain = frame_to_chain(frame)
            # create a tempfile for the chain and run foldseek
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
                chain.to_pdb(tmp.name)
                tmp_chain_file = tmp.name
                out_file = tmp_chain_file + ".3di"
                seq_3di = foldseek_process(tmp_chain_file, out_file)
                shp.append(seq_3di)
    shp = seq_list_to_tensor(shp)

    # Explicit cleanup
    del traj
    gc.collect()  # Force garbage collection
    # logger.info(f"Worker {os.getpid()} memory after cleanup: {process.memory_info().rss / 1024 / 1024} MB")
    # logger.info(f"Worker {os.getpid()} has {len(gc.get_objects())} total objects")

    return {
        "h5_file": mdcath_f,
        "pdb_code": pdb_code,
        "rep": rep,
        "temp": temp,
        "fs_shp": shp,
    }


if __name__ == "__main__":
    mdcath_f = sys.argv[1]
    pdb_code = mdcath_f.split("_")[-1].split(".")[0]
    rep = int(sys.argv[2])
    temp = int(sys.argv[3])
    result = compute_trajectory_shp((mdcath_f, rep, temp), stride=10)

    os.makedirs(str(RESULT_DIR / pdb_code), exist_ok=True)
    save_loc = str(RESULT_DIR / pdb_code / f"{pdb_code}_rep_{rep}_temp{temp}.pt")
    logger.info(f"Saving to {save_loc}")
    torch.save(result, save_loc)
