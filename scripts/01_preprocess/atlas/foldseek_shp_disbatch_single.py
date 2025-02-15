import sys
import gc
import os
import torch
import subprocess as sp
import shlex
import mdtraj as md
import tempfile
from tqdm import tqdm
from loguru import logger
from rocketshp import config
from rocketshp.trajectory import normalize, frame_to_chain

ATLAS_DATA_DIR = config.RAW_DATA_DIR / "atlas"
RESULT_DIR = config.PROCESSED_DATA_DIR / "atlas/fs_shp"

FS_3DI_LIST = [
        "L", "A", "G", "V", "S", "E", "R", "T", "I", "D",
        "P", "K", "Q", "N", "F", "Y", "M", "H", "W", "C"
    ]

def foldseek_process(pdb_file, out_file, threads=30):
    cmd = f"foldseek structureto3didescriptor {pdb_file} {out_file} --threads {threads}"
    proc = sp.Popen(shlex.split(cmd), stdout=sp.PIPE, stderr=sp.PIPE)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise Exception(f"foldseek failed: {stderr.decode()}")
    else:
        logger.debug(stdout)
    with open(out_file,"r") as f:
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
    pdb_code, rep = pdb_id
    logger.info(f"Processing {pdb_code}:{rep}")
    # logger.info(f"Worker {os.getpid()} memory before trajectory: {process.memory_info().rss / 1024 / 1024} MB")
    
    # Load trajectory
    xtc_f = ATLAS_DATA_DIR / pdb_code[:2] / f"{pdb_code}_prod_R{rep}_fit.xtc"
    pdb_f = ATLAS_DATA_DIR / pdb_code[:2] / f"{pdb_code}.pdb"
    traj = md.load(str(xtc_f), top=pdb_f)
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
        "pdb_code": pdb_code,
        "rep": rep,
        "xtc_file": xtc_f.stem,
        "pdb_file": pdb_f.stem,
        "fs_shp": shp,
    }

if __name__ == "__main__":
    pdb_id = sys.argv[1]
    rep = int(sys.argv[2])
    result = compute_trajectory_shp((pdb_id, rep))
    
    os.makedirs(str(RESULT_DIR / pdb_id[:2]), exist_ok=True)
    torch.save(result, str(RESULT_DIR / pdb_id[:2] / f"{pdb_id}_rep_{rep}.pt"))