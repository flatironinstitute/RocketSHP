import os
import shlex
import subprocess as sp
import tempfile
from pathlib import Path

import h5py
import numpy as np
import typer
from loguru import logger
from torchmdnet.datasets import MDCATH

from rocketshp.config import INTERIM_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


def download(target_directory=f"{RAW_DATA_DIR}/mdcath"):
    mdc = MDCATH(target_directory)
    mdc.download()


def _open_h5_file(h5):
    if isinstance(h5, str) or isinstance(h5, Path):
        h5 = h5py.File(h5, "r")
    code = [_ for _ in h5][0]
    return h5, code


def _extract_structure_and_coordinates(h5, code, temp, replica):
    """
    Extracts the structure in PDB format and coordinates from an H5 file based on temperature and replica.

    Parameters:
    h5 : h5py.File
        An opened H5 file object containing protein structures and simulation data.
    code : str
        The identifier for the dataset in the H5 file.
    temp : int or float
        The temperature (in Kelvin).
    replica : int
        The replica number.

    Returns:
    tuple
        A tuple containing the PDB data as bytes and the coordinates as a numpy array.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as pdbfile:
        pdb = h5[code]["pdbProteinAtoms"][()]
        pdbfile.write(pdb)
        pdbfile.flush()
        coords = h5[code][f"{temp}"][f"{replica}"]["coords"][:]
    coords = coords / 10.0
    return pdbfile.name, coords


def convert_to_mdtraj(h5, temp, replica):
    """
    Convert data from an H5 file to an MDTraj trajectory object.

    This function extracts the first protein atom structure and coordinates
    for a given temperature and replica from an H5 file and creates an MDTraj
    trajectory object. This object can be used for further molecular dynamics
    analysis.

    Parameters:
    h5 : h5py.File
        An opened H5 file object containing protein structures and simulation data.
    temp : int or float
        The temperature (in Kelvin) at which the simulation was run. This is used
        to select the corresponding dataset within the H5 file.
    replica : int
        The replica number of the simulation to extract data from. This is used
        to select the corresponding dataset within the H5 file.

    Returns:
    md.Trajectory
        An MDTraj trajectory object containing the loaded protein structure and
        simulation coordinates.

    Example:
    -------
    import h5py
    import mdtraj as md

    # Open the H5 file
    with h5py.File('simulation_data.h5', 'r') as h5file:
        traj = convert_to_mdtraj(h5file, 300, 1)

    # Now 'traj' can be used for analysis with MDTraj
    """
    import mdtraj as md

    h5, code = _open_h5_file(h5)
    pdb_file_name, coords = _extract_structure_and_coordinates(h5, code, temp, replica)
    trj = md.load(pdb_file_name)
    os.unlink(pdb_file_name)
    trj.xyz = coords.copy()
    trj.time = np.arange(1, coords.shape[0] + 1)
    return trj


def convert_to_files(
    fn, basename=None, temp_list=[320, 348, 379, 413, 450], replica_list=[0, 1, 2, 3, 4]
):
    """
    Converts data from an H5 file to separate PDB and XTC files based on specified temperatures and replicas.

    This function reads protein atom structures and simulation data from an H5 file and writes a single PDB file
    and multiple XTC files. Each XTC file corresponds to a specific temperature and replica combination. The
    function uses `convert_to_mdtraj` to generate MDTraj trajectory objects which are then saved in the XTC format.

    Parameters:
    fn : str
        The file name or path to the H5 file containing the simulation data.
    basename : str
        The base name to use for output files.  If None, it is taken from the domain ID.
    temp_list : list of int, optional
        A list of temperatures (in Kelvin) for which the simulations were run. Defaults to [320, 348, 379, 413, 450].
    replica_list : list of int, optional
        A list of replica numbers to extract data for. Defaults to [0, 1, 2, 3, 4].

    Outputs:
    Creates a PDB file named `{basename}.pdb` and multiple XTC files named `{basename}_{temp}_{replica}.xtc`,
    where `{temp}` and `{replica}` are values from `temp_list` and `replica_list`.

    Example:
    -------
    # Convert data to files with base name 'protein_simulation'
    convert_to_files('simulation_data.h5', 'protein_simulation')
    """

    h5, code = _open_h5_file(fn)

    if not basename:
        basename = code

    pdbpath = f"{basename}.pdb"
    with open(pdbpath, "wb") as pdbfile:
        pdb = h5[code]["pdbProteinAtoms"][()]
        pdbfile.write(pdb)
        logger.info(f"Wrote {pdbpath}")

    for temp in temp_list:
        for replica in replica_list:
            xtcpath = f"{basename}_{temp}_{replica}.xtc"
            trj = convert_to_mdtraj(h5, temp, replica)
            trj.save_xtc(xtcpath)
            logger.info(f"Wrote {xtcpath}")

    return pdbpath, xtcpath


############
# Commands #
############


@app.command()
def get_files(
    domain: str,
    mdcath_dir: Path = RAW_DATA_DIR / "mdcath",
    output_dir: Path = INTERIM_DATA_DIR / "mdcath",
    temp: int = 348,
    repl: int = 0,
):
    logger.info(f"Getting trajectory for {domain}, {temp}, {repl}")
    fn = mdcath_dir / f"mdcath_dataset_{domain}.h5"
    basename = output_dir / domain / domain
    (output_dir / domain).mkdir(parents=True, exist_ok=True)
    pdbpath, xtcpath = convert_to_files(fn, basename, [temp], [repl])
    logger.success(f"Domain {domain} successfully converted")
    return pdbpath, xtcpath


@app.command()
def convert_foldseek(
    domain: str,
    mdcath_dir: Path = RAW_DATA_DIR / "mdcath",
    output_dir: Path = INTERIM_DATA_DIR / "mdcath",
    temp: int = 348,
    repl: int = 0,
    FOLDSEEK_PATH="/mnt/home/ssledzieski/local/foldseek/9-427df8a/bin/foldseek",
    # FOLDSEEK_PATH = "foldseek",
):
    logger.info(f"Converting FoldSeek data for {domain}")

    fn = mdcath_dir / f"mdcath_dataset_{domain}.h5"
    traj_fn = output_dir / domain / f"{domain}_{temp}_{repl}_traj.pdb"
    fs_fn = output_dir / domain / f"{domain}_{temp}_{repl}_traj.3di"

    logger.info(f"Getting trajectory for {domain}, {temp}, {repl}")
    traj = convert_to_mdtraj(fn, temp, repl)
    traj.save_pdb(traj_fn)

    cmd = f"{FOLDSEEK_PATH} structureto3didescriptor {traj_fn} {fs_fn}"
    logger.info(f"Running: {cmd}")
    proc = sp.Popen(shlex.split(cmd), stdout=sp.PIPE, stderr=sp.PIPE)
    out, err = proc.communicate()
    logger.info(out.decode())

    if proc.returncode:
        logger.error(f"Exited with {proc.returncode}: {err.decode()}")
        raise RuntimeError(err.decode())
    else:
        logger.success(f"FoldSeek data for {domain} successfully converted")
        return fs_fn


if __name__ == "__main__":
    app()
