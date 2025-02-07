import os
import tempfile
from pathlib import Path

# from torchmdnet.datasets import MDCATH
import h5py
import numpy as np

from rocketshp.config import PROCESSED_DATA_DIR
from rocketshp.datasets.utils import MDDataModule, MDDataset

MDCATH_PROCESSED_H5 = PROCESSED_DATA_DIR / "mdcath/mdcath_processed.h5"
MDCATH_TEMPS = [320, 348, 379, 413, 450]
MDCATH_REPS = [0, 1, 2, 3, 4]
MDCATH_FOLDSEEK_CLUSTERS_FILE = PROCESSED_DATA_DIR / "mdcath/foldseek_mdcath_0.2_cluster.tsv"

# def download(target_directory=f"{RAW_DATA_DIR}/mdcath"):
#     mdc = MDCATH(target_directory)
#     mdc.download()

def _renumber_pdb(filename):
    # First read all lines into memory
    with open(filename) as f:
        lines = f.readlines()

    # Process the lines
    new_lines = []
    current_chain = None
    current_resid = None  # Will store "resnum+insertion" as identifier
    new_number = None

    for line in lines:
        if line.startswith(('ATOM', 'HETATM')):
            chain = line[21]
            resnum = line[22:26].strip()  # Current residue number
            ins_code = line[26]           # Insertion code
            current_full_resid = resnum + ins_code  # Combine for unique identifier

            # If we hit a new chain, reset our counters
            if chain != current_chain:
                current_chain = chain
                current_resid = None
                new_number = None

            # If we hit a new residue (checking full residue identifier)
            if current_full_resid != current_resid:
                if new_number is None:
                    # First residue in chain - start with its number
                    new_number = 0
                else:
                    # Increment for new residue
                    new_number += 1
                current_resid = current_full_resid

            # Create new line with updated residue number and no insertion code
            new_line = (line[:22] +
                      f"{new_number:>4}" +
                      " " +    # Replace insertion code with space
                      line[27:])
            new_lines.append(new_line)
        else:
            new_lines.append(line)

    # Write back to the same file
    with open(filename, 'w') as f:
        f.writelines(new_lines)

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
        ### REPLACE HSD, HSP, HSE with HIS
        pdb = pdb.replace(b"HSD", b"HIS")
        pdb = pdb.replace(b"HSP", b"HIS")
        pdb = pdb.replace(b"HSE", b"HIS")
        ###
        pdbfile.write(pdb)
        pdbfile.flush()
        coords = h5[code][f"{temp}"][f"{replica}"]["coords"][:]

        _renumber_pdb(pdbfile.name)

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
    h5.close()
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
        ### REPLACE HSD, HSP, HSE with HIS
        pdb = pdb.replace(b"HSD", b"HIS")
        pdb = pdb.replace(b"HSP", b"HIS")
        pdb = pdb.replace(b"HSE", b"HIS")
        ###
        pdbfile.write(pdb)
        # logger.info(f"Wrote {pdbpath}")

    for temp in temp_list:
        for replica in replica_list:
            xtcpath = f"{basename}_{temp}_{replica}.xtc"
            trj = convert_to_mdtraj(h5, temp, replica)
            trj.save_xtc(xtcpath)
            # logger.info(f"Wrote {xtcpath}")

    return pdbpath, xtcpath

class MDCathDataset(MDDataset):
    def __init__(
        self,
        processed_h5: Path,
        seq_features: bool = True,
        struct_features: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(
            processed_h5=processed_h5,
            seq_features=seq_features,
            struct_features=struct_features,
            *args,
            **kwargs,
        )

    def _get_keys(self):
        return list(self._handle.keys())

    def _get_samples(self):
        return [f"{k}/T{t}/R{r}" for k in self.keys for r in MDCATH_REPS for t in MDCATH_TEMPS]

    def _code_rep_temp(self, key):
        pdb_code, temp, rep = key.split("/")
        return pdb_code, rep, int(temp.lstrip("T"))

    def _handle_path(self, pdb_code, rep, temp, is_label=False):
        if is_label:
            return f"{pdb_code}/T{temp}/{rep}"
        return pdb_code

class MDCathDataModule(MDDataModule):
    def __init__(
        self,
        processed_h5: Path = MDCATH_PROCESSED_H5,
        clusters_file: Path = MDCATH_FOLDSEEK_CLUSTERS_FILE,
        **kwargs,
    ):
        super().__init__(
            processed_h5=processed_h5,
            clusters_file=clusters_file,
            **kwargs,
        )

    def _get_dataset_class(self):
        return MDCathDataset
