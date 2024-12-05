import os
import tempfile
from pathlib import Path

import h5py
import torch
import numpy as np
import pandas as pd
from loguru import logger
from torch.utils.data import DataLoader, Dataset, Subset
from lightning import LightningDataModule
# from torchmdnet.datasets import MDCATH

from rocketshp.config import PROCESSED_DATA_DIR
from rocketshp.datasets.data_utils import _dict_collate_fn

MDCATH_PROCESSED_H5 = PROCESSED_DATA_DIR / "mdcath/mdcath_processed.h5"
MDCATH_TEMPS = [320, 348, 379, 413, 450]
MDCATH_REPS = [0, 1, 2, 3, 4]
FOLDSEEK_CLUSTERS_FILE = PROCESSED_DATA_DIR / "mdcath/foldseek_mdcath_0.2_cluster.tsv"

# def download(target_directory=f"{RAW_DATA_DIR}/mdcath"):
#     mdc = MDCATH(target_directory)
#     mdc.download()


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
        # logger.info(f"Wrote {pdbpath}")

    for temp in temp_list:
        for replica in replica_list:
            xtcpath = f"{basename}_{temp}_{replica}.xtc"
            trj = convert_to_mdtraj(h5, temp, replica)
            trj.save_xtc(xtcpath)
            # logger.info(f"Wrote {xtcpath}")

    return pdbpath, xtcpath

class MDCathDataset(Dataset):
    def __init__(
        self,
        processed_h5: Path = MDCATH_PROCESSED_H5,
        target: str = "rmsf",
        seq_features: bool = True,
        struct_features: bool = True,
    ):
        super().__init__()
        self._path = processed_h5
        self._handle = h5py.File(self._path, "r")
        self._target = target

        self._use_seq = seq_features
        self._use_struct = struct_features

        self.keys = list(self._handle.keys())
        self.samples = [f"{k}/T{t}/R{r+1}" for k in self.keys for r in MDCATH_REPS for t in MDCATH_TEMPS]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rep_key = self.samples[idx]
        pdb_code, temp, rep = rep_key.split("/")

        features = {}
        if self._use_seq:
            seq_features = self._handle[f"{pdb_code}/embedding"][:]
            features["seq_feats"] = torch.from_numpy(seq_features)
        if self._use_struct:
            struct_features = self._handle[f"{pdb_code}/struct_tokens"][:]
            features["struct_feats"] = torch.from_numpy(struct_features)
        features["temp"] = int(temp.lstrip("T"))

        labels = {}
        labels["rmsf"] = torch.from_numpy(self._handle[f"{pdb_code}/{temp}/{rep}/{self._target}"][:])
        labels["ca_dist"] = torch.from_numpy(self._handle[f"{pdb_code}/{temp}/{rep}/ca_distances"][:]).squeeze()

        return features, labels

    def __del__(self):
        if hasattr(self, "_handle") and isinstance(self._handle, h5py.File):
            self._handle.close()

class MDCathDataModule(LightningDataModule):
    def __init__(
        self,
        processed_h5: Path = MDCATH_PROCESSED_H5,
        target: str = "rmsf",
        seq_features: bool = True,
        struct_features: bool = True,
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 0,
        train_pct: float = 0.8,
        val_pct: float = 0.1,
        clusters_file: Path = FOLDSEEK_CLUSTERS_FILE,
        random_seed: int = 42,
    ):
        super().__init__()

        self._processed_h5 = processed_h5
        self._target = target
        self._seq_features = seq_features
        self._struct_features = struct_features

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.train_pct = train_pct
        self.val_pct = val_pct
        self.clusters = pd.read_csv(clusters_file,sep="\t",header=None)

        self.random_seed = random_seed

    def prepare_data(self):
        pass

    def setup(self, stage: str):

        self.dataset = MDCathDataset(
            self._processed_h5,
            target=self._target,
            seq_features=self._seq_features,
            struct_features=self._struct_features,
        )

        all_clusters = self.clusters[0].unique()
        rng = np.random.default_rng(self.random_seed)
        shuffled_entities = rng.permutation(all_clusters)
        n_entities = len(shuffled_entities)

        train_size = int(self.train_pct * n_entities)
        val_size = int(self.val_pct * n_entities)
        self.train_subset = shuffled_entities[:train_size]
        self.val_subset = shuffled_entities[train_size : train_size + val_size]
        self.test_subset = shuffled_entities[train_size + val_size :]

        train_subset_pdb = self.clusters[self.clusters[0].isin(self.train_subset)][1].unique()
        val_subset_pdb = self.clusters[self.clusters[0].isin(self.val_subset)][1].unique()
        test_subset_pdb = self.clusters[self.clusters[0].isin(self.test_subset)][1].unique()

        train_sample_idx, val_sample_idx, test_sample_idx = [], [], []
        for i, s in enumerate(self.dataset.samples):
            pdb_code, _, _ = s.split("/")
            if pdb_code in train_subset_pdb:
                train_sample_idx.append(i)
            elif pdb_code in val_subset_pdb:
                val_sample_idx.append(i)
            elif pdb_code in test_subset_pdb:
                test_sample_idx.append(i)
            else:
                raise ValueError(f"Sample {s} not found in any subset.")

        self.train_data = Subset(self.dataset, train_sample_idx)
        self.val_data = Subset(self.dataset, val_sample_idx)
        self.test_data = Subset(self.dataset, test_sample_idx)
        assert len(self.train_data) + len(self.val_data) + len(self.test_data) == len(self.dataset)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=_dict_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_dict_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_dict_collate_fn,
        )
