import torch
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, Subset
from lightning import LightningDataModule

from rocketshp.config import PROCESSED_DATA_DIR
from rocketshp.datasets.data_utils import MDDataset, MDDataModule

ATLAS_FOLDSEEK_CLUSTERS_FILE = PROCESSED_DATA_DIR / "atlas/foldseek_atlas_0.2_cluster.tsv"
ATLAS_PROCESSED_H5 = PROCESSED_DATA_DIR / "atlas/atlas_processed.h5"
ATLAS_TEMP = 300
ATLAS_REPS = [1, 2, 3]

class ATLASDataset(MDDataset):
    def __init__(
        self,
        processed_h5: Path,
        seq_features: bool = True,
        struct_features: bool = True,
    ):
        super().__init__(
            processed_h5=processed_h5,
            seq_features=seq_features,
            struct_features=struct_features,
        )
        
    def _get_keys(self):
        keys = []
        with open(ATLAS_FOLDSEEK_CLUSTERS_FILE, "r") as f:
            for line in f:
                keys.append(line.strip().split("\t")[1])
        return sorted(keys)
    
    def _get_samples(self):
        return [f"{k}/R{i}" for k in self.keys for i in ATLAS_REPS]

    def _code_rep_temp(self, key):
        pdb_code, rep = key.split("/")
        temp = ATLAS_TEMP
        return pdb_code, rep, temp
    
    def _handle_path(self, pdb_code, rep, _, is_label=False):
        if is_label:
            return f"{pdb_code}/{rep}"
        return pdb_code


class ATLASDataModule(MDDataModule):
    def __init__(
        self,
        processed_h5: Path = ATLAS_PROCESSED_H5,
        clusters_file: Path = ATLAS_FOLDSEEK_CLUSTERS_FILE,
        shuffle: bool = True,
        **kwargs,
    ):
        super().__init__(
            processed_h5=processed_h5,
            clusters_file=clusters_file,
            shuffle=shuffle,
            **kwargs,
        )

    def _get_dataset_class(self):
        return ATLASDataset