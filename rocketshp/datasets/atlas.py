import torch
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, Subset
from lightning import LightningDataModule

from rocketshp.config import PROCESSED_DATA_DIR
from rocketshp.datasets.data_utils import _dict_collate_fn

FOLDSEEK_CLUSTERS_FILE = PROCESSED_DATA_DIR / "atlas/foldseek_atlas_0.2_cluster.tsv"
ATLAS_PROCESSED_H5 = PROCESSED_DATA_DIR / "atlas/atlas_processed.h5"
ATLAS_TEMP = 300
ATLAS_REPS = [1, 2, 3]

class ATLASDataset(Dataset):
    def __init__(
        self,
        processed_h5: Path,
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
        self.samples = [f"{k}/R{i}" for k in self.keys for i in ATLAS_REPS]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rep_key = self.samples[idx]
        pdb_code, rep = rep_key.split("/")

        features = {}
        if self._use_seq:
            seq_features = self._handle[f"{pdb_code}/embedding"][:]
            features["seq_feats"] = torch.from_numpy(seq_features)
        if self._use_struct:
            struct_features = self._handle[f"{pdb_code}/struct_tokens"][:]
            features["struct_feats"] = torch.from_numpy(struct_features)
        features["temp"] = torch.tensor([ATLAS_TEMP])

        labels = {}
        labels["rmsf"] = torch.from_numpy(self._handle[f"{pdb_code}/{rep}/{self._target}"][:])
        labels["ca_dist"] = torch.from_numpy(self._handle[f"{pdb_code}/{rep}/ca_distances"][:]).squeeze()

        return features, labels

    def __del__(self):
        if hasattr(self, "_handle") and isinstance(self._handle, h5py.File):
            self._handle.close()


class ATLASDataModule(LightningDataModule):
    def __init__(
        self,
        processed_h5: Path = ATLAS_PROCESSED_H5,
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

        self.dataset = ATLASDataset(
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
            pdb_code, _ = s.split("/")
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