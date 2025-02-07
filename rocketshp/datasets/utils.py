from functools import lru_cache
from pathlib import Path

import h5py
import lightning as L
import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset, Subset

from rocketshp.esm3 import get_structure_vae
from rocketshp.features import esm3_vqvae, ramachandran_angles
from rocketshp.structure.protein_chain import ProteinChain

MAX_CACHE_SIZE = 500

def _stack_variable_length_tensors(
    sequences: list[torch.Tensor],
    constant_value: int | float = 0,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Automatically stack tensors together, padding variable lengths with the
    value in constant_value. Handles an arbitrary number of dimensions.

    From https://github.com/evolutionaryscale/esm/blob/main/esm/utils/misc.py#L126

    Examples:
        >>> tensor1, tensor2 = torch.ones([2]), torch.ones([5])
        >>> stack_variable_length_tensors(tensor1, tensor2)
        tensor of shape [2, 5]. First row is [1, 1, 0, 0, 0]. Second row is all ones.

        >>> tensor1, tensor2 = torch.ones([2, 4]), torch.ones([5, 3])
        >>> stack_variable_length_tensors(tensor1, tensor2)
        tensor of shape [2, 5, 4]
    """
    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()

    if dtype is None:
        dtype = sequences[0].dtype
    device = sequences[0].device

    array = torch.full(shape, constant_value, dtype=dtype, device=device)
    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array

def _unstack_variable_length_tensors(stacked, lengths) -> list[torch.Tensor]:
    """
    Unstack a variable length tensor given the lengths of the sequences
    """
    return [stacked[i, :lengths[i]] for i in range(len(lengths))]

def _unstack_variable_size_squareforms(stacked, lengths):

    return [stacked[i, :sl, :sl] for i, sl in enumerate(lengths)]

def _get_seq_lengths(sequences: list[torch.Tensor]) -> torch.Tensor:
    return torch.tensor([len(seq) for seq in sequences], dtype=torch.long)

def _dict_collate_fn(batch):
    """
    Custom collate function for batching variable length sequences.
    
    Args:
        batch: List of tuples (features_dict, labels_dict) from the dataset
        
    Returns:
        features_dict: Dictionary of padded feature tensors
        labels_dict: Dictionary of label tensors
        lengths: Tensor of sequence lengths
    """
    # Separate features and labels
    features, labels = zip(*batch)

    # Get sequence lengths (assuming all features have same length)
    lengths = torch.tensor([next(iter(feat.values())).shape[0] for feat in features])

    # Initialize output dictionaries
    padded_features = {}
    padded_labels = {}

    # Pad and stack features
    for key in features[0].keys():
        sequences = [feat[key] for feat in features]
        padded_features[key] = _stack_variable_length_tensors(sequences)

    # Stack labels (assuming they don't need padding)
    for key in labels[0].keys():
        label_tensors = [label[key] for label in labels]
        padded_labels[key] = _stack_variable_length_tensors(label_tensors)

    return padded_features, padded_labels, lengths

def update_h5_dataset(f, dataset_name, data, overwrite: bool = True):
    if dataset_name in f and overwrite:
            del f[dataset_name]
            f.create_dataset(dataset_name, data=data)
    elif dataset_name not in f:
        f.create_dataset(dataset_name, data=data)
    else:
        logger.warning(f"Dataset {dataset_name} already exists in file and overwrite=False.")

class MDDataset(Dataset):
    def __init__(
        self,
        processed_h5: Path,
        seq_features: bool = True,
        struct_features: bool = True,
        struct_stage: str = "quantized"
    ):
        super().__init__()
        self._pdb_file_map = {}
        self._path = processed_h5
        self._handle = h5py.File(self._path, "r")

        self._use_seq = seq_features
        self._use_struct = struct_features
        self.keys = self._get_keys()
        self.samples = self._get_samples()

        self.struct_encoder = get_structure_vae()
        self.struct_stage = struct_stage

    def _get_keys(self):
        raise NotImplementedError

    def _get_samples(self):
        raise NotImplementedError

    def _code_rep_temp(self, key):
        raise NotImplementedError

    def _handle_key(self, pdb_code, rep, temp, ):
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)

    @lru_cache(maxsize=MAX_CACHE_SIZE)
    def __getitem__(self, idx):
        rep_key = self.samples[idx]
        pdb_code, rep, temp = self._code_rep_temp(rep_key)

        features = {}
        if self._use_seq:
            seq_features = self._handle[f"{self._handle_path(pdb_code, rep, temp, False)}/embedding"][:]
            features["seq_feats"] = torch.from_numpy(seq_features)
            features["temp"] = torch.ones(features["seq_feats"].shape[0]) * temp
        if self._use_struct:
            struct_features = self._handle[f"{self._handle_path(pdb_code, rep, temp, False)}/struct_tokens"][:]

            ###### ALTERNATE STRUCT FEATURES HERE #########
            # features["struct_feats"] = torch.from_numpy(struct_features)
            pc = ProteinChain.from_pdb(self._pdb_file_map[pdb_code])
            if self.struct_stage == "ramachandran":
                features["struct_feats"] = ramachandran_angles(pc).squeeze()
            else:
                with torch.inference_mode():
                    struct_features = esm3_vqvae(pc, self.struct_encoder, stage = self.struct_stage).detach().squeeze()
                features["struct_feats"] = struct_features
            ###############################################

            features["temp"] = torch.ones(features["struct_feats"].shape[0]) * temp

        labels = {}
        labels["rmsf"] = torch.from_numpy(self._handle[f"{self._handle_path(pdb_code, rep, temp, True)}/rmsf"][:])
        try:
            labels["ca_dist"] = torch.from_numpy(self._handle[f"{self._handle_path(pdb_code, rep, temp, True)}/ca_distances"][:]).squeeze()
        except KeyError:
            pass
        try:
            labels["dyn_corr"] = torch.from_numpy(self._handle[f"{self._handle_path(pdb_code, rep, temp, True)}/dyn_corr"][:])
        except KeyError:
            pass

        return features, labels

    def __del__(self):
        if hasattr(self, "_handle") and self._handle is not None:# and isinstance(self._handle, h5py._hl.files.File):
            self._handle.close()

class MDDataModule(L.LightningDataModule):
    def __init__(
        self,
        processed_h5: Path,
        seq_features: bool,
        struct_features: bool,
        struct_stage: str,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        train_pct: float,
        val_pct: float,
        clusters_file: Path,
        random_seed: int,
    ):
        super().__init__()

        self._processed_h5 = processed_h5
        self._seq_features = seq_features
        self._struct_features = struct_features
        self._struct_stage = struct_stage

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.train_pct = train_pct
        self.val_pct = val_pct
        self.clusters = pd.read_csv(clusters_file,sep="\t",header=None)

        self.random_seed = random_seed

    def _get_dataset_class(self):
        raise NotImplementedError

    def prepare_data(self):
        pass

    def setup(self, stage: str):

        self.dataset = self._get_dataset_class()(
            self._processed_h5,
            seq_features=self._seq_features,
            struct_features=self._struct_features,
            struct_stage=self._struct_stage
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
            pdb_code, _, _ = self.dataset._code_rep_temp(s)
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
