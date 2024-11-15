from pathlib import Path

import numpy as np
import h5py
import torch
import typer
from loguru import logger
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from rocketshp.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

ATLAS_PROCESSED_H5 = PROCESSED_DATA_DIR / "atlas/atlas_processed.h5"
ATLAS_N_REPS = 3

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

def _get_seq_lengths(sequences: list[torch.Tensor]) -> torch.Tensor:
    return torch.tensor([len(seq) for seq in sequences], dtype=torch.long)

def _collate_fn_single(batch):
    features, labels = zip(*batch)
    features = list(zip(*features))
    lengths = _get_seq_lengths(features[0])
    return _stack_variable_length_tensors(features[0]), lengths, _stack_variable_length_tensors(labels)


def _collate_fn_multi(batch):
    features, labels = zip(*batch)
    features = list(zip(*features))
    lengths = _get_seq_lengths(features[0])
    features = [_stack_variable_length_tensors(f) for f in features]
    return features, lengths, _stack_variable_length_tensors(labels)


def _collate_fn(batch):
    f0, _ = batch[0]
    if len(f0) == 1:
        return _collate_fn_single(batch)
    else:
        return _collate_fn_multi(batch)


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
        self.samples = [f"{k}/R{i+1}" for k in self.keys for i in range(ATLAS_N_REPS)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rep_key = self.samples[idx]
        pdb_code, rep = rep_key.split("/")

        features = []
        if self._use_seq:
            seq_features = self._handle[f"{pdb_code}/embedding"][:]
            features.append(torch.from_numpy(seq_features))
        if self._use_struct:
            struct_features = self._handle[f"{pdb_code}/struct_tokens"][:]
            features.append(torch.from_numpy(struct_features))

        label = torch.from_numpy(self._handle[f"{pdb_code}/{rep}/{self._target}"][:])

        return features, label

    def __del__(self):
        if hasattr(self, "_handle") and isinstance(self._handle, h5py.File):
            self._handle.close()


class ATLASDataModule(LightningDataModule):
    def __init__(
        self,
        processed_h5: Path,
        target: str = "rmsf",
        seq_features: bool = True,
        struct_features: bool = True,
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 0,
        train_pct: float = 0.8,
        val_pct: float = 0.1,
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

        pdb_codes = list(set([s.split("/")[0] for s in self.dataset.samples]))
        n_pdbs = len(pdb_codes)

        train_size = int(self.train_pct * n_pdbs)
        val_size = int(self.val_pct * n_pdbs)

        rng = np.random.default_rng(self.random_seed)
        pdb_shuffled = rng.permutation(pdb_codes)
        train_pdb_codes = pdb_shuffled[:train_size]
        val_pdb_codes = pdb_shuffled[train_size : train_size + val_size]
        test_pdb_codes = pdb_shuffled[train_size + val_size :]

        train_sample_idx = [i for i, s in enumerate(self.dataset.samples) if s.split("/")[0] in train_pdb_codes]
        val_sample_idx = [i for i, s in enumerate(self.dataset.samples) if s.split("/")[0] in val_pdb_codes]
        test_sample_idx = [i for i, s in enumerate(self.dataset.samples) if s.split("/")[0] in test_pdb_codes]

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
            collate_fn=_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_collate_fn,
        )


@app.command()
def main(
    # ---- ARGUMENTS ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- CODE ----
    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
