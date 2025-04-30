import tempfile
from io import StringIO
from itertools import combinations
from pathlib import Path

import mdtraj as md
import numpy as np
import nglview as nv
import torch
import biotite.structure as bs
from biotite.structure.alphabet import to_3di
from statsmodels.tsa.stattools import acf

from rocketshp.structure.protein_chain import ProteinChain


def load_trajectory(filename: str):
    return md.load(filename)


def save_trajectory(traj: md.Trajectory, filename: str):
    traj.save(filename)


def value_to_hex(value, cmap="red-blue"):
    """Map value between 0 and 1 to hex color code"""
    # Ensure value is between 0 and 1
    value = max(0, min(1, value))

    if cmap == "red-blue":
        # Red (value=1) to Blue (value=0)
        red = int(255 * value)
        blue = int(255 * (1 - value))
        return f"#{red:02x}00{blue:02x}"

    elif cmap == "blue-red":
        # Blue (value=1) to Red (value=0)
        red = int(255 * (1 - value))
        blue = int(255 * value)
        return f"#{red:02x}00{blue:02x}"


def display_trajectory(
    traj, coloring="residueindex", bfactor=None, normalize=True, RMAX=1
):
    view = nv.show_mdtraj(traj)
    view.clear()

    if coloring in nv.color.COLOR_SCHEMES and coloring != "bfactor":
        view.add_representation("cartoon", colorScheme=coloring)
        return view
    elif coloring == "bfactor":
        assert bfactor is not None
        if not isinstance(bfactor, np.ndarray):
            bfactor = np.array(bfactor)
        if normalize:
            RMAX = max(0.3, bfactor.min())
            denom = bfactor.clip(0, RMAX).max() - bfactor.clip(0, bfactor.max()).min()
            if not denom:
                denom = 1
            bfactor_new = (bfactor.clip(0, RMAX) - bfactor.clip(0, RMAX).min()) / (
                denom
            )
        else:
            bfactor_new = bfactor

        view.add_representation("cartoon", colorScheme=coloring)

        def _set_color_by_residue(self, colors, component_index=0, repr_index=0):
            self._remote_call(
                "setColorByResidue",
                target="Widget",
                args=[colors, component_index, repr_index],
            )

        scheme = [value_to_hex(x).upper().replace("#", "0x") for x in bfactor_new]
        _set_color_by_residue(view, scheme)

        return view  # , scheme, bfactor_new
    else:
        raise ValueError(
            f"Coloring scheme {coloring} not supported: valid options {nv.color.COLOR_SCHEMES}"
        )

def normalize(traj: md.Trajectory, ca_only: bool = True):
    traj.superpose(traj)
    traj.center_coordinates()
    if ca_only:
        atom_indices = traj.top.select("name CA")
        return traj.atom_slice(atom_indices)
    else:
        return traj


def compute_rmsf(traj: md.Trajectory, normalized: bool = False, ca_only: bool = True):
    if not normalized:
        traj = normalize(traj, ca_only=ca_only)
    if ca_only:
        atom_indices = traj.top.select("name CA")
    else:
        atom_indices = None
    return md.rmsf(traj, traj, 0, atom_indices=atom_indices)


def compute_contacts(
    traj: md.Trajectory,
    scheme: str = "ca",
    ignore_nonprotein: bool = True,
    normalized: bool = False,
    ca_only: bool = True,
):
    if not normalized:
        traj = normalize(traj, ca_only=ca_only)
    return md.geometry.squareform(
        *md.compute_contacts(
            traj,
            contacts=list(combinations(np.arange(traj.n_residues), 2)),
            scheme=scheme,
            ignore_nonprotein=ignore_nonprotein,
        )
    )

def compute_generalized_correlation_lmi(
    top_file: str | Path,
    traj_file: str | Path,
    stride: int = 1,
    verbose: bool = False,
):
    from mdigest.core.parsetrajectory import MDS
    from mdigest.core.correlation import DynCorr

    mds = MDS()
    mds.set_num_replicas(1)
    mds.load_system(top_file, traj_file)
    mds.align_traj(selection='name CA')
    mds.set_selection('protein and name CA', 'protein')
    mds.stride_trajectory(initial=0, final=-1, step=stride)

    dyncorr = DynCorr(mds)
    dyncorr.parse_dynamics(scale=True, normalize=True, LMI="gaussian", MI="None", CENTRALITY=False, VERBOSE=verbose)
    return dyncorr.gcc_allreplicas["rep_0"]["gcc_lmi"]

def compute_autocorrelation_DEPRECATED(
    traj: md.Trajectory,
    lag: int = 1,
    atom_indices: list = None,
    precomputed_contacts: np.ndarray = None,
    normalized: bool = False,
    ca_only: bool = True,
):
    if precomputed_contacts is None:
        contacts = compute_contacts(traj, normalized=normalized, ca_only=ca_only)
    else:
        contacts = precomputed_contacts
    correlations = np.zeros((contacts.shape[1], contacts.shape[1]))
    for c_i, c_j in combinations(range(contacts.shape[1]), 2):
        corrs_ = acf(contacts[:, c_i, c_j], nlags=contacts.shape[0] - 1, fft=True)
        correlations[c_i, c_j] = corrs_[lag]
        correlations[c_j, c_i] = corrs_[lag]
    return correlations

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

def seq_list_to_tensor(seq_list):
    max_len = max([len(i) for i in seq_list])
    seq_tensor = torch.zeros(len(seq_list), max_len, dtype=torch.long)
    for i, seq in enumerate(seq_list):
        exploded = [FS_3DI_LIST.index(j) for j in seq]
        seq_tensor[i] = torch.tensor(exploded)
    return seq_tensor

def convert_to_normalized_shp(preshp, max_dim=20):
        preshp = torch.tensor(preshp).squeeze()
        shp = torch.stack([torch.bincount(i, minlength=max_dim) for i in preshp.T])
        shp = shp.T / shp.sum(axis=1)
        return shp.T

def compute_shp(aa_stack: bs.AtomArrayStack, start=0, stop=None, stride=1):
    """
    Compute SHP from a stack of atoms
    """
    seqs_3di = []
    if stop is None: stop = len(aa_stack)
    for i in range(start, stop, stride):
        structure = aa_stack[i]
        i3d = str(to_3di(structure)[0][0]).upper()
        seqs_3di.append(i3d)
    preshp = seq_list_to_tensor(seqs_3di)
    shp = convert_to_normalized_shp(preshp)

    return shp

def frame_to_chain(F):
    with tempfile.NamedTemporaryFile(suffix=".pdb") as tmp:
        F.save_pdb(tmp.name)
        tmp.seek(0)
        pdb_content = tmp.read()
        with StringIO(pdb_content.decode()) as sio:
            esmc = ProteinChain.from_pdb(sio)
    return esmc
