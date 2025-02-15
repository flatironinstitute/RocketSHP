import tempfile
from io import StringIO
from itertools import combinations

import mdtraj as md
import numpy as np
from statsmodels.tsa.stattools import acf

from rocketshp.structure.protein_chain import ProteinChain


def load_trajectory(filename: str):
    return md.load(filename)

def save_trajectory(traj: md.Trajectory, filename: str):
    traj.save(filename)

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

def compute_contacts(traj: md.Trajectory, scheme: str = "ca", ignore_nonprotein: bool = True, normalized: bool = False, ca_only: bool = True):
    if not normalized:
        traj = normalize(traj, ca_only=ca_only)
    return md.geometry.squareform(
        *md.compute_contacts(
            traj, contacts=list(combinations(np.arange(traj.n_residues), 2)), scheme=scheme, ignore_nonprotein=ignore_nonprotein
            )
        )

def compute_autocorrelation(traj: md.Trajectory, lag: int = 1, atom_indices: list = None, precomputed_contacts: np.ndarray = None, normalized: bool = False, ca_only: bool = True):
    if precomputed_contacts is None:
        contacts = compute_contacts(traj, normalized=normalized, ca_only=ca_only)
    else:
        contacts = precomputed_contacts
    correlations = np.zeros((contacts.shape[1], contacts.shape[1]))
    for (c_i, c_j) in combinations(range(contacts.shape[1]),2):
        corrs_ = acf(contacts[:,c_i,c_j], nlags=contacts.shape[0]-1, fft=True)
        correlations[c_i, c_j] = corrs_[lag]
        correlations[c_j, c_i] = corrs_[lag]
    return correlations

def frame_to_chain(F):
    with tempfile.NamedTemporaryFile(suffix=".pdb") as tmp:
        F.save_pdb(tmp.name)
        tmp.seek(0)
        pdb_content = tmp.read()
        with StringIO(pdb_content.decode()) as sio:
            esmc = ProteinChain.from_pdb(sio)
    return esmc
