"""
Compute VQ-VAE-based Structure Heterogeneity Profiles (SHP)

Processes MD trajectory frames through trained VQ-VAE encoder to extract discrete tokens,
then computes per-residue token distributions as SHP (analogous to Foldseek 3Di-based SHP).
"""

import argparse
import gc
import sys
from pathlib import Path

import h5py
import mdtraj as md
import torch
from loguru import logger
from tqdm import tqdm

from planet_md import config
from planet_md.modeling.md_vqvae import MD_VQVAE
from planet_md.trajectory import normalize


def load_trained_vqvae(checkpoint_path: str, device: str = 'cuda') -> MD_VQVAE:
    """
    Load trained VQ-VAE model from checkpoint.

    Args:
        checkpoint_path: Path to Lightning checkpoint file
        device: Device to load model on

    Returns:
        Loaded VQ-VAE model in eval mode
    """
    from scripts.train_vqvae import VQVAELightningModule

    # Load checkpoint
    model = VQVAELightningModule.load_from_checkpoint(checkpoint_path)
    model = model.model  # Extract the actual VQ-VAE model
    model = model.to(device)
    model.eval()

    logger.info(f"Loaded VQ-VAE from {checkpoint_path}")
    logger.info(f"Codebook size: {model.codebook_size}")

    return model


def tokenize_trajectory_frames(
    traj: md.Trajectory,
    vqvae_model: MD_VQVAE,
    device: str = 'cuda',
) -> torch.Tensor:
    """
    Tokenize trajectory frames using VQ-VAE encoder.

    Args:
        traj: MDTraj trajectory (already normalized and subsampled)
        vqvae_model: Trained VQ-VAE model
        device: Device for computation

    Returns:
        Token indices [n_frames, seq_len]
    """
    n_frames = traj.n_frames
    n_residues = len(list(traj.top.residues))

    tokens = []

    with torch.no_grad():
        for frame_idx in range(n_frames):
            # Extract frame
            frame = traj[frame_idx]

            # Get backbone atoms (N, CA, C)
            backbone_indices = frame.top.select("backbone and (name N or name CA or name C)")
            frame_backbone = frame.atom_slice(backbone_indices)

            # Get coordinates and reshape to [1, n_residues, 3, 3]
            coords = frame_backbone.xyz[0]  # [n_atoms, 3]
            coords = coords.reshape(1, n_residues, 3, 3)  # [1, n_residues, 3, 3]
            coords = torch.from_numpy(coords).float().to(device)

            # Create mask (all valid)
            mask = torch.ones(1, n_residues, dtype=torch.bool, device=device)

            # Encode to get token indices
            _, indices, _ = vqvae_model.encode(coords, mask)

            tokens.append(indices[0].cpu())  # [seq_len]

    return torch.stack(tokens)  # [n_frames, seq_len]


def tokens_to_shp(tokens: torch.Tensor, codebook_size: int) -> torch.Tensor:
    """
    Convert token sequences to Structure Heterogeneity Profile.

    Args:
        tokens: Token indices [n_frames, seq_len]
        codebook_size: Size of codebook vocabulary

    Returns:
        SHP [seq_len, codebook_size] - per-residue token distribution
    """
    n_frames, seq_len = tokens.shape

    # Initialize SHP
    shp = torch.zeros(seq_len, codebook_size)

    # Count token occurrences per residue
    for res_idx in range(seq_len):
        res_tokens = tokens[:, res_idx]  # [n_frames]
        counts = torch.bincount(res_tokens, minlength=codebook_size)
        shp[res_idx] = counts.float() / n_frames

    return shp


def compute_vqvae_shp_single(
    pdb_code: str,
    rep: int,
    vqvae_model: MD_VQVAE,
    atlas_dir: Path,
    start: int = 0,
    end: int = None,
    stride: int = 100,
    device: str = 'cuda',
) -> dict:
    """
    Compute VQ-VAE SHP for a single ATLAS trajectory.

    Args:
        pdb_code: PDB code
        rep: Replica number
        vqvae_model: Trained VQ-VAE model
        atlas_dir: Path to ATLAS data directory
        start: Start frame index
        end: End frame index
        stride: Frame stride
        device: Computation device

    Returns:
        Dictionary with SHP and metadata
    """
    logger.info(f"Processing {pdb_code}:R{rep}")

    # Load trajectory
    xtc_f = atlas_dir / pdb_code[:2] / f"{pdb_code}_prod_R{rep}_fit.xtc"
    pdb_f = atlas_dir / pdb_code[:2] / f"{pdb_code}.pdb"

    if not xtc_f.exists() or not pdb_f.exists():
        logger.warning(f"Files not found for {pdb_code}:R{rep}")
        return None

    traj = md.load(str(xtc_f), top=str(pdb_f))

    # Normalize and subsample
    traj = normalize(traj, ca_only=False)
    traj = traj[start:end:stride]

    logger.info(f"Trajectory shape: {traj.n_frames} frames, {traj.n_atoms} atoms")

    # Tokenize frames
    tokens = tokenize_trajectory_frames(traj, vqvae_model, device=device)

    # Convert to SHP
    shp = tokens_to_shp(tokens, vqvae_model.codebook_size)

    # Cleanup
    del traj
    gc.collect()

    return {
        'pdb_code': pdb_code,
        'rep': rep,
        'tokens': tokens,  # [n_frames, seq_len]
        'shp': shp,        # [seq_len, codebook_size]
    }


def save_to_hdf5(result: dict, h5_path: Path, dataset_name: str = 'atlas'):
    """
    Save VQ-VAE SHP to HDF5 file.

    Args:
        result: Result dictionary from compute_vqvae_shp_single
        h5_path: Path to HDF5 file
        dataset_name: Dataset name (atlas or mdcath)
    """
    pdb_code = result['pdb_code']
    rep = result['rep']

    with h5py.File(h5_path, 'a') as f:
        # Create group path: /{pdb_code}/R{rep}/
        group_path = f"{pdb_code}/R{rep}"

        if group_path not in f:
            f.create_group(group_path)

        group = f[group_path]

        # Save SHP
        if 'vqvae_shp' in group:
            del group['vqvae_shp']
        group.create_dataset('vqvae_shp', data=result['shp'].numpy())

        # Optionally save tokens for analysis
        if 'vqvae_tokens' in group:
            del group['vqvae_tokens']
        group.create_dataset('vqvae_tokens', data=result['tokens'].numpy())

        logger.info(f"Saved VQ-VAE SHP for {pdb_code}:R{rep} to {h5_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute VQ-VAE SHP for ATLAS trajectories")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to VQ-VAE checkpoint')
    parser.add_argument('--pdb_code', type=str, required=True, help='PDB code')
    parser.add_argument('--rep', type=int, required=True, help='Replica number')
    parser.add_argument('--output', type=str, required=True, help='Output HDF5 file path')
    parser.add_argument('--stride', type=int, default=100, help='Frame stride')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    args = parser.parse_args()

    # Setup paths
    atlas_dir = config.RAW_DATA_DIR / "atlas"

    # Load VQ-VAE
    vqvae_model = load_trained_vqvae(args.checkpoint, device=args.device)

    # Compute SHP
    result = compute_vqvae_shp_single(
        pdb_code=args.pdb_code,
        rep=args.rep,
        vqvae_model=vqvae_model,
        atlas_dir=atlas_dir,
        stride=args.stride,
        device=args.device,
    )

    if result is not None:
        # Save to HDF5
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_to_hdf5(result, output_path)

        logger.info(f"Successfully computed and saved VQ-VAE SHP for {args.pdb_code}:R{args.rep}")
    else:
        logger.error(f"Failed to process {args.pdb_code}:R{args.rep}")
        sys.exit(1)


if __name__ == '__main__':
    main()
