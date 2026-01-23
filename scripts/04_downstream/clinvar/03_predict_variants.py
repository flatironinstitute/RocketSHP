#!/usr/bin/env python3
"""
Generate RocketSHP predictions for variant proteins in ClinVar dataset.

This script:
1. Loads the variant dataset CSV
2. For each variant:
   - Uses variant sequence with wild-type structure
   - Generates RocketSHP predictions
   - Builds allosteric networks and calculates centrality
3. Saves all predictions to HDF5 file
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from biotite.structure.io import pdb
from loguru import logger
from omegaconf import OmegaConf
from torch.nn.functional import softmax
from tqdm import tqdm

from rocketshp import RocketSHP, load_sequence, load_structure
from rocketshp.network import build_allosteric_network, calculate_centrality, cluster_network


def load_config(config_path: str):
    """Load configuration file."""
    cfg = OmegaConf.load(config_path)
    return OmegaConf.to_container(cfg, resolve=True)


def predict_variant(
    variant_id: str,
    variant_sequence: str,
    pdb_path: Path,
    model: RocketSHP,
    device: torch.device,
    cfg: dict,
) -> dict:
    """
    Generate predictions for a variant sequence using WT structure.

    Args:
        variant_id: Variant identifier
        variant_sequence: Mutated protein sequence
        pdb_path: Path to wild-type AlphaFold PDB structure
        model: Loaded RocketSHP model
        device: Torch device
        cfg: Configuration dictionary

    Returns:
        Dictionary with predictions and network analysis
    """
    try:
        # Load WT structure
        structure = pdb.PDBFile.read(pdb_path).get_structure()
        struct_features = load_structure(structure, device=device)

        # Use variant sequence
        seq_features = load_sequence(variant_sequence, device=device)

        # Verify sequence and structure lengths match (critical for SNVs)
        # Structure length from number of CA atoms
        ca_count = sum(1 for atom in structure if atom.atom_name == "CA")
        if len(variant_sequence) != ca_count:
            logger.error(
                f"Length mismatch for {variant_id}: "
                f"sequence={len(variant_sequence)}, structure={ca_count}"
            )
            return None

        # Run prediction
        with torch.inference_mode():
            dynamics_pred = model({
                "seq_feats": seq_features,
                "struct_feats": struct_features,
            })

        # Extract predictions
        rmsf = dynamics_pred["rmsf"].squeeze().cpu().numpy()
        gcc_lmi = dynamics_pred["gcc_lmi"].squeeze().cpu().numpy()
        ca_dist = dynamics_pred["ca_dist"].squeeze().cpu().numpy()
        shp = softmax(dynamics_pred["shp"].squeeze(), dim=1).cpu().numpy()

        # Build allosteric network
        try:
            network = build_allosteric_network(
                gcc_lmi,
                ca_dist,
                distance_cutoff=cfg["distance_cutoff"]
            )

            # Calculate centrality metrics
            centrality = calculate_centrality(
                network,
                do_betweenness=True,
                do_closeness=True,
                do_degree=True
            )

            # Detect communities
            try:
                communities = cluster_network(network, k=cfg["num_communities"])
            except (nx.NetworkXError, StopIteration, ValueError):
                # Network too small/disconnected for clustering
                logger.warning(f"Network clustering failed for {variant_id}, using single community")
                communities = (tuple(range(network.number_of_nodes())),)

            # Convert communities to node labels
            community_labels = np.zeros(len(variant_sequence), dtype=int)
            for comm_id, community in enumerate(communities):
                for node in community:
                    community_labels[node] = comm_id

            # Network statistics
            network_stats = {
                "num_nodes": network.number_of_nodes(),
                "num_edges": network.number_of_edges(),
                "num_communities": len(communities),
                "avg_degree": np.mean([d for _, d in network.degree()]),
            }

        except Exception as e:
            logger.warning(f"Network analysis failed for {variant_id}: {e}")
            # Set default values if network analysis fails
            centrality = {
                "betweenness": np.zeros(len(variant_sequence)),
                "closeness": np.zeros(len(variant_sequence)),
                "degree": np.zeros(len(variant_sequence)),
            }
            community_labels = np.zeros(len(variant_sequence), dtype=int)
            network_stats = {}

        # Free GPU memory
        torch.cuda.empty_cache()

        return {
            "variant_id": variant_id,
            "sequence": variant_sequence,
            "rmsf": rmsf,
            "gcc_lmi": gcc_lmi,
            "ca_dist": ca_dist,
            "shp": shp,
            "centrality_betweenness": centrality["betweenness"],
            "centrality_closeness": centrality["closeness"],
            "centrality_degree": centrality["degree"],
            "communities": community_labels,
            "network_stats": network_stats,
        }

    except Exception as e:
        logger.error(f"Error predicting variant {variant_id}: {e}")
        return None


def save_predictions_hdf5(predictions: list, output_path: Path):
    """
    Save variant predictions to HDF5 file.

    Args:
        predictions: List of prediction dictionaries
        output_path: Output HDF5 file path
    """
    with h5py.File(output_path, 'w') as f:
        for pred in predictions:
            if pred is None:
                continue

            variant_id = pred["variant_id"]
            # Replace special characters in variant_id for HDF5 group name
            group_name = variant_id.replace("/", "_").replace(":", "_")
            grp = f.create_group(group_name)

            # Save sequence as attribute
            grp.attrs["sequence"] = pred["sequence"]
            grp.attrs["variant_id"] = variant_id

            # Save arrays
            for key in ["rmsf", "gcc_lmi", "ca_dist", "shp",
                        "centrality_betweenness", "centrality_closeness",
                        "centrality_degree", "communities"]:
                if key in pred and pred[key] is not None:
                    grp.create_dataset(key, data=pred[key], compression="gzip")

            # Save network stats as attributes
            if pred["network_stats"]:
                for stat_key, stat_val in pred["network_stats"].items():
                    # Convert numpy types to Python types for HDF5 compatibility
                    if isinstance(stat_val, (np.integer, np.floating)):
                        stat_val = stat_val.item()
                    grp.attrs[stat_key] = stat_val

    logger.info(f"Saved variant predictions to {output_path}")


def load_checkpoint(checkpoint_path: Path) -> set:
    """Load set of already processed variant IDs from checkpoint."""
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            return set(line.strip() for line in f)
    return set()


def save_checkpoint(checkpoint_path: Path, processed_ids: set):
    """Save processed variant IDs to checkpoint file."""
    with open(checkpoint_path, 'w') as f:
        for variant_id in sorted(processed_ids):
            f.write(f"{variant_id}\n")


def main(config_path: str = "configs/clinvar_analysis_config.yml"):
    """Main function to predict variant proteins."""

    # Load configuration
    cfg = load_config(config_path)

    # Setup logging
    logger.add(cfg["log_file"], level=cfg["log_level"])
    logger.info("Starting variant predictions")

    # Load variant dataset
    dataset_path = Path(cfg["output_dir"]) / "variant_dataset.csv"
    if not dataset_path.exists():
        logger.error(f"Variant dataset not found: {dataset_path}")
        logger.error("Please run 01_prepare_clinvar_dataset.py first")
        return

    df = pd.read_csv(dataset_path)
    logger.info(f"Loaded {len(df)} variants from dataset")

    # Setup checkpointing
    checkpoint_path = Path(cfg["output_dir"]) / "variant_predictions_checkpoint.txt"
    processed_ids = load_checkpoint(checkpoint_path)
    logger.info(f"Loaded checkpoint with {len(processed_ids)} already processed variants")

    # Filter out already processed
    df = df[~df["variant_id"].isin(processed_ids)]
    logger.info(f"Remaining variants to process: {len(df)}")

    if len(df) == 0:
        logger.info("All variants already processed!")
        return

    # Load model
    logger.info(f"Loading RocketSHP model: {cfg['model_checkpoint']}")
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    model = RocketSHP.load_from_checkpoint(cfg["model_checkpoint"]).to(device)
    logger.info(f"Model loaded on {device}")

    # Process variants
    predictions = []
    checkpoint_counter = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting variants"):
        variant_id = row["variant_id"]
        variant_sequence = row["variant_sequence"]
        pdb_path = Path(row["pdb_path"])

        if not pdb_path.exists():
            logger.warning(f"PDB file not found for {variant_id}: {pdb_path}")
            continue

        # Generate prediction
        pred = predict_variant(
            variant_id,
            variant_sequence,
            pdb_path,
            model,
            device,
            cfg
        )

        if pred is not None:
            predictions.append(pred)
            processed_ids.add(variant_id)
            checkpoint_counter += 1

            # Save checkpoint periodically
            if checkpoint_counter >= cfg["checkpoint_interval"]:
                save_checkpoint(checkpoint_path, processed_ids)
                checkpoint_counter = 0
                logger.info(f"Checkpoint saved: {len(processed_ids)} variants processed")

    # Final checkpoint save
    save_checkpoint(checkpoint_path, processed_ids)

    # Save all predictions to HDF5
    if predictions:
        output_path = Path(cfg["output_dir"]) / "variant_predictions.h5"
        save_predictions_hdf5(predictions, output_path)

        logger.info("\n=== Prediction Summary ===")
        logger.info(f"Total variants predicted: {len(predictions)}")
        logger.info(f"Predictions saved to: {output_path}")

        # Log pathogenicity distribution
        pathogenicity_counts = df[df["variant_id"].isin(processed_ids)]["pathogenicity"].value_counts()
        logger.info(f"\nPathogenicity distribution:")
        for category, count in pathogenicity_counts.items():
            logger.info(f"  {category}: {count}")
    else:
        logger.warning("No predictions generated")


if __name__ == "__main__":
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/clinvar_analysis_config.yml"
    main(config_path)
