#!/usr/bin/env python3
"""
Extract features from WT and variant predictions for statistical analysis.

This script:
1. Loads variant dataset CSV
2. Loads WT and variant predictions from HDF5 files
3. Extracts per-position and global features
4. Computes deltas between WT and variant
5. Adds covariates (protein length, position, pLDDT, AA properties)
6. Creates analysis-ready CSV
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from biotite.structure.io import pdb
from loguru import logger
from omegaconf import OmegaConf
from sklearn.metrics import normalized_mutual_info_score
from tqdm import tqdm


# Amino acid properties for covariate analysis
AA_PROPERTIES = {
    # Hydrophobicity scale (Kyte-Doolittle)
    'hydrophobicity': {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
        'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
        'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
        'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
    },
    # Charge
    'charge': {
        'R': 1, 'K': 1, 'D': -1, 'E': -1,
        **{aa: 0 for aa in 'ACFGHILMNPQSTVWY'}
    },
    # Volume (approximate in Ų)
    'volume': {
        'A': 88.6, 'R': 173.4, 'N': 114.1, 'D': 111.1, 'C': 108.5,
        'Q': 143.8, 'E': 138.4, 'G': 60.1, 'H': 153.2, 'I': 166.7,
        'L': 166.7, 'K': 168.6, 'M': 162.9, 'F': 189.9, 'P': 112.7,
        'S': 89.0, 'T': 116.1, 'W': 227.8, 'Y': 193.6, 'V': 140.0,
    },
}


def load_config(config_path: str):
    """Load configuration file."""
    cfg = OmegaConf.load(config_path)
    return OmegaConf.to_container(cfg, resolve=True)


def get_plddt_at_position(pdb_path: Path, position: int) -> float:
    """
    Extract pLDDT score at a specific position from AlphaFold PDB.

    Args:
        pdb_path: Path to PDB file
        position: Residue position (1-indexed)

    Returns:
        pLDDT score at position
    """
    try:
        pdb_file = pdb.PDBFile.read(pdb_path)
        structure = pdb_file.get_structure()

        # Get B-factors (pLDDT scores) for CA atoms
        ca_mask = structure.atom_name == "CA"
        b_factors = structure.b_factor[ca_mask]
        res_ids = structure.res_id[ca_mask]

        # Get unique residues
        unique_res_ids = np.unique(res_ids)

        if position - 1 < len(unique_res_ids):
            # Get pLDDT for this position
            target_res_id = unique_res_ids[position - 1]
            plddt = b_factors[res_ids == target_res_id].mean()
            return float(plddt)
        else:
            return np.nan

    except Exception as e:
        logger.debug(f"Error extracting pLDDT at position {position}: {e}")
        return np.nan


def get_aa_property_change(wt_aa: str, var_aa: str, property_name: str) -> float:
    """Calculate change in amino acid property."""
    try:
        wt_prop = AA_PROPERTIES[property_name].get(wt_aa, 0)
        var_prop = AA_PROPERTIES[property_name].get(var_aa, 0)
        return var_prop - wt_prop
    except:
        return np.nan


def community_to_labels(communities, num_nodes: int) -> np.ndarray:
    """Convert community tuples to node labels array."""
    labels = np.zeros(num_nodes, dtype=int)
    for comm_id, community in enumerate(communities):
        for node in community:
            labels[node] = comm_id
    return labels


def extract_variant_features(
    variant_row: pd.Series,
    wt_h5: h5py.File,
    var_h5: h5py.File,
) -> dict:
    """
    Extract all features for a single variant.

    Args:
        variant_row: Row from variant dataset
        wt_h5: HDF5 file with WT predictions
        var_h5: HDF5 file with variant predictions

    Returns:
        Dictionary with extracted features
    """
    variant_id = variant_row["variant_id"]
    uniprot_id = variant_row["uniprot_id"]
    protein_pos = int(variant_row["protein_pos"]) - 1  # Convert to 0-indexed
    wt_aa = variant_row["wt_aa"]
    var_aa = variant_row["variant_aa"]
    protein_length = int(variant_row["protein_length"])

    features = {
        "variant_id": variant_id,
        "uniprot_id": uniprot_id,
        "gene_name": variant_row["gene_name"],
        "protein_pos": protein_pos + 1,  # Keep as 1-indexed in output
        "wt_aa": wt_aa,
        "variant_aa": var_aa,
        "pathogenicity": variant_row["pathogenicity"],
        "protein_length": protein_length,
    }

    try:
        # Load WT predictions
        if uniprot_id not in wt_h5:
            logger.warning(f"WT prediction not found for {uniprot_id}")
            return None

        wt_grp = wt_h5[uniprot_id]
        wt_rmsf = wt_grp["rmsf"][:]
        wt_betweenness = wt_grp["centrality_betweenness"][:]
        wt_closeness = wt_grp["centrality_closeness"][:]
        wt_degree = wt_grp["centrality_degree"][:]
        wt_communities = wt_grp["communities"][:]

        # Load variant predictions
        variant_group_name = variant_id.replace("/", "_").replace(":", "_")
        if variant_group_name not in var_h5:
            logger.warning(f"Variant prediction not found for {variant_id}")
            return None

        var_grp = var_h5[variant_group_name]
        var_rmsf = var_grp["rmsf"][:]
        var_betweenness = var_grp["centrality_betweenness"][:]
        var_closeness = var_grp["centrality_closeness"][:]
        var_degree = var_grp["centrality_degree"][:]
        var_communities = var_grp["communities"][:]

        # Extract per-position features
        if protein_pos < len(wt_rmsf) and protein_pos < len(var_rmsf):
            features["wt_rmsf"] = float(wt_rmsf[protein_pos])
            features["variant_rmsf"] = float(var_rmsf[protein_pos])
            features["delta_rmsf"] = features["variant_rmsf"] - features["wt_rmsf"]

            features["wt_betweenness"] = float(wt_betweenness[protein_pos])
            features["variant_betweenness"] = float(var_betweenness[protein_pos])
            features["delta_betweenness"] = features["variant_betweenness"] - features["wt_betweenness"]

            features["wt_closeness"] = float(wt_closeness[protein_pos])
            features["variant_closeness"] = float(var_closeness[protein_pos])
            features["delta_closeness"] = features["variant_closeness"] - features["wt_closeness"]

            features["wt_degree"] = float(wt_degree[protein_pos])
            features["variant_degree"] = float(var_degree[protein_pos])
            features["delta_degree"] = features["variant_degree"] - features["wt_degree"]
        else:
            logger.warning(f"Position {protein_pos} out of bounds for {variant_id}")
            return None

        # Extract global features
        features["wt_mean_rmsf"] = float(np.mean(wt_rmsf))
        features["variant_mean_rmsf"] = float(np.mean(var_rmsf))
        features["delta_mean_rmsf"] = features["variant_mean_rmsf"] - features["wt_mean_rmsf"]

        features["wt_std_rmsf"] = float(np.std(wt_rmsf))
        features["variant_std_rmsf"] = float(np.std(var_rmsf))
        features["delta_std_rmsf"] = features["variant_std_rmsf"] - features["wt_std_rmsf"]

        # Network topology features
        wt_num_communities = int(wt_grp.attrs.get("num_communities", 0))
        var_num_communities = int(var_grp.attrs.get("num_communities", 0))
        features["wt_num_communities"] = wt_num_communities
        features["variant_num_communities"] = var_num_communities
        features["delta_num_communities"] = var_num_communities - wt_num_communities

        wt_num_edges = int(wt_grp.attrs.get("num_edges", 0))
        var_num_edges = int(var_grp.attrs.get("num_edges", 0))
        features["wt_num_edges"] = wt_num_edges
        features["variant_num_edges"] = var_num_edges
        features["delta_num_edges"] = var_num_edges - wt_num_edges

        # Community similarity (normalized mutual information)
        if len(wt_communities) == len(var_communities):
            try:
                features["community_similarity"] = float(
                    normalized_mutual_info_score(wt_communities, var_communities)
                )
            except:
                features["community_similarity"] = np.nan
        else:
            features["community_similarity"] = np.nan

        # Covariates
        features["relative_position"] = (protein_pos + 1) / protein_length
        features["plddt_avg"] = float(variant_row.get("plddt_avg", np.nan))

        # Get local pLDDT at variant position
        if "pdb_path" in variant_row and pd.notna(variant_row["pdb_path"]):
            features["local_plddt"] = get_plddt_at_position(
                Path(variant_row["pdb_path"]),
                protein_pos + 1
            )
        else:
            features["local_plddt"] = np.nan

        # Amino acid property changes
        features["delta_hydrophobicity"] = get_aa_property_change(wt_aa, var_aa, "hydrophobicity")
        features["delta_charge"] = get_aa_property_change(wt_aa, var_aa, "charge")
        features["delta_volume"] = get_aa_property_change(wt_aa, var_aa, "volume")

        return features

    except Exception as e:
        logger.error(f"Error extracting features for {variant_id}: {e}")
        return None


def main(config_path: str = "configs/clinvar_analysis_config.yml"):
    """Main function to extract features."""

    # Load configuration
    cfg = load_config(config_path)

    # Setup logging
    logger.add(cfg["log_file"], level=cfg["log_level"])
    logger.info("Starting feature extraction")

    # Load variant dataset
    dataset_path = Path(cfg["output_dir"]) / "variant_dataset.csv"
    if not dataset_path.exists():
        logger.error(f"Variant dataset not found: {dataset_path}")
        return

    df = pd.read_csv(dataset_path)
    logger.info(f"Loaded {len(df)} variants")

    # Load prediction HDF5 files
    wt_pred_path = Path(cfg["output_dir"]) / "wildtype_predictions.h5"
    var_pred_path = Path(cfg["output_dir"]) / "variant_predictions.h5"

    if not wt_pred_path.exists():
        logger.error(f"WT predictions not found: {wt_pred_path}")
        logger.error("Please run 02_predict_wildtype.py first")
        return

    if not var_pred_path.exists():
        logger.error(f"Variant predictions not found: {var_pred_path}")
        logger.error("Please run 03_predict_variants.py first")
        return

    logger.info("Loading prediction files...")
    with h5py.File(wt_pred_path, 'r') as wt_h5, \
         h5py.File(var_pred_path, 'r') as var_h5:

        logger.info(f"WT predictions available for {len(wt_h5.keys())} proteins")
        logger.info(f"Variant predictions available for {len(var_h5.keys())} variants")

        # Extract features for each variant
        all_features = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
            features = extract_variant_features(row, wt_h5, var_h5)
            if features is not None:
                all_features.append(features)

    # Create DataFrame
    if all_features:
        features_df = pd.DataFrame(all_features)

        # Save to CSV
        output_path = Path(cfg["output_dir"]) / "variant_features.csv"
        features_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(features_df)} variant features to {output_path}")

        # Print summary
        logger.info("\n=== Feature Extraction Summary ===")
        logger.info(f"Total variants with features: {len(features_df)}")
        logger.info(f"\nPathogenicity distribution:")
        logger.info(features_df["pathogenicity"].value_counts())

        logger.info(f"\nFeature columns: {len(features_df.columns)}")
        logger.info(f"Missing values per column:")
        missing = features_df.isnull().sum()
        logger.info(missing[missing > 0])

        return features_df
    else:
        logger.error("No features extracted")
        return None


if __name__ == "__main__":
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/clinvar_analysis_config.yml"
    main(config_path)
