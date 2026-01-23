#!/usr/bin/env python3
"""
Visualize ClinVar pathogenicity analysis results.

This script creates:
1. Distribution plots (violin/box plots) for pathogenic vs benign vs VUS
2. ROC curves for classification performance
3. Feature importance from logistic regression
4. Network topology example comparisons
5. Summary tables
"""

import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib.gridspec import GridSpec
from omegaconf import OmegaConf

from rocketshp.network import build_allosteric_network


def load_config(config_path: str):
    """Load configuration file."""
    cfg = OmegaConf.load(config_path)
    return OmegaConf.to_container(cfg, resolve=True)


def plot_distributions(df: pd.DataFrame, output_dir: Path):
    """
    Create violin plots showing distributions across pathogenicity categories.

    Args:
        df: DataFrame with variant features
        output_dir: Directory to save figures
    """
    features = [
        ("delta_rmsf", "Δ RMSF"),
        ("delta_betweenness", "Δ Betweenness Centrality"),
        ("delta_closeness", "Δ Closeness Centrality"),
        ("delta_degree", "Δ Degree Centrality"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Color palette
    palette = {"Pathogenic": "#d62728", "Benign": "#2ca02c", "VUS": "#ff7f0e"}

    for idx, (feature, label) in enumerate(features):
        ax = axes[idx]

        # Filter out NaNs
        plot_df = df[df[feature].notna()]

        # Violin plot
        sns.violinplot(
            data=plot_df,
            x="pathogenicity",
            y=feature,
            palette=palette,
            ax=ax,
            order=["Pathogenic", "VUS", "Benign"]
        )

        # Add box plot overlay
        sns.boxplot(
            data=plot_df,
            x="pathogenicity",
            y=feature,
            palette=palette,
            ax=ax,
            order=["Pathogenic", "VUS", "Benign"],
            width=0.3,
            boxprops=dict(alpha=0.7),
            showfliers=False
        )

        ax.set_xlabel("Pathogenicity", fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(label, fontsize=14, fontweight='bold')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

        # Add sample sizes
        for i, category in enumerate(["Pathogenic", "VUS", "Benign"]):
            n = len(plot_df[plot_df["pathogenicity"] == category])
            ax.text(i, ax.get_ylim()[1] * 0.9, f"n={n}", ha='center', fontsize=10)

    plt.tight_layout()
    output_path = output_dir / "distributions.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved distribution plot to {output_path}")


def plot_roc_curves(results: dict, output_dir: Path):
    """
    Create ROC curves for different features and combined model.

    Args:
        results: Dictionary with ROC analysis results
        output_dir: Directory to save figures
    """
    if not results:
        logger.warning("No ROC results to plot")
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    # Color map
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for idx, (feature, data) in enumerate(results.items()):
        if "fpr" in data and "tpr" in data:
            label = f"{feature.replace('_', ' ').title()} (AUC={data['auc']:.3f})"
            ax.plot(
                data["fpr"],
                data["tpr"],
                color=colors[idx],
                label=label,
                linewidth=2
            )

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.500)', linewidth=1)

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves: Pathogenic vs Benign Classification', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    output_path = output_dir / "roc_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved ROC curves to {output_path}")


def plot_feature_importance(results: dict, output_dir: Path):
    """
    Create bar plot of logistic regression feature importance.

    Args:
        results: Dictionary with logistic regression results
        output_dir: Directory to save figures
    """
    if not results or "sorted_coefficients" not in results:
        logger.warning("No logistic regression results to plot")
        return

    # Get top features
    sorted_coefs = results["sorted_coefficients"][:10]  # Top 10
    features = [x[0].replace("_", " ").title() for x in sorted_coefs]
    coefficients = [x[1] for x in sorted_coefs]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#d62728' if c > 0 else '#2ca02c' for c in coefficients]

    bars = ax.barh(features, coefficients, color=colors, alpha=0.7, edgecolor='black')

    ax.set_xlabel('Coefficient Value', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title(
        f'Logistic Regression Feature Importance\n(CV AUC: {results["cv_auc_mean"]:.3f} ± {results["cv_auc_std"]:.3f})',
        fontsize=14,
        fontweight='bold'
    )
    ax.axvline(0, color='black', linewidth=1)
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for bar, coef in zip(bars, coefficients):
        width = bar.get_width()
        ax.text(
            width + (0.01 if width > 0 else -0.01),
            bar.get_y() + bar.get_height() / 2,
            f'{coef:.3f}',
            ha='left' if width > 0 else 'right',
            va='center',
            fontsize=10
        )

    plt.tight_layout()
    output_path = output_dir / "feature_importance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved feature importance plot to {output_path}")


def plot_summary_table(mw_results: dict, output_dir: Path):
    """
    Create summary table of Mann-Whitney test results.

    Args:
        mw_results: Dictionary with Mann-Whitney test results
        output_dir: Directory to save figures
    """
    # Extract significant results
    rows = []

    for feature, comparisons in mw_results.items():
        for comparison, results in comparisons.items():
            if results.get("significant", False):
                rows.append({
                    "Feature": feature.replace("_", " ").title(),
                    "Comparison": comparison.replace("_", " "),
                    "P-value": f"{results['p_value']:.2e}",
                    "P-corrected": f"{results['p_corrected']:.2e}",
                    "Effect Size": f"{results['effect_size']:.3f}",
                    "N1": results['n1'],
                    "N2": results['n2'],
                })

    if not rows:
        logger.warning("No significant results to tabulate")
        return

    df_table = pd.DataFrame(rows)

    # Create figure with table
    fig, ax = plt.subplots(figsize=(12, len(df_table) * 0.4 + 1))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(
        cellText=df_table.values,
        colLabels=df_table.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.2, 0.2, 0.12, 0.12, 0.12, 0.08, 0.08]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Style header
    for i in range(len(df_table.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(df_table) + 1):
        for j in range(len(df_table.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    plt.title('Significant Mann-Whitney U Test Results (FDR < 0.05)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    output_path = output_dir / "summary_table.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved summary table to {output_path}")


def plot_network_example(
    variant_id: str,
    wt_data: dict,
    var_data: dict,
    variant_pos: int,
    output_path: Path,
    cfg: dict
):
    """
    Plot side-by-side WT vs variant network visualization.

    Args:
        variant_id: Variant identifier
        wt_data: WT prediction data
        var_data: Variant prediction data
        variant_pos: Position of variant (0-indexed)
        output_path: Output file path
        cfg: Configuration dictionary
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Build networks
    wt_network = build_allosteric_network(
        wt_data["gcc_lmi"],
        wt_data["ca_dist"],
        distance_cutoff=cfg["distance_cutoff"]
    )

    var_network = build_allosteric_network(
        var_data["gcc_lmi"],
        var_data["ca_dist"],
        distance_cutoff=cfg["distance_cutoff"]
    )

    # Layout using spring layout (same for both for comparison)
    pos = nx.spring_layout(wt_network, seed=42)

    # Node colors based on centrality
    wt_betweenness = wt_data["centrality_betweenness"]
    var_betweenness = var_data["centrality_betweenness"]

    # Normalize for coloring
    vmin = min(wt_betweenness.min(), var_betweenness.min())
    vmax = max(wt_betweenness.max(), var_betweenness.max())

    # Plot WT network
    node_colors_wt = [wt_betweenness[node] for node in wt_network.nodes()]
    node_sizes = [100 if node == variant_pos else 50 for node in wt_network.nodes()]

    nx.draw_networkx(
        wt_network,
        pos=pos,
        ax=ax1,
        node_color=node_colors_wt,
        node_size=node_sizes,
        cmap='viridis',
        vmin=vmin,
        vmax=vmax,
        with_labels=False,
        edge_color='gray',
        alpha=0.6
    )

    # Highlight variant position
    if variant_pos in wt_network.nodes():
        nx.draw_networkx_nodes(
            wt_network,
            pos,
            nodelist=[variant_pos],
            node_color='red',
            node_size=200,
            ax=ax1
        )

    ax1.set_title(f'Wild-Type Network\n({wt_network.number_of_edges()} edges)', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # Plot variant network
    node_colors_var = [var_betweenness[node] for node in var_network.nodes()]

    nx.draw_networkx(
        var_network,
        pos=pos,
        ax=ax2,
        node_color=node_colors_var,
        node_size=node_sizes,
        cmap='viridis',
        vmin=vmin,
        vmax=vmax,
        with_labels=False,
        edge_color='gray',
        alpha=0.6
    )

    # Highlight variant position
    if variant_pos in var_network.nodes():
        nx.draw_networkx_nodes(
            var_network,
            pos,
            nodelist=[variant_pos],
            node_color='red',
            node_size=200,
            ax=ax2
        )

    ax2.set_title(f'Variant Network\n({var_network.number_of_edges()} edges)', fontsize=12, fontweight='bold')
    ax2.axis('off')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=[ax1, ax2], orientation='horizontal', fraction=0.046, pad=0.04)
    cbar.set_label('Betweenness Centrality', fontsize=11)

    plt.suptitle(f'Network Topology: {variant_id}\n(Red node = variant position)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_network_examples(df: pd.DataFrame, wt_h5_path: Path, var_h5_path: Path, output_dir: Path, cfg: dict):
    """
    Create network visualization examples for high-impact variants.

    Args:
        df: DataFrame with variant features
        wt_h5_path: Path to WT predictions HDF5
        var_h5_path: Path to variant predictions HDF5
        output_dir: Directory to save figures
        cfg: Configuration dictionary
    """
    network_dir = output_dir / "network_examples"
    network_dir.mkdir(parents=True, exist_ok=True)

    # Select top variants by impact (largest delta in centrality)
    df_sorted = df.copy()
    df_sorted["total_impact"] = (
        df_sorted["delta_betweenness"].abs() +
        df_sorted["delta_closeness"].abs() +
        df_sorted["delta_degree"].abs()
    )
    df_sorted = df_sorted.nlargest(5, "total_impact")

    with h5py.File(wt_h5_path, 'r') as wt_h5, \
         h5py.File(var_h5_path, 'r') as var_h5:

        for idx, row in df_sorted.iterrows():
            try:
                variant_id = row["variant_id"]
                uniprot_id = row["uniprot_id"]
                protein_pos = int(row["protein_pos"]) - 1  # Convert to 0-indexed

                # Load WT data
                if uniprot_id not in wt_h5:
                    continue

                wt_grp = wt_h5[uniprot_id]
                wt_data = {
                    "gcc_lmi": wt_grp["gcc_lmi"][:],
                    "ca_dist": wt_grp["ca_dist"][:],
                    "centrality_betweenness": wt_grp["centrality_betweenness"][:],
                }

                # Load variant data
                variant_group_name = variant_id.replace("/", "_").replace(":", "_")
                if variant_group_name not in var_h5:
                    continue

                var_grp = var_h5[variant_group_name]
                var_data = {
                    "gcc_lmi": var_grp["gcc_lmi"][:],
                    "ca_dist": var_grp["ca_dist"][:],
                    "centrality_betweenness": var_grp["centrality_betweenness"][:],
                }

                # Create plot
                output_path = network_dir / f"{variant_id.replace('/', '_').replace(':', '_')}_network.png"
                plot_network_example(variant_id, wt_data, var_data, protein_pos, output_path, cfg)
                logger.info(f"Created network example for {variant_id}")

            except Exception as e:
                logger.warning(f"Error creating network example for {row['variant_id']}: {e}")


def main(config_path: str = "configs/clinvar_analysis_config.yml"):
    """Main function for visualization."""

    # Load configuration
    cfg = load_config(config_path)

    # Setup logging
    logger.add(cfg["log_file"], level=cfg["log_level"])
    logger.info("Starting visualization")

    # Setup output directory
    output_dir = Path(cfg["reports_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    features_path = Path(cfg["output_dir"]) / "variant_features.csv"
    results_path = Path(cfg["output_dir"]) / "statistical_results.json"

    if not features_path.exists():
        logger.error(f"Features file not found: {features_path}")
        return

    if not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        return

    df = pd.read_csv(features_path)
    with open(results_path, 'r') as f:
        results = json.load(f)

    logger.info(f"Loaded {len(df)} variants")

    # Create plots
    logger.info("Creating distribution plots...")
    plot_distributions(df, output_dir)

    logger.info("Creating ROC curves...")
    plot_roc_curves(results.get("roc_analysis", {}), output_dir)

    logger.info("Creating feature importance plot...")
    plot_feature_importance(results.get("logistic_regression", {}), output_dir)

    logger.info("Creating summary table...")
    plot_summary_table(results.get("mann_whitney_tests", {}), output_dir)

    # Create network examples
    logger.info("Creating network visualization examples...")
    wt_h5_path = Path(cfg["output_dir"]) / "wildtype_predictions.h5"
    var_h5_path = Path(cfg["output_dir"]) / "variant_predictions.h5"

    if wt_h5_path.exists() and var_h5_path.exists():
        create_network_examples(df, wt_h5_path, var_h5_path, output_dir, cfg)

    logger.info(f"\nAll visualizations saved to {output_dir}")


if __name__ == "__main__":
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/clinvar_analysis_config.yml"
    main(config_path)
