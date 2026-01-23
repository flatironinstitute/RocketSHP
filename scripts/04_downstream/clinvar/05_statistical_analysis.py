#!/usr/bin/env python3
"""
Perform statistical analysis on ClinVar variant features.

This script:
1. Loads variant features CSV
2. Performs Mann-Whitney U tests for pathogenic vs benign vs VUS
3. Builds logistic regression models with covariates
4. Generates ROC/AUC analysis
5. Calculates effect sizes
6. Saves all results to JSON
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


def load_config(config_path: str):
    """Load configuration file."""
    cfg = OmegaConf.load(config_path)
    return OmegaConf.to_container(cfg, resolve=True)


def mann_whitney_test(group1: np.ndarray, group2: np.ndarray) -> dict:
    """
    Perform Mann-Whitney U test and calculate effect size.

    Args:
        group1: First group values
        group2: Second group values

    Returns:
        Dictionary with test results
    """
    # Remove NaNs
    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]

    if len(group1) < 3 or len(group2) < 3:
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "effect_size": np.nan,
            "n1": len(group1),
            "n2": len(group2),
        }

    # Mann-Whitney U test
    statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')

    # Calculate rank-biserial correlation (effect size for Mann-Whitney)
    # r_rb = 1 - (2U) / (n1 * n2)
    n1, n2 = len(group1), len(group2)
    effect_size = 1 - (2 * statistic) / (n1 * n2)

    # Also calculate medians
    median1 = np.median(group1)
    median2 = np.median(group2)

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "effect_size": float(effect_size),
        "n1": int(n1),
        "n2": int(n2),
        "median1": float(median1),
        "median2": float(median2),
    }


def perform_mann_whitney_tests(df: pd.DataFrame, features: list) -> dict:
    """
    Perform Mann-Whitney U tests for all feature comparisons.

    Args:
        df: DataFrame with variant features
        features: List of feature names to test

    Returns:
        Dictionary with test results
    """
    results = {}

    comparisons = [
        ("Pathogenic", "Benign"),
        ("Pathogenic", "VUS"),
        ("Benign", "VUS"),
    ]

    for feature in features:
        results[feature] = {}

        for group1_name, group2_name in comparisons:
            group1 = df[df["pathogenicity"] == group1_name][feature].values
            group2 = df[df["pathogenicity"] == group2_name][feature].values

            test_result = mann_whitney_test(group1, group2)
            results[feature][f"{group1_name}_vs_{group2_name}"] = test_result

    return results


def correct_multiple_testing(results: dict, alpha: float = 0.05) -> dict:
    """
    Apply Benjamini-Hochberg FDR correction to p-values.

    Args:
        results: Dictionary with test results
        alpha: FDR threshold

    Returns:
        Dictionary with corrected p-values
    """
    # Collect all p-values
    p_values = []
    keys = []

    for feature, comparisons in results.items():
        for comparison, test_result in comparisons.items():
            if not np.isnan(test_result["p_value"]):
                p_values.append(test_result["p_value"])
                keys.append((feature, comparison))

    # Apply Benjamini-Hochberg correction
    if len(p_values) > 0:
        reject, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')

        # Add corrected p-values to results
        for i, (feature, comparison) in enumerate(keys):
            results[feature][comparison]["p_corrected"] = float(p_corrected[i])
            results[feature][comparison]["significant"] = bool(reject[i])

    return results


def build_logistic_regression(df: pd.DataFrame, cfg: dict) -> dict:
    """
    Build logistic regression model for pathogenicity classification.

    Args:
        df: DataFrame with variant features
        cfg: Configuration dictionary

    Returns:
        Dictionary with model results
    """
    # Select features for model
    feature_cols = [
        "delta_rmsf",
        "delta_betweenness",
        "delta_closeness",
        "delta_degree",
        "delta_mean_rmsf",
        "delta_num_edges",
        "community_similarity",
        "protein_length",
        "relative_position",
        "local_plddt",
        "delta_hydrophobicity",
        "delta_charge",
        "delta_volume",
    ]

    # Filter for Pathogenic vs Benign
    df_binary = df[df["pathogenicity"].isin(["Pathogenic", "Benign"])].copy()

    # Remove rows with missing values
    df_binary = df_binary.dropna(subset=feature_cols)

    if len(df_binary) < 10:
        logger.warning("Insufficient data for logistic regression")
        return {}

    # Prepare data
    X = df_binary[feature_cols].values
    y = (df_binary["pathogenicity"] == "Pathogenic").astype(int).values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train logistic regression
    model = LogisticRegression(
        random_state=cfg["random_seed"],
        max_iter=1000,
        penalty='l2',
        C=1.0
    )
    model.fit(X_scaled, y)

    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')

    # Coefficients and feature importance
    coefficients = {
        feature_cols[i]: float(model.coef_[0][i])
        for i in range(len(feature_cols))
    }

    # Sort by absolute value
    sorted_coefs = sorted(
        coefficients.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    return {
        "n_samples": int(len(df_binary)),
        "n_pathogenic": int(y.sum()),
        "n_benign": int(len(y) - y.sum()),
        "coefficients": coefficients,
        "sorted_coefficients": sorted_coefs,
        "intercept": float(model.intercept_[0]),
        "cv_auc_mean": float(cv_scores.mean()),
        "cv_auc_std": float(cv_scores.std()),
        "feature_names": feature_cols,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }


def perform_roc_analysis(df: pd.DataFrame, cfg: dict) -> dict:
    """
    Perform ROC/AUC analysis for individual features and combined model.

    Args:
        df: DataFrame with variant features
        cfg: Configuration dictionary

    Returns:
        Dictionary with ROC results
    """
    # Filter for Pathogenic vs Benign
    df_binary = df[df["pathogenicity"].isin(["Pathogenic", "Benign"])].copy()

    if len(df_binary) < 10:
        logger.warning("Insufficient data for ROC analysis")
        return {}

    y_true = (df_binary["pathogenicity"] == "Pathogenic").astype(int).values

    results = {}

    # Individual features
    feature_cols = [
        "delta_rmsf",
        "delta_betweenness",
        "delta_closeness",
        "delta_degree",
        "delta_mean_rmsf",
    ]

    for feature in feature_cols:
        # Remove NaNs
        valid_mask = ~df_binary[feature].isna()
        if valid_mask.sum() < 10:
            continue

        y_score = df_binary.loc[valid_mask, feature].values
        y = y_true[valid_mask]

        try:
            # Calculate AUC
            auc_score = roc_auc_score(y, y_score)

            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(y, y_score)

            results[feature] = {
                "auc": float(auc_score),
                "n_samples": int(len(y)),
                # Store a subset of points for plotting
                "fpr": fpr[::max(1, len(fpr)//100)].tolist(),
                "tpr": tpr[::max(1, len(tpr)//100)].tolist(),
            }
        except Exception as e:
            logger.warning(f"Error calculating ROC for {feature}: {e}")

    # Combined model
    model_features = [
        "delta_rmsf", "delta_betweenness", "delta_closeness",
        "protein_length", "relative_position", "local_plddt"
    ]

    df_model = df_binary.dropna(subset=model_features)
    if len(df_model) >= 10:
        X = df_model[model_features].values
        y = (df_model["pathogenicity"] == "Pathogenic").astype(int).values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(random_state=cfg["random_seed"], max_iter=1000)
        model.fit(X_scaled, y)

        y_score = model.predict_proba(X_scaled)[:, 1]
        auc_score = roc_auc_score(y, y_score)
        fpr, tpr, _ = roc_curve(y, y_score)

        results["combined_model"] = {
            "auc": float(auc_score),
            "n_samples": int(len(y)),
            "fpr": fpr[::max(1, len(fpr)//100)].tolist(),
            "tpr": tpr[::max(1, len(tpr)//100)].tolist(),
        }

    return results


def main(config_path: str = "configs/clinvar_analysis_config.yml"):
    """Main function for statistical analysis."""

    # Load configuration
    cfg = load_config(config_path)

    # Setup logging
    logger.add(cfg["log_file"], level=cfg["log_level"])
    logger.info("Starting statistical analysis")

    # Load variant features
    features_path = Path(cfg["output_dir"]) / "variant_features.csv"
    if not features_path.exists():
        logger.error(f"Variant features not found: {features_path}")
        logger.error("Please run 04_extract_features.py first")
        return

    df = pd.read_csv(features_path)
    logger.info(f"Loaded {len(df)} variants with features")
    logger.info(f"Pathogenicity distribution:\n{df['pathogenicity'].value_counts()}")

    # Define features to test
    test_features = [
        "delta_rmsf",
        "delta_betweenness",
        "delta_closeness",
        "delta_degree",
        "delta_mean_rmsf",
        "delta_std_rmsf",
        "delta_num_edges",
        "delta_num_communities",
        "community_similarity",
    ]

    # Perform Mann-Whitney U tests
    logger.info("\nPerforming Mann-Whitney U tests...")
    mw_results = perform_mann_whitney_tests(df, test_features)

    # Apply multiple testing correction
    logger.info("Applying FDR correction...")
    mw_results = correct_multiple_testing(mw_results, alpha=cfg["fdr_threshold"])

    # Build logistic regression model
    logger.info("\nBuilding logistic regression model...")
    lr_results = build_logistic_regression(df, cfg)

    # Perform ROC analysis
    logger.info("\nPerforming ROC/AUC analysis...")
    roc_results = perform_roc_analysis(df, cfg)

    # Compile all results
    all_results = {
        "mann_whitney_tests": mw_results,
        "logistic_regression": lr_results,
        "roc_analysis": roc_results,
        "summary": {
            "total_variants": int(len(df)),
            "pathogenicity_counts": df["pathogenicity"].value_counts().to_dict(),
            "test_features": test_features,
        }
    }

    # Save results
    output_path = Path(cfg["output_dir"]) / "statistical_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nSaved statistical results to {output_path}")

    # Print summary of significant results
    logger.info("\n=== Significant Results (FDR < 0.05) ===")
    for feature, comparisons in mw_results.items():
        for comparison, results in comparisons.items():
            if results.get("significant", False):
                logger.info(
                    f"{feature} - {comparison}: "
                    f"p={results['p_value']:.2e}, "
                    f"p_corrected={results['p_corrected']:.2e}, "
                    f"effect_size={results['effect_size']:.3f}"
                )

    if lr_results:
        logger.info("\n=== Logistic Regression Top Features ===")
        for feature, coef in lr_results["sorted_coefficients"][:5]:
            logger.info(f"{feature}: {coef:.3f}")
        logger.info(f"\nCross-validated AUC: {lr_results['cv_auc_mean']:.3f} ± {lr_results['cv_auc_std']:.3f}")

    if roc_results:
        logger.info("\n=== ROC AUC Scores ===")
        for feature, results in roc_results.items():
            logger.info(f"{feature}: AUC = {results['auc']:.3f} (n={results['n_samples']})")


if __name__ == "__main__":
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/clinvar_analysis_config.yml"
    main(config_path)
