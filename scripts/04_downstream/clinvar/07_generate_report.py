#!/usr/bin/env python3
"""
Generate comprehensive summary report for ClinVar pathogenicity analysis.

This script:
1. Compiles all analysis results
2. Generates summary statistics
3. Creates supplementary tables
4. Produces final HTML report
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf


def load_config(config_path: str):
    """Load configuration file."""
    cfg = OmegaConf.load(config_path)
    return OmegaConf.to_container(cfg, resolve=True)


def generate_html_report(
    df: pd.DataFrame,
    results: dict,
    cfg: dict,
    output_path: Path
):
    """
    Generate HTML report with all analysis results.

    Args:
        df: DataFrame with variant features
        results: Dictionary with statistical results
        cfg: Configuration dictionary
        output_path: Output HTML file path
    """
    html_content = []

    # Header
    html_content.append("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ClinVar Pathogenicity Analysis Report</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 40px;
                line-height: 1.6;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }
            h2 {
                color: #34495e;
                border-bottom: 2px solid #95a5a6;
                padding-bottom: 5px;
                margin-top: 30px;
            }
            h3 {
                color: #7f8c8d;
                margin-top: 20px;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }
            th, td {
                border: 1px solid #bdc3c7;
                padding: 12px;
                text-align: left;
            }
            th {
                background-color: #3498db;
                color: white;
                font-weight: bold;
            }
            tr:nth-child(even) {
                background-color: #ecf0f1;
            }
            .summary-box {
                background-color: #e8f4f8;
                border-left: 5px solid #3498db;
                padding: 15px;
                margin: 20px 0;
            }
            .warning-box {
                background-color: #fff3cd;
                border-left: 5px solid #ffc107;
                padding: 15px;
                margin: 20px 0;
            }
            .success-box {
                background-color: #d4edda;
                border-left: 5px solid #28a745;
                padding: 15px;
                margin: 20px 0;
            }
            img {
                max-width: 100%;
                height: auto;
                margin: 20px 0;
            }
            .metric {
                display: inline-block;
                margin: 10px 20px 10px 0;
            }
            .metric-value {
                font-size: 2em;
                font-weight: bold;
                color: #3498db;
            }
            .metric-label {
                font-size: 0.9em;
                color: #7f8c8d;
            }
        </style>
    </head>
    <body>
        <h1>ClinVar Pathogenicity Analysis Report</h1>
        <p><strong>Analysis Date:</strong> {date}</p>
        <p><strong>Configuration:</strong> {config_file}</p>
    """.format(
        date=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        config_file=cfg.get("config_path", "clinvar_analysis_config.yml")
    ))

    # Executive Summary
    html_content.append("<h2>Executive Summary</h2>")
    html_content.append('<div class="summary-box">')

    pathogenicity_counts = df["pathogenicity"].value_counts().to_dict()
    html_content.append(f"""
        <div class="metric">
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-label">Total Variants Analyzed</div>
        </div>
        <div class="metric">
            <div class="metric-value">{df['uniprot_id'].nunique():,}</div>
            <div class="metric-label">Unique Proteins</div>
        </div>
        <div class="metric">
            <div class="metric-value">{df['gene_name'].nunique():,}</div>
            <div class="metric-label">Unique Genes</div>
        </div>
    """)

    html_content.append("</div>")

    # Dataset Overview
    html_content.append("<h2>Dataset Overview</h2>")

    html_content.append("<h3>Pathogenicity Distribution</h3>")
    html_content.append("<table>")
    html_content.append("<tr><th>Category</th><th>Count</th><th>Percentage</th></tr>")
    total = len(df)
    for category, count in pathogenicity_counts.items():
        pct = (count / total) * 100
        html_content.append(f"<tr><td>{category}</td><td>{count:,}</td><td>{pct:.1f}%</td></tr>")
    html_content.append("</table>")

    # Protein Statistics
    html_content.append("<h3>Protein Statistics</h3>")
    html_content.append("<table>")
    html_content.append("<tr><th>Metric</th><th>Value</th></tr>")
    html_content.append(f"<tr><td>Mean Protein Length</td><td>{df['protein_length'].mean():.0f} aa</td></tr>")
    html_content.append(f"<tr><td>Median Protein Length</td><td>{df['protein_length'].median():.0f} aa</td></tr>")
    html_content.append(f"<tr><td>Protein Length Range</td><td>{df['protein_length'].min():.0f} - {df['protein_length'].max():.0f} aa</td></tr>")
    html_content.append(f"<tr><td>Mean pLDDT</td><td>{df['plddt_avg'].mean():.1f}</td></tr>")
    html_content.append(f"<tr><td>Median pLDDT</td><td>{df['plddt_avg'].median():.1f}</td></tr>")
    html_content.append("</table>")

    # Statistical Results
    html_content.append("<h2>Statistical Analysis Results</h2>")

    mw_results = results.get("mann_whitney_tests", {})
    lr_results = results.get("logistic_regression", {})
    roc_results = results.get("roc_analysis", {})

    # Mann-Whitney U Tests
    html_content.append("<h3>Mann-Whitney U Test Results</h3>")

    significant_results = []
    for feature, comparisons in mw_results.items():
        for comparison, test_result in comparisons.items():
            if test_result.get("significant", False):
                significant_results.append({
                    "Feature": feature,
                    "Comparison": comparison,
                    "P-value": test_result["p_value"],
                    "P-corrected": test_result["p_corrected"],
                    "Effect Size": test_result["effect_size"],
                    "Median 1": test_result.get("median1", "N/A"),
                    "Median 2": test_result.get("median2", "N/A"),
                })

    if significant_results:
        html_content.append('<div class="success-box">')
        html_content.append(f"<strong>{len(significant_results)} significant associations found (FDR < 0.05)</strong>")
        html_content.append("</div>")

        html_content.append("<table>")
        html_content.append("<tr><th>Feature</th><th>Comparison</th><th>P-value</th><th>P-corrected</th><th>Effect Size</th></tr>")
        for result in significant_results[:20]:  # Top 20
            html_content.append(
                f"<tr>"
                f"<td>{result['Feature']}</td>"
                f"<td>{result['Comparison']}</td>"
                f"<td>{result['P-value']:.2e}</td>"
                f"<td>{result['P-corrected']:.2e}</td>"
                f"<td>{result['Effect Size']:.3f}</td>"
                f"</tr>"
            )
        html_content.append("</table>")
    else:
        html_content.append('<div class="warning-box">')
        html_content.append("No significant associations found after FDR correction.")
        html_content.append("</div>")

    # Logistic Regression
    if lr_results:
        html_content.append("<h3>Logistic Regression Model</h3>")
        html_content.append(f"<p><strong>Cross-validated AUC:</strong> {lr_results['cv_auc_mean']:.3f} ± {lr_results['cv_auc_std']:.3f}</p>")
        html_content.append(f"<p><strong>Training Samples:</strong> {lr_results['n_samples']:,} ({lr_results['n_pathogenic']} pathogenic, {lr_results['n_benign']} benign)</p>")

        html_content.append("<p><strong>Top 10 Features by Importance:</strong></p>")
        html_content.append("<table>")
        html_content.append("<tr><th>Feature</th><th>Coefficient</th></tr>")
        for feature, coef in lr_results["sorted_coefficients"][:10]:
            html_content.append(f"<tr><td>{feature}</td><td>{coef:.3f}</td></tr>")
        html_content.append("</table>")

    # ROC Analysis
    if roc_results:
        html_content.append("<h3>ROC/AUC Analysis</h3>")
        html_content.append("<table>")
        html_content.append("<tr><th>Feature/Model</th><th>AUC</th><th>Samples</th></tr>")
        for feature, data in sorted(roc_results.items(), key=lambda x: x[1]["auc"], reverse=True):
            html_content.append(f"<tr><td>{feature}</td><td>{data['auc']:.3f}</td><td>{data['n_samples']:,}</td></tr>")
        html_content.append("</table>")

    # Key Findings
    html_content.append("<h2>Key Findings</h2>")
    html_content.append("<ul>")

    # Determine best predictive feature
    if roc_results:
        best_feature = max(roc_results.items(), key=lambda x: x[1]["auc"])
        html_content.append(f"<li>The <strong>{best_feature[0]}</strong> shows the highest individual predictive power with AUC = {best_feature[1]['auc']:.3f}</li>")

    # Significant features
    if significant_results:
        sig_features = set(r["Feature"] for r in significant_results)
        html_content.append(f"<li><strong>{len(sig_features)}</strong> dynamic features show significant association with pathogenicity after FDR correction</li>")

    # Top effect sizes
    if significant_results:
        largest_effect = max(significant_results, key=lambda x: abs(x["Effect Size"]))
        html_content.append(f"<li>Largest effect size observed for <strong>{largest_effect['Feature']}</strong> ({largest_effect['Comparison']}): {largest_effect['Effect Size']:.3f}</li>")

    html_content.append("</ul>")

    # Visualizations
    html_content.append("<h2>Visualizations</h2>")

    figures_dir = Path(cfg["reports_dir"])
    figures = [
        ("distributions.png", "Distribution of Dynamic Properties by Pathogenicity"),
        ("roc_curves.png", "ROC Curves for Pathogenic vs Benign Classification"),
        ("feature_importance.png", "Logistic Regression Feature Importance"),
        ("summary_table.png", "Summary of Significant Statistical Tests"),
    ]

    for fig_file, caption in figures:
        fig_path = figures_dir / fig_file
        if fig_path.exists():
            html_content.append(f"<h3>{caption}</h3>")
            html_content.append(f'<img src="{fig_file}" alt="{caption}" />')

    # Network examples
    network_dir = figures_dir / "network_examples"
    if network_dir.exists():
        network_examples = list(network_dir.glob("*.png"))
        if network_examples:
            html_content.append("<h3>Network Topology Examples</h3>")
            for example in network_examples[:5]:  # Show first 5
                html_content.append(f'<img src="network_examples/{example.name}" alt="Network example" />')

    # Supplementary Tables
    html_content.append("<h2>Supplementary Tables</h2>")

    # Top variants by impact
    html_content.append("<h3>Top 20 Variants by Total Impact Score</h3>")
    df_impact = df.copy()
    df_impact["total_impact"] = (
        df_impact["delta_rmsf"].abs().fillna(0) +
        df_impact["delta_betweenness"].abs().fillna(0) +
        df_impact["delta_closeness"].abs().fillna(0)
    )
    top_variants = df_impact.nlargest(20, "total_impact")[[
        "variant_id", "gene_name", "wt_aa", "protein_pos", "variant_aa",
        "pathogenicity", "delta_rmsf", "delta_betweenness", "delta_closeness"
    ]]

    html_content.append("<table>")
    html_content.append("<tr><th>Variant ID</th><th>Gene</th><th>Mutation</th><th>Pathogenicity</th><th>ΔRMSF</th><th>ΔBetweenness</th><th>ΔCloseness</th></tr>")
    for _, row in top_variants.iterrows():
        mutation = f"{row['wt_aa']}{row['protein_pos']}{row['variant_aa']}"
        html_content.append(
            f"<tr>"
            f"<td>{row['variant_id']}</td>"
            f"<td>{row['gene_name']}</td>"
            f"<td>{mutation}</td>"
            f"<td>{row['pathogenicity']}</td>"
            f"<td>{row['delta_rmsf']:.3f}</td>"
            f"<td>{row['delta_betweenness']:.3f}</td>"
            f"<td>{row['delta_closeness']:.3f}</td>"
            f"</tr>"
        )
    html_content.append("</table>")

    # Gene-level summary
    html_content.append("<h3>Top 20 Genes by Variant Count</h3>")
    gene_counts = df.groupby("gene_name").agg({
        "variant_id": "count",
        "pathogenicity": lambda x: (x == "Pathogenic").sum()
    }).rename(columns={"variant_id": "total_variants", "pathogenicity": "pathogenic_variants"})
    gene_counts = gene_counts.nlargest(20, "total_variants")

    html_content.append("<table>")
    html_content.append("<tr><th>Gene</th><th>Total Variants</th><th>Pathogenic Variants</th><th>% Pathogenic</th></tr>")
    for gene, row in gene_counts.iterrows():
        pct = (row["pathogenic_variants"] / row["total_variants"]) * 100
        html_content.append(
            f"<tr>"
            f"<td>{gene}</td>"
            f"<td>{row['total_variants']}</td>"
            f"<td>{row['pathogenic_variants']}</td>"
            f"<td>{pct:.1f}%</td>"
            f"</tr>"
        )
    html_content.append("</table>")

    # Footer
    html_content.append("""
        <hr>
        <p style="color: #7f8c8d; font-size: 0.9em;">
            Report generated by RocketSHP ClinVar Pathogenicity Analysis Pipeline<br>
            For more information, see the project documentation.
        </p>
    </body>
    </html>
    """)

    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(html_content))


def main(config_path: str = "configs/clinvar_analysis_config.yml"):
    """Main function to generate report."""

    # Load configuration
    cfg = load_config(config_path)
    cfg["config_path"] = config_path

    # Setup logging
    logger.add(cfg["log_file"], level=cfg["log_level"])
    logger.info("Starting report generation")

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

    # Generate HTML report
    output_path = Path(cfg["reports_dir"]) / "clinvar_pathogenicity_report.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generate_html_report(df, results, cfg, output_path)

    logger.info(f"\n{'='*60}")
    logger.info(f"Report generated successfully!")
    logger.info(f"{'='*60}")
    logger.info(f"Output: {output_path}")
    logger.info(f"\nOpen the report in your browser:")
    logger.info(f"  file://{output_path.absolute()}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/clinvar_analysis_config.yml"
    main(config_path)
