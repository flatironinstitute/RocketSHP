#!/usr/bin/env python
"""Test ClinVar variant mapping."""

import sys
import importlib.util
from pathlib import Path
import yaml

# Load the script as a module
script_path = Path("scripts/04_downstream/clinvar/01_prepare_clinvar_dataset.py")
spec = importlib.util.spec_from_file_location("clinvar_prep", script_path)
clinvar_prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clinvar_prep)

# Load config
with open('configs/clinvar_analysis_config.yml') as f:
    cfg = yaml.safe_load(f)

# Parse VCF (just first 1000 variants for testing)
vcf_path = Path(cfg["clinvar_vcf"])
variants_df = clinvar_prep.parse_clinvar_vcf(vcf_path, cfg)

print(f"Parsed {len(variants_df)} variants")
print(f"Columns: {list(variants_df.columns)}")
print(f"\nFirst few variants:")
print(variants_df.head())

# Test mapping
genome = clinvar_prep.get_genome(cfg["genome_version"], cfg["genome_release"])
print(f"\nGenome loaded: release {genome.release}")

# Test mapping first 1000 variants
stats = {}
success_count = 0
for i in range(min(1000, len(variants_df))):
    variant = variants_df.iloc[i].to_dict()
    mapped = clinvar_prep.map_variant_to_protein(variant, genome, cfg, stats)
    if mapped:
        success_count += 1
        if success_count <= 3:  # Print first 3 successes
            print(f"\nVariant {i}: {variant['chrom']}:{variant['pos']} {variant['ref']}>{variant['alt']}")
            print(f"  -> {len(mapped)} mappings")
            print(f"  Gene: {mapped[0]['gene_name']}, Position: {mapped[0]['protein_pos']}")
            print(f"  {mapped[0]['wt_aa']} -> {mapped[0]['variant_aa']}")

print(f"\nStats after 1000 variants:")
print(f"  Successful mappings: {stats.get('mapped_successfully', 0)}")
for key, val in sorted(stats.items(), key=lambda x: x[1], reverse=True):
    print(f"  {key}: {val}")
