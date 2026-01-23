#!/usr/bin/env python3
"""
Prepare ClinVar dataset for pathogenicity analysis.

This script:
1. Parses ClinVar VCF and categorizes variants (Pathogenic/Benign/VUS)
2. Maps variants to protein sequences using pyensembl
3. Retrieves and validates AlphaFold structures
4. Applies quality filters
5. Creates analysis-ready dataset
"""

import gzip
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyensembl
import vcfpy
from biotite.database import afdb, uniprot
from biotite.sequence import NucleotideSequence
from biotite.structure.io import pdb
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm

from rocketshp import config as rocketshp_config


def load_config(config_path: str) -> Dict:
    """Load configuration file."""
    cfg = OmegaConf.load(config_path)
    return OmegaConf.to_container(cfg, resolve=True)


def categorize_variant(clnsig_values: List[str], cfg: Dict) -> Optional[str]:
    """
    Categorize variant based on clinical significance.

    Args:
        clnsig_values: List of clinical significance values
        cfg: Configuration dictionary

    Returns:
        Category: "Pathogenic", "Benign", "VUS", or None if conflicting/other
    """
    has_pathogenic = False
    has_benign = False
    has_vus = False

    for sig in clnsig_values:
        # Check for pathogenic
        if any(keyword in sig for keyword in cfg["pathogenic_keywords"]):
            has_pathogenic = True
        # Check for benign
        if any(keyword in sig for keyword in cfg["benign_keywords"]):
            has_benign = True
        # Check for VUS
        if any(keyword in sig for keyword in cfg["vus_keywords"]):
            has_vus = True

    # Handle conflicting interpretations
    if cfg["exclude_conflicting"]:
        if (has_pathogenic and has_benign) or \
           (has_pathogenic and has_vus) or \
           (has_benign and has_vus):
            return None

    # Return single category
    if has_pathogenic:
        return "Pathogenic"
    elif has_benign:
        return "Benign"
    elif has_vus:
        return "VUS"
    else:
        return None


def parse_clinvar_vcf(vcf_path: Path, cfg: Dict) -> pd.DataFrame:
    """
    Parse ClinVar VCF file and extract categorized variants.

    Args:
        vcf_path: Path to ClinVar VCF file
        cfg: Configuration dictionary

    Returns:
        DataFrame with variant information
    """
    variants = []

    logger.info(f"Parsing ClinVar VCF: {vcf_path}")

    with gzip.open(vcf_path, 'rt') as f:
        reader = vcfpy.Reader(f)

        for record in tqdm(reader, desc="Parsing VCF"):
            # Check for clinical significance
            if "CLNSIG" not in record.INFO:
                continue

            clnsig_values = record.INFO["CLNSIG"]
            category = categorize_variant(clnsig_values, cfg)

            if category is None:
                continue

            # Extract variant information for each alt allele
            for alt_allele in record.ALT:
                variant_info = {
                    "chrom": record.CHROM,
                    "pos": record.POS,
                    "variant_id": record.ID[0] if record.ID else f"{record.CHROM}_{record.POS}_{record.REF}_{alt_allele.value}",
                    "ref": record.REF,
                    "alt": alt_allele.value,  # Extract the value from the Substitution object
                    "pathogenicity": category,
                    "clnsig": "|".join(clnsig_values),
                    "qual": record.QUAL,
                }

                # Add optional INFO fields
                for field in ["CLNDN", "CLNHGVS", "GENEINFO"]:
                    if field in record.INFO:
                        variant_info[field.lower()] = "|".join(record.INFO[field]) if isinstance(record.INFO[field], list) else record.INFO[field]

                variants.append(variant_info)

    df = pd.DataFrame(variants)
    logger.info(f"Parsed {len(df)} variants across {len(df['pathogenicity'].unique())} categories")
    logger.info(f"Category counts:\n{df['pathogenicity'].value_counts()}")

    return df


def get_genome(genome_version: str, release: int):
    """Initialize and download genome if needed."""
    genome = pyensembl.EnsemblRelease(release, species="human")

    try:
        # Try to load existing index
        genome.index()
    except Exception as e:
        # Download and index if not available
        logger.info(f"Downloading Ensembl release {release}... (Error: {e})")
        genome.download()
        genome.index()

    return genome


def map_variant_to_protein(variant: Dict, genome, cfg: Dict, stats: Dict = None) -> List[Dict]:
    """
    Map genomic variant to protein sequence(s).

    Args:
        variant: Variant information dictionary
        genome: pyensembl genome object
        cfg: Configuration dictionary
        stats: Statistics dictionary for tracking filter reasons

    Returns:
        List of mapped variant-transcript pairs
    """
    results = []

    if stats is None:
        stats = {}

    try:
        # Normalize chromosome format
        chrom = variant["chrom"].replace("chr", "")

        # Get transcripts at position
        transcripts = genome.transcripts_at_locus(contig=chrom, position=variant["pos"])

        if not transcripts:
            stats["no_transcripts"] = stats.get("no_transcripts", 0) + 1
            return results

        for transcript in transcripts:
            # Only process protein-coding transcripts
            if not transcript.is_protein_coding:
                stats["not_protein_coding"] = stats.get("not_protein_coding", 0) + 1
                continue

            try:
                # Get original sequences
                protein_seq = transcript.protein_sequence

                if not protein_seq:
                    stats["no_protein_seq"] = stats.get("no_protein_seq", 0) + 1
                    continue

                # Get coding sequence (CDS only, no UTRs)
                try:
                    cds_seq = transcript.coding_sequence
                except Exception as e:
                    # Some transcripts may not have CDS
                    logger.debug(f"No CDS for transcript {transcript.id}: {e}")
                    stats["no_cds"] = stats.get("no_cds", 0) + 1
                    continue

                # Find position in transcript coordinates
                genomic_pos = variant["pos"]
                transcript_pos = None

                for i, exon in enumerate(transcript.exons):
                    if exon.start <= genomic_pos <= exon.end:
                        if transcript.strand == "+":
                            transcript_pos = sum(len(e) for e in transcript.exons[:i]) + (genomic_pos - exon.start)
                        else:
                            transcript_pos = sum(len(e) for e in transcript.exons[i+1:]) + (exon.end - genomic_pos)
                        break

                if transcript_pos is None:
                    # Variant not in exon
                    stats["not_in_exon"] = stats.get("not_in_exon", 0) + 1
                    continue

                # Convert genomic position to CDS position
                # coding_sequence_position_ranges returns genomic coordinates
                cds_ranges = transcript.coding_sequence_position_ranges
                cds_pos = None

                # Find which CDS range contains this variant
                cumulative_length = 0
                for cds_start, cds_end in cds_ranges:
                    if cds_start <= genomic_pos <= cds_end:
                        # Variant is in this CDS range
                        if transcript.strand == "+":
                            cds_pos = cumulative_length + (genomic_pos - cds_start)
                        else:
                            cds_pos = cumulative_length + (cds_end - genomic_pos)
                        break
                    # Add length of this CDS range to cumulative
                    cumulative_length += (cds_end - cds_start + 1)

                # Check if position is within CDS
                if cds_pos is None or cds_pos < 0 or cds_pos >= len(cds_seq):
                    # Variant not in CDS
                    stats["not_in_cds"] = stats.get("not_in_cds", 0) + 1
                    continue

                # Calculate protein position (0-indexed)
                protein_pos = cds_pos // 3

                # Check if position is too close to termini
                if protein_pos < cfg["terminus_buffer"] or \
                   protein_pos >= len(protein_seq) - cfg["terminus_buffer"]:
                    stats["near_terminus"] = stats.get("near_terminus", 0) + 1
                    continue

                # Apply mutation to protein sequence for simple substitutions
                # NOTE: Only handling SNVs - indels are skipped
                if len(variant["ref"]) == 1 and len(variant["alt"]) == 1:
                    # Simple SNV
                    wt_aa = protein_seq[protein_pos] if protein_pos < len(protein_seq) else None

                    # Apply mutation to CDS
                    mut_cds = list(cds_seq)
                    mut_cds[cds_pos] = variant["alt"]
                    mut_cds_str = ''.join(mut_cds)

                    # Translate mutant CDS
                    try:
                        # Use complete=True since CDS length should be multiple of 3
                        dna_biotite = NucleotideSequence(mut_cds_str)
                        protein_biotite = dna_biotite.translate(complete=True)
                        mut_protein_seq = str(protein_biotite)
                        variant_aa = mut_protein_seq[protein_pos] if protein_pos < len(mut_protein_seq) else None
                    except Exception as e:
                        # Translation failed, skip
                        logger.debug(f"Translation failed for variant {variant['variant_id']}: {e}")
                        stats["translation_failed"] = stats.get("translation_failed", 0) + 1
                        continue

                    if wt_aa and variant_aa:
                        stats["mapped_successfully"] = stats.get("mapped_successfully", 0) + 1
                        results.append({
                            **variant,
                            "transcript_id": transcript.id,
                            "transcript_name": transcript.name,
                            "gene_id": transcript.gene.id,
                            "gene_name": transcript.gene.name,
                            "protein_pos": protein_pos + 1,  # Convert to 1-indexed
                            "wt_aa": wt_aa,
                            "variant_aa": variant_aa,
                            "wt_sequence": protein_seq,
                            "variant_sequence": mut_protein_seq,
                            "protein_length": len(protein_seq),
                        })
                    else:
                        stats["missing_aa"] = stats.get("missing_aa", 0) + 1
                else:
                    stats["not_snv"] = stats.get("not_snv", 0) + 1

            except Exception as e:
                logger.debug(f"Error processing transcript {transcript.id}: {e}")
                continue

    except Exception as e:
        logger.debug(f"Error mapping variant {variant['variant_id']}: {e}")

    return results


def get_uniprot_id(gene_name: str, gene_id: Optional[str] = None) -> Optional[str]:
    """
    Get UniProt ID for a gene name using biotite's UniProt interface.

    Args:
        gene_name: Gene name
        gene_id: Ensembl gene ID (optional, e.g., ENSG00000157764)

    Returns:
        UniProt accession or None
    """
    # Try multiple query strategies in order of preference
    queries = []

    # 1. Try gene name with reviewed entries (most reliable)
    if gene_name:
        queries.append(f"gene:{gene_name} AND organism_id:9606 AND reviewed:true")

    # 2. Try Ensembl gene ID with reviewed entries
    if gene_id:
        queries.append(f"xref:ensembl-{gene_id} AND organism_id:9606 AND reviewed:true")

    # 3. Try gene name without reviewed restriction (includes TrEMBL)
    if gene_name:
        queries.append(f"gene:{gene_name} AND organism_id:9606")

    # 4. Try Ensembl gene ID without reviewed restriction
    if gene_id:
        queries.append(f"xref:ensembl-{gene_id} AND organism_id:9606")

    for query_str in queries:
        try:
            # Use biotite's search function
            results = uniprot.search(query_str, number=1)
            if results:
                uniprot_id = results[0]
                logger.debug(f"Found UniProt ID {uniprot_id} for {gene_name} (gene_id: {gene_id})")
                return uniprot_id
        except Exception as e:
            logger.debug(f"Query failed: {query_str[:100]}... Error: {e}")
            continue

    logger.debug(f"No UniProt ID found for {gene_name} (gene_id: {gene_id})")
    return None


def get_alphafold_plddt(uniprot_id: str) -> Optional[float]:
    """
    Query AlphaFold metadata API to get average pLDDT score without downloading structure.

    Args:
        uniprot_id: UniProt accession

    Returns:
        Average pLDDT score or None if not available
    """
    try:
        import json
        import ssl
        import urllib.request

        # Create SSL context that doesn't verify certificates (needed for HPC)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"

        with urllib.request.urlopen(url, timeout=10, context=ssl_context) as response:
            data = json.loads(response.read().decode())
            # globalMetricValue is the average pLDDT
            avg_plddt = data[0]["globalMetricValue"] if isinstance(data, list) else data["globalMetricValue"]
            logger.debug(f"AlphaFold metadata for {uniprot_id}: pLDDT = {avg_plddt:.2f}")
            return avg_plddt
    except Exception as e:
        logger.debug(f"Error querying AlphaFold metadata for {uniprot_id}: {e}")
        return None


def download_alphafold_structure(uniprot_id: str, output_dir: Path, cfg: Dict) -> Optional[Path]:
    """
    Download AlphaFold structure for UniProt ID using biotite's afdb interface.

    Args:
        uniprot_id: UniProt accession
        output_dir: Directory to save PDB file
        cfg: Configuration dictionary

    Returns:
        Path to downloaded PDB file or None
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = output_dir / f"AF-{uniprot_id}-F1-model_{cfg['alphafold_version']}.pdb"

    # Check if already downloaded
    if pdb_path.exists():
        return pdb_path

    try:
        # Use biotite's fetch function to download the PDB file
        pdb_file = afdb.fetch(uniprot_id, format="pdb")
        pdb_content = pdb_file.read()

        # Write to file
        with open(pdb_path, 'w') as f:
            f.write(pdb_content)

        logger.debug(f"Downloaded structure for {uniprot_id}")
        return pdb_path

    except Exception as e:
        logger.debug(f"Error downloading AlphaFold structure for {uniprot_id}: {e}")
        return None


def get_plddt_scores(pdb_path: Path) -> Tuple[float, np.ndarray]:
    """
    Extract pLDDT scores from AlphaFold PDB file.

    pLDDT scores are stored in the B-factor column.

    Args:
        pdb_path: Path to PDB file

    Returns:
        (average_plddt, per_residue_plddt_array)
    """
    try:
        pdb_file = pdb.PDBFile.read(pdb_path)
        # Request b_factor in extra_fields
        structure = pdb_file.get_structure(extra_fields=["b_factor"])

        # Handle AtomArrayStack (multiple models) - take first model
        if len(structure.shape) > 1:
            structure = structure[0]

        # Get B-factors (pLDDT scores)
        b_factors = structure.b_factor

        # Get per-residue average (handling multiple atoms per residue)
        atom_res_ids = structure.res_id
        unique_res_ids = np.unique(atom_res_ids)

        per_residue_plddt = np.array([
            b_factors[atom_res_ids == res_id].mean()
            for res_id in unique_res_ids
        ])

        avg_plddt = per_residue_plddt.mean()

        return avg_plddt, per_residue_plddt

    except Exception as e:
        logger.debug(f"Error extracting pLDDT from {pdb_path}: {e}")
        return 0.0, np.array([])


def main(config_path: str = "configs/clinvar_analysis_config.yml"):
    """Main function to prepare ClinVar dataset."""

    # Load configuration
    cfg = load_config(config_path)

    # Setup logging
    logger.add(cfg["log_file"], level=cfg["log_level"])
    logger.info("Starting ClinVar dataset preparation")

    # Setup output directory
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    alphafold_dir = Path(cfg["alphafold_dir"])
    alphafold_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Parse ClinVar VCF
    vcf_path = Path(cfg["clinvar_vcf"])
    if not vcf_path.exists():
        logger.error(f"ClinVar VCF not found: {vcf_path}")
        return

    variants_df = parse_clinvar_vcf(vcf_path, cfg)

    # Step 2: Map variants to proteins
    logger.info("Mapping variants to protein sequences...")
    genome = get_genome(cfg["genome_version"], cfg["genome_release"])

    mapped_variants = []
    filter_stats = {}
    for _, variant in tqdm(variants_df.iterrows(), total=len(variants_df), desc="Mapping variants"):
        mapped = map_variant_to_protein(variant.to_dict(), genome, cfg, filter_stats)
        mapped_variants.extend(mapped)

    # Log filter statistics
    logger.info("Variant mapping filter statistics:")
    for reason, count in sorted(filter_stats.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {reason}: {count}")

    if not mapped_variants:
        logger.error("No variants successfully mapped to proteins")
        return

    mapped_df = pd.DataFrame(mapped_variants)
    logger.info(f"Mapped {len(mapped_df)} variant-transcript pairs")

    # Apply protein length filter
    mapped_df = mapped_df[
        (mapped_df["protein_length"] >= cfg["min_protein_length"]) &
        (mapped_df["protein_length"] <= cfg["max_protein_length"])
    ]
    logger.info(f"After length filter: {len(mapped_df)} variants")

    # Step 3: Get UniProt IDs and AlphaFold structures
    logger.info("Fetching UniProt IDs and AlphaFold structures...")

    # Get unique gene_name/gene_id pairs
    unique_genes = mapped_df[["gene_name", "gene_id"]].drop_duplicates()
    gene_to_uniprot = {}

    for _, row in tqdm(unique_genes.iterrows(), total=len(unique_genes), desc="Fetching UniProt IDs"):
        gene_name = row["gene_name"]
        gene_id = row["gene_id"]
        uniprot_id = get_uniprot_id(gene_name, gene_id)
        if uniprot_id:
            gene_to_uniprot[gene_name] = uniprot_id

    # Add UniProt IDs to dataframe
    mapped_df["uniprot_id"] = mapped_df["gene_name"].map(gene_to_uniprot)

    # Filter out genes without UniProt IDs
    mapped_df = mapped_df[mapped_df["uniprot_id"].notna()]
    logger.info(f"After UniProt filter: {len(mapped_df)} variants with UniProt IDs")

    # Step 1: Query AlphaFold metadata to get pLDDT scores (fast, no download)
    logger.info("Querying AlphaFold metadata for pLDDT scores...")
    uniprot_to_plddt = {}
    unique_uniprot_ids = mapped_df["uniprot_id"].unique()

    for uniprot_id in tqdm(unique_uniprot_ids, desc="Querying pLDDT"):
        avg_plddt = get_alphafold_plddt(uniprot_id)
        if avg_plddt is not None:
            uniprot_to_plddt[uniprot_id] = avg_plddt

    # Add pLDDT scores to dataframe
    mapped_df["plddt_avg"] = mapped_df["uniprot_id"].map(uniprot_to_plddt)

    # Filter by pLDDT threshold BEFORE downloading structures
    mapped_df = mapped_df[
        (mapped_df["plddt_avg"].notna()) &
        (mapped_df["plddt_avg"] >= cfg["min_plddt"])
    ]
    logger.info(f"After pLDDT filter (>={cfg['min_plddt']}): {len(mapped_df)} variants")
    logger.info(f"Unique proteins passing filter: {mapped_df['uniprot_id'].nunique()}")

    # Step 2: Download only structures that passed the pLDDT filter
    logger.info("Downloading AlphaFold structures for high-quality proteins...")
    uniprot_to_pdb = {}
    passing_uniprot_ids = mapped_df["uniprot_id"].unique()

    for uniprot_id in tqdm(passing_uniprot_ids, desc="Downloading structures"):
        pdb_path = download_alphafold_structure(uniprot_id, alphafold_dir, cfg)
        if pdb_path and pdb_path.exists():
            uniprot_to_pdb[uniprot_id] = str(pdb_path)

    # Add PDB paths to dataframe
    mapped_df["pdb_path"] = mapped_df["uniprot_id"].map(uniprot_to_pdb)

    # Final filter: only keep variants with downloaded structures
    mapped_df = mapped_df[mapped_df["pdb_path"].notna()]
    logger.info(f"After structure download: {len(mapped_df)} variants with structures")

    # Step 4: Save final dataset
    output_path = output_dir / "variant_dataset.csv"
    mapped_df.to_csv(output_path, index=False)
    logger.info(f"Saved variant dataset to {output_path}")

    # Print summary statistics
    logger.info("\n=== Dataset Summary ===")
    logger.info(f"Total variants: {len(mapped_df)}")
    logger.info(f"Unique proteins: {mapped_df['uniprot_id'].nunique()}")
    logger.info(f"Unique genes: {mapped_df['gene_name'].nunique()}")
    logger.info(f"\nPathogenicity distribution:\n{mapped_df['pathogenicity'].value_counts()}")
    logger.info(f"\nProtein length stats:\n{mapped_df['protein_length'].describe()}")
    logger.info(f"\npLDDT stats:\n{mapped_df['plddt_avg'].describe()}")

    return mapped_df


if __name__ == "__main__":
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/clinvar_analysis_config.yml"
    main(config_path)
