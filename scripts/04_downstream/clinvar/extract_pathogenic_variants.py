#!/usr/bin/env python3
"""
Extract pathogenic variants from ClinVar VCF and get transcript sequences with mutations applied.

This script parses the ClinVar VCF file, identifies pathogenic variants, finds overlapping
transcripts using pyensembl, and returns DNA and protein sequences with mutations applied.
"""

import gzip
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
from tqdm import tqdm

try:
    import vcfpy
except ImportError:
    print("vcfpy not found. Install with: pip install vcfpy")
    exit(1)

try:
    import pyensembl
except ImportError:
    print("pyensembl not found. Install with: pip install pyensembl")
    exit(1)

try:
    import biotite.sequence as seq
    import biotite.sequence.io.fasta as fasta
    from biotite.sequence import ProteinSequence, NucleotideSequence
except ImportError:
    print("biotite not found. Install with: pip install biotite")
    exit(1)

from rocketshp import config


def parse_clinvar_pathogenic_variants(vcf_path: Path) -> List[Dict]:
    """
    Parse ClinVar VCF file and extract pathogenic variants.
    
    Args:
        vcf_path: Path to the ClinVar VCF file
        
    Returns:
        List of dictionaries containing pathogenic variant information
    """
    pathogenic_variants = []
    
    with gzip.open(vcf_path, 'rt') as f:
        reader = vcfpy.Reader(f)
        
        for record in tqdm(reader, desc="Parsing ClinVar VCF"):
            # Check if variant has clinical significance info
            if "CLNSIG" not in record.INFO:
                continue
                
            # Check if any clinical significance contains "Pathogenic"
            clnsig_values = record.INFO["CLNSIG"]
            is_pathogenic = False
            
            for sig in clnsig_values:
                if "Pathogenic" in sig and "Benign" not in sig:
                    is_pathogenic = True
                    break
                    
            if not is_pathogenic:
                continue
                
            # Extract variant information
            for alt_allele in record.ALT:
                variant_info = {
                    "chrom": record.CHROM,
                    "pos": record.POS,
                    "id": record.ID[0] if record.ID else None,
                    "ref": record.REF,
                    "alt": str(alt_allele),
                    "clnsig": clnsig_values,
                    "qual": record.QUAL,
                    "filter": record.FILTER
                }
                
                # Add additional INFO fields if available
                for info_field in ["CLNDN", "CLNHGVS", "GENEINFO"]:
                    if info_field in record.INFO:
                        variant_info[info_field.lower()] = record.INFO[info_field]
                        
                pathogenic_variants.append(variant_info)
    
    return pathogenic_variants


def get_transcripts_at_position(chrom: str, pos: int, genome_version: str = "GRCh38") -> List[Dict]:
    """
    Get all transcripts overlapping a genomic position using pyensembl.
    
    Args:
        chrom: Chromosome (e.g., "1", "X", "Y")
        pos: Genomic position (1-based)
        genome_version: Genome version ("GRCh37" or "GRCh38")
        
    Returns:
        List of transcript information dictionaries
    """
    try:
        # Initialize genome
        if genome_version == "GRCh38":
            genome = pyensembl.EnsemblRelease(108)  # Latest GRCh38
        else:
            genome = pyensembl.EnsemblRelease(75)   # GRCh37
            
        # Ensure genome data is downloaded
        genome.download()
        genome.index()
        
        # Convert chromosome format
        if chrom.startswith("chr"):
            chrom = chrom[3:]
            
        # Get transcripts at position
        transcripts = genome.transcripts_at_locus(contig=chrom, position=pos)
        
        transcript_info = []
        for transcript in transcripts:
            info = {
                "transcript_id": transcript.id,
                "transcript_name": transcript.name,
                "gene_id": transcript.gene.id,
                "gene_name": transcript.gene.name,
                "strand": transcript.strand,
                "start": transcript.start,
                "end": transcript.end,
                "biotype": transcript.biotype if hasattr(transcript, 'biotype') else None,
                "is_protein_coding": transcript.is_protein_coding
            }
            transcript_info.append(info)
            
        return transcript_info
        
    except Exception as e:
        print(f"Error getting transcripts for {chrom}:{pos}: {e}")
        return []


def apply_mutation_to_sequence(ref_seq: str, pos: int, ref_allele: str, alt_allele: str) -> str:
    """
    Apply a point mutation to a DNA sequence.
    
    Args:
        ref_seq: Reference DNA sequence
        pos: Position in sequence (0-based)
        ref_allele: Reference allele
        alt_allele: Alternative allele
        
    Returns:
        Mutated sequence
    """
    if pos < 0 or pos >= len(ref_seq):
        raise ValueError(f"Position {pos} is out of sequence bounds")
        
    # For simple substitutions
    if len(ref_allele) == 1 and len(alt_allele) == 1:
        mutated_seq = ref_seq[:pos] + alt_allele + ref_seq[pos + 1:]
    # For deletions
    elif len(ref_allele) > len(alt_allele):
        del_length = len(ref_allele) - len(alt_allele)
        mutated_seq = ref_seq[:pos] + alt_allele + ref_seq[pos + len(ref_allele):]
    # For insertions
    elif len(alt_allele) > len(ref_allele):
        mutated_seq = ref_seq[:pos + len(ref_allele)] + alt_allele[len(ref_allele):] + ref_seq[pos + len(ref_allele):]
    else:
        mutated_seq = ref_seq[:pos] + alt_allele + ref_seq[pos + len(ref_allele):]
        
    return mutated_seq


def get_transcript_sequences_with_mutation(variant: Dict, transcript_info: Dict, 
                                         genome_version: str = "GRCh38") -> Optional[Dict]:
    """
    Get DNA and protein sequences for a transcript with mutation applied.
    
    Args:
        variant: Variant information dictionary
        transcript_info: Transcript information dictionary
        genome_version: Genome version
        
    Returns:
        Dictionary with original and mutated sequences
    """
    try:
        # Initialize genome
        if genome_version == "GRCh38":
            genome = pyensembl.EnsemblRelease(108)
        else:
            genome = pyensembl.EnsemblRelease(75)
            
        # Get transcript object
        transcript = genome.transcript_by_id(transcript_info["transcript_id"])
        
        # Skip non-protein-coding transcripts
        if not transcript.is_protein_coding:
            return None
            
        # Get sequences
        try:
            dna_sequence = transcript.sequence
            protein_sequence = transcript.protein_sequence
        except Exception as e:
            print(f"Could not get sequences for transcript {transcript_info['transcript_id']}: {e}")
            return None
            
        # Calculate position in transcript coordinates
        genomic_pos = variant["pos"]
        transcript_pos = None
        
        # Convert genomic position to transcript position
        for i, exon in enumerate(transcript.exons):
            if exon.start <= genomic_pos <= exon.end:
                # Position is within this exon
                if transcript.strand == "+":
                    transcript_pos = sum(len(e) for e in transcript.exons[:i]) + (genomic_pos - exon.start)
                else:
                    transcript_pos = sum(len(e) for e in transcript.exons[i+1:]) + (exon.end - genomic_pos)
                break
                
        if transcript_pos is None:
            # Variant is not in an exon
            return None
            
        # Apply mutation to DNA sequence
        try:
            mutated_dna = apply_mutation_to_sequence(
                dna_sequence, transcript_pos, variant["ref"], variant["alt"]
            )
        except Exception as e:
            print(f"Could not apply mutation to DNA sequence: {e}")
            return None
            
        # Translate mutated DNA to protein
        try:
            # Use biotite for translation
            dna_biotite = NucleotideSequence(mutated_dna)
            protein_biotite = dna_biotite.translate(complete=False)
            mutated_protein = str(protein_biotite)
        except Exception as e:
            print(f"Could not translate mutated sequence: {e}")
            mutated_protein = None
            
        return {
            "transcript_id": transcript_info["transcript_id"],
            "gene_name": transcript_info["gene_name"],
            "original_dna": dna_sequence,
            "mutated_dna": mutated_dna,
            "original_protein": protein_sequence,
            "mutated_protein": mutated_protein,
            "mutation_position_transcript": transcript_pos,
            "mutation_position_protein": transcript_pos // 3 if transcript_pos is not None else None
        }
        
    except Exception as e:
        print(f"Error processing transcript {transcript_info['transcript_id']}: {e}")
        return None


def main():
    """Main function to extract pathogenic variants and their transcript sequences."""
    
    # Set up paths
    clinvar_vcf_path = config.PROCESSED_DATA_DIR / "clinvar" / "clinvar.vcf.gz"
    
    if not clinvar_vcf_path.exists():
        print(f"ClinVar VCF file not found at {clinvar_vcf_path}")
        return
        
    print("Parsing ClinVar VCF for pathogenic variants...")
    pathogenic_variants = parse_clinvar_pathogenic_variants(clinvar_vcf_path)
    print(f"Found {len(pathogenic_variants)} pathogenic variants")
    
    # Process variants and get transcript information
    results = []
    
    for variant in tqdm(pathogenic_variants[:100], desc="Processing variants"):  # Limit to first 100 for testing
        # Get transcripts at this position
        transcripts = get_transcripts_at_position(variant["chrom"], variant["pos"])
        
        if not transcripts:
            continue
            
        # Process each transcript
        for transcript_info in transcripts:
            if not transcript_info["is_protein_coding"]:
                continue
                
            sequence_info = get_transcript_sequences_with_mutation(
                variant, transcript_info
            )
            
            if sequence_info:
                result = {
                    **variant,
                    **sequence_info
                }
                results.append(result)
    
    # Convert to DataFrame and save
    if results:
        df = pd.DataFrame(results)
        output_path = config.PROCESSED_DATA_DIR / "clinvar" / "pathogenic_variants_with_sequences.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved {len(results)} results to {output_path}")
        
        # Display summary
        print(f"\nSummary:")
        print(f"Unique variants: {df['id'].nunique()}")
        print(f"Unique genes: {df['gene_name'].nunique()}")
        print(f"Total transcript-variant pairs: {len(df)}")
        
        return df
    else:
        print("No results found")
        return None


if __name__ == "__main__":
    results = main()