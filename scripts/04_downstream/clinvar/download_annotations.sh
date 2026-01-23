#!/bin/bash
# Download Ensembl GTF annotation for GRCh38

CLINVAR_DIR="/mnt/home/ssledzieski/Projects/rocketshp/data/processed/clinvar"

# Download Ensembl GTF
echo "Downloading Ensembl GTF annotation..."
wget -P "$CLINVAR_DIR" \
    ftp://ftp.ensembl.org/pub/release-113/gtf/homo_sapiens/Homo_sapiens.GRCh38.113.gtf.gz

# Download protein sequences from Ensembl (optional, for validation)
echo "Downloading Ensembl protein sequences..."
wget -P "$CLINVAR_DIR" \
    ftp://ftp.ensembl.org/pub/release-113/fasta/homo_sapiens/pep/Homo_sapiens.GRCh38.pep.all.fa.gz

echo "Done! Files downloaded to $CLINVAR_DIR"
