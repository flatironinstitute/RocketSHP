from rocketshp import config

clinvar_vcf_path = config.PROCESSED_DATA_DIR / "clinvar" / "clinvar.vcf.gz"

def parse_clinvar_vcf(vcf_path):
    """
    Parse a ClinVar VCF file and extract relevant information.

    Args:
        vcf_path (str): Path to the ClinVar VCF file.

    Returns:
        list: A list of dictionaries containing parsed ClinVar records.
    """
    import gzip
    from vcfpy import Reader

    records = []
    with gzip.open(vcf_path, 'rt') as f:
        reader = Reader(f)
        for record in reader:
            rec_dict = {
                "chrom": record.CHROM,
                "pos": record.POS,
                "id": record.ID,
                "ref": record.REF,
                "alt": [str(alt) for alt in record.ALT],
                "qual": record.QUAL,
                "filter": record.FILTER,
                "info": record.INFO,
            }
            records.append(rec_dict)
    
    return records

records = parse_clinvar_vcf(clinvar_vcf_path)