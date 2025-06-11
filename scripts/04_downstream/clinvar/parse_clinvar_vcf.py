#%%
from rocketshp import config
from tqdm import tqdm

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

#%% Run the parser
clinvar_vcf_path = config.PROCESSED_DATA_DIR / "clinvar" / "clinvar.vcf.gz"
records = parse_clinvar_vcf(clinvar_vcf_path)
# %%

pathogenic_records = [i for i in records if "CLNSIG" in i["info"] and "Pathogenic" in i["info"]["CLNSIG"]]
benign_records = [i for i in records if "CLNSIG" in i["info"] and "Benign" in i["info"]["CLNSIG"]]

clin_sig_types = []
for r in tqdm(records):
    if "CLNSIG" in r["info"]:
        clnsig = r["info"]["CLNSIG"]
        clnsig_clean = []
        for sig in clnsig:
            if "|" in sig:
                sigs = sig.split("|")
                clnsig_clean.extend(sigs)
            else:
                clnsig_clean.append(sig)
        clnsig_clean = [s.strip() for s in clnsig_clean if s.strip()]
        clin_sig_types.extend(clnsig_clean)
clin_sig_types = set(clin_sig_types)

# %%
