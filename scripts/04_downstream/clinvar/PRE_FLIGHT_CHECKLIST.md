# ClinVar Pipeline Pre-Flight Checklist

## ✅ Critical Fixes Applied

The following critical bugs have been **FIXED**:

1. ✅ **Community detection error handling** - Won't crash on small/disconnected networks
2. ✅ **HDF5 attribute serialization** - Numpy types converted to Python types
3. ✅ **Sequence-structure length verification** - Added assertion for variant predictions
4. ✅ **SNV-only documentation** - Clearly commented in code

## ⚠️ MUST DO Before Running

### 1. ✅ Distance Units - VERIFIED CORRECT

**Status:** RocketSHP predicts distances in **nanometers** (confirmed by developer)
- The network building code is already correct
- Default cutoff: 8 Å → 0.8 nm (appropriate for protein contacts)
- No changes needed

Optional: You can still run the test script to verify on your system:
```bash
python scripts/04_downstream/clinvar/test_distance_units.py kras_afdb.pdb cuda:0
```

---

### 2. Ensure Dependencies are Installed

```bash
cd $HOME/Projects/rocketshp
uv sync
```

This ensures all required packages (especially `vcfpy`, `pyensembl`, `scikit-learn`) are installed.

---

### 3. Check ClinVar VCF File Exists

```bash
ls -lh /mnt/home/ssledzieski/database/clinvar/clinvar.vcf.gz
```

If missing, download from:
- https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz

---

### 4. Install and Index Ensembl Genome

The first run will download ~10GB of Ensembl data. You can pre-install:

```python
import pyensembl
genome = pyensembl.EnsemblRelease(113, species="human")
genome.download()
genome.index()
```

Or let Step 1 handle it automatically (will take longer on first run).

---

## 📋 Running the Pipeline

### Quick Start
```bash
cd scripts/04_downstream/clinvar
./submit_all_pipeline.sh
```

### Monitor Progress
```bash
# Check job status
squeue -u $USER

# Check pipeline status
./check_pipeline_status.sh

# View logs
tail -f slurm_logs/clinvar_*.out
```

---

## 🔍 After Step 1: Data Quality Checks

Once Step 1 completes, **BEFORE** proceeding to Steps 2-3:

### 1. Check Sample Sizes

```bash
cd data/processed/clinvar
head -1 variant_dataset.csv
wc -l variant_dataset.csv
```

**Expected:**
- At least 1000-5000 variants total
- Reasonable distribution across pathogenic/benign/VUS

### 2. Spot Check Known Variants

Verify a few well-known variants mapped correctly:

```python
import pandas as pd
df = pd.read_csv("data/processed/clinvar/variant_dataset.csv")

# Example: Check BRAF V600E
braf = df[(df['gene_name'] == 'BRAF') & (df['protein_pos'] == 600)]
print(braf[['variant_id', 'wt_aa', 'variant_aa', 'pathogenicity']])
# Should show V→E mutation, pathogenic

# Example: Check a benign variant you know
# Add your own test cases here
```

### 3. Check for Duplicates

```python
# Check for same variant mapped to multiple transcripts
duplicates = df.groupby(['gene_name', 'protein_pos', 'wt_aa', 'variant_aa']).size()
duplicates = duplicates[duplicates > 1]
print(f"Variants with multiple transcripts: {len(duplicates)}")

# If too many (>20% of total), consider filtering to canonical transcripts only
```

### 4. Review Summary Statistics

```bash
# Check the log file for summary stats
tail -50 slurm_logs/clinvar_01_prepare*.out
```

Should show:
- Total variants processed
- Pathogenicity distribution
- Protein length distribution
- pLDDT statistics

**If sample size is too small (<500 per category):**
- Consider relaxing pLDDT threshold (e.g., 60 instead of 70)
- Consider including more variant types (requires code changes)
- Consider reducing terminus_buffer (5 → 3 residues)

---

## 🐛 Known Limitations

### Scientific Limitations:
1. **SNVs only** - No insertions/deletions/frameshifts
2. **Protein-coding only** - No UTR, intronic, or regulatory variants
3. **Wild-type structure** - Doesn't model structural changes from mutation
4. **Single isoform** - May not capture isoform-specific effects
5. **AlphaFold quality** - Limited by structure prediction accuracy

### Data Limitations:
1. **AlphaFold coverage** - Not all proteins have structures
2. **UniProt mapping** - Some genes may not map to UniProt
3. **Transcript ambiguity** - Multiple transcripts may create duplicates
4. **Stop codons** - Nonsense variants included (may affect predictions)

These are acceptable for an exploratory analysis but should be clearly documented in any publication.

---

## 📊 Expected Runtime & Resources

| Step | Time | Peak Memory | GPU | Critical? |
|------|------|-------------|-----|-----------|
| 1 | ~6-12 hours | 32-64GB | No | Must check results |
| 2 | ~1-2 days | 96-128GB | Yes | Can resume |
| 3 | ~1-2 days | 96-128GB | Yes | Can resume |
| 4 | ~2-4 hours | 16-32GB | No | Fast |
| 5 | ~1-2 hours | 16-32GB | No | Fast |
| 6 | ~1-2 hours | 16-32GB | No | Fast |
| 7 | ~30 min | 8-16GB | No | Fast |

**Total:** ~4-5 days end-to-end (Steps 2 & 3 run in parallel)

Steps 2 and 3 have checkpointing - if they timeout, just resubmit and they'll resume.

---

## 🆘 Troubleshooting

### Job fails immediately with module error
```bash
# Check if uv module is available
module avail uv

# If not, you may need to use python directly:
# Edit sbatch files to replace:
#   module load uv
#   uv run python ...
# With:
#   module load python/3.11.7
#   source .venv/bin/activate
#   python ...
```

### Step 1 takes too long (>24 hours)
- Downloading Ensembl data for first time is slow
- AlphaFold structure downloads can timeout
- Consider running with smaller test set first

### Out of memory in Steps 2-3
- Reduce `--mem` expectation if needed
- Large proteins (>1000 residues) use more memory
- Consider filtering dataset for protein_length < 1000

### Zero variants after filtering
- Check ClinVar VCF path is correct
- Check internet connection (for UniProt/AlphaFold downloads)
- Review logs for specific errors
- Try relaxing quality filters

---

## 📝 Final Checklist

Before running `./submit_all_pipeline.sh`:

- [x] Distance units verified correct (nm) - no action needed
- [ ] Ran `uv sync` to install dependencies
- [ ] Verified ClinVar VCF file exists
- [ ] Created `slurm_logs/` directory (auto-created by scripts)
- [ ] Read and understood known limitations
- [ ] Reviewed expected resource requirements
- [ ] Cluster has available GPU nodes

After Step 1 completes:

- [ ] Checked sample sizes (>500 per category)
- [ ] Spot-checked known variants
- [ ] Reviewed pathogenicity distribution
- [ ] Checked for excessive duplicates
- [ ] Ready to proceed with Steps 2-3

Good luck! 🚀
