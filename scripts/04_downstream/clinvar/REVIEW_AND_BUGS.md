# ClinVar Pipeline - Code Review & Potential Issues

## Critical Issues to Address Before Running

### 1. ⚠️ CRITICAL: Variant Type Limitation
**Location:** `scripts/04_downstream/clinvar/01_prepare_clinvar_dataset.py:124-130`

**Issue:** Pipeline only handles Single Nucleotide Variants (SNVs), not insertions or deletions.

```python
# Only processes if len(variant["ref"]) == 1 and len(variant["alt"]) == 1
if len(variant["ref"]) == 1 and len(variant["alt"]) == 1:
    # Simple SNV
```

**Impact:**
- Indels (insertions/deletions) are silently skipped
- Reduces total number of analyzable variants significantly
- ClinVar has many pathogenic indels that won't be analyzed

**Recommendation:** Document this limitation prominently. Consider extending to handle indels in future iterations.

**Status:** ✅ ACCEPTABLE for initial analysis, but MUST be documented

---

### 2. ✅ VERIFIED: Distance Units are Correct
**Location:** `rocketshp/network.py:72-74`

**Status:** ✅ VERIFIED CORRECT

```python
def build_allosteric_network(gcc_lmi, ca_dist, distance_cutoff=8.0):
    dist_thresh_nm = distance_cutoff / 10.0  # Converts Angstroms to nanometers
    mask = ca_dist < dist_thresh_nm  # ca_dist is in nanometers
```

**Confirmation:**
- RocketSHP predicts `ca_dist` in **nanometers** (base unit for mdtraj and biotite)
- `distance_cutoff=8.0` is in **Angstroms** (standard for protein contacts)
- Division by 10 converts: 8.0 Å → 0.8 nm
- Default cutoff of 8 Å (0.8 nm) is appropriate for protein contact networks

**No action needed** - implementation is correct as-is.

---

### 3. ⚠️ HIGH: Genomic to Protein Coordinate Mapping (Negative Strand)
**Location:** `scripts/04_downstream/clinvar/01_prepare_clinvar_dataset.py:113-122`

**Issue:** Coordinate conversion for negative strand genes needs verification.

```python
if transcript.strand == "+":
    transcript_pos = sum(len(e) for e in transcript.exons[:i]) + (genomic_pos - exon.start)
else:
    transcript_pos = sum(len(e) for e in transcript.exons[i+1:]) + (exon.end - genomic_pos)
```

**Concern:**
- Negative strand coordinate mapping is complex
- pyensembl handles this internally, but our manual calculation might have off-by-one errors
- Could result in wrong amino acid being analyzed

**Recommendation:**
```python
# Alternative: Use pyensembl's built-in coordinate conversion
try:
    # pyensembl has methods for this
    transcript_offset = transcript.spliced_offset(genomic_pos)
    protein_pos = transcript_offset // 3
except:
    # Fallback to manual calculation
```

**Test Cases:**
- Known pathogenic variant on + strand gene (e.g., BRCA1)
- Known pathogenic variant on - strand gene (e.g., PTEN)
- Verify amino acid matches expected from literature

**Status:** ⚠️ SHOULD VERIFY with test cases

---

### 4. ⚠️ MEDIUM: Multiple Transcripts Per Gene
**Location:** `scripts/04_downstream/clinvar/01_prepare_clinvar_dataset.py:96-106`

**Issue:** Same variant mapped to multiple transcripts creates duplicate entries.

**Impact:**
- Statistical tests will have non-independent observations
- Same variant counted multiple times
- Could inflate significance

**Current Behavior:**
```python
for transcript in transcripts:  # Loops over ALL transcripts
    # Creates entry for each transcript
    results.append({
        "variant_id": variant_id,  # Same variant_id
        "transcript_id": transcript.id,  # Different transcript
        ...
    })
```

**Recommendation:**
```python
# Option 1: Filter to canonical transcript only
transcripts = [t for t in transcripts if t.is_canonical]

# Option 2: Keep longest transcript per gene
transcripts = sorted(transcripts, key=lambda t: len(t.protein_sequence), reverse=True)
transcripts = [transcripts[0]] if transcripts else []

# Option 3: Deduplicate in feature extraction by gene+position+mutation
```

**Status:** ⚠️ SHOULD FIX to avoid pseudo-replication

---

### 5. ⚠️ MEDIUM: Community Detection May Fail for Small/Disconnected Networks
**Location:** `scripts/04_downstream/clinvar/02_predict_wildtype.py:57-69`

**Issue:** Girvan-Newman algorithm with k=5 communities may fail for:
- Small proteins (<50 residues)
- Highly disconnected networks
- Networks with no edges

```python
communities = cluster_network(network, k=cfg["num_communities"])
```

**Current Error Handling:** None - will crash

**Fix:**
```python
try:
    communities = cluster_network(network, k=cfg["num_communities"])
except (nx.NetworkXError, StopIteration):
    # Network too small/disconnected for clustering
    # Assign all nodes to single community
    communities = (tuple(range(network.number_of_nodes())),)
    logger.warning(f"Network clustering failed for {uniprot_id}, using single community")
```

**Status:** ⚠️ SHOULD FIX to prevent crashes

---

### 6. ⚠️ MEDIUM: HDF5 Attribute Serialization
**Location:** `scripts/04_downstream/clinvar/02_predict_wildtype.py:148-150`

**Issue:** Numpy types may not serialize to HDF5 attributes correctly.

```python
for stat_key, stat_val in pred["network_stats"].items():
    grp.attrs[stat_key] = stat_val  # stat_val might be numpy type
```

**Fix:**
```python
for stat_key, stat_val in pred["network_stats"].items():
    # Convert numpy types to Python types
    if isinstance(stat_val, (np.integer, np.floating)):
        stat_val = stat_val.item()
    grp.attrs[stat_key] = stat_val
```

**Status:** ⚠️ SHOULD FIX to prevent serialization errors

---

### 7. ⚠️ LOW: Broad Exception Handling
**Location:** Multiple locations

**Issue:** Bare `except:` clauses hide bugs and make debugging difficult.

**Examples:**
```python
# Script 01
except Exception as e:
    logger.debug(f"Error processing transcript {transcript.id}: {e}")
    continue

# Script 04
except Exception as e:
    logger.error(f"Error extracting features for {variant_id}: {e}")
    return None
```

**Recommendation:** Catch specific exceptions or at minimum log full traceback:
```python
import traceback
except Exception as e:
    logger.error(f"Error: {e}")
    logger.debug(traceback.format_exc())
```

**Status:** ✅ ACCEPTABLE but could be improved

---

## Scientific/Statistical Issues

### 8. ⚠️ MEDIUM: Small Sample Size After Filters
**Location:** Entire pipeline

**Issue:** Multiple filters reduce sample size:
1. Only SNVs (~80% of variants lost)
2. Protein-coding only
3. pLDDT > 70
4. Protein length 50-2000
5. Exclude termini (first/last 5 residues)
6. AlphaFold structure available
7. UniProt ID mappable
8. Not conflicting pathogenicity

**Impact:**
- Final dataset might be <1000 variants per category
- Statistical power limited
- Some genes/diseases underrepresented

**Recommendation:**
- Run Step 1 first and check dataset size before continuing
- If too small, consider relaxing filters (e.g., pLDDT > 60)
- Report sample sizes prominently

**Status:** ⚠️ CHECK AFTER STEP 1

---

### 9. ⚠️ MEDIUM: Sequence-Structure Length Mismatch for Variants
**Location:** `scripts/04_downstream/clinvar/03_predict_variants.py:51-53`

**Issue:** Using variant sequence with WT structure assumes same length.

```python
# Uses variant sequence length
seq_features = load_sequence(variant_sequence, device=device)
# But WT structure length
struct_features = load_structure(structure, device=device)
```

**Impact:**
- Will crash if sequence and structure have different lengths
- Only safe for SNVs (which we filter for)

**Verification:**
```python
# Add length check
if len(variant_sequence) != len(wt_sequence):
    logger.error(f"Sequence length mismatch for {variant_id}")
    return None
```

**Status:** ✅ SAFE for SNV-only pipeline, but add assertion

---

### 10. ⚠️ LOW: Stop Codons Not Handled
**Location:** `scripts/04_downstream/clinvar/01_prepare_clinvar_dataset.py:127-137`

**Issue:** Variants introducing premature stop codons are not filtered out.

**Impact:**
- Truncated proteins in analysis
- Predictions for truncated sequence vs full-length structure
- Scientifically valid (truncations are pathogenic) but results need careful interpretation

**Recommendation:** Add flag for stop-gain variants:
```python
if "*" in mut_protein_seq and "*" not in protein_seq:
    # Variant introduces stop codon
    is_stop_gain = True
```

**Status:** ✅ ACCEPTABLE - these ARE pathogenic, worth analyzing

---

## Data Quality Checks

### 11. ✅ Position Indexing Consistency
**Status:** VERIFIED CORRECT

Script 01 stores as 1-indexed:
```python
"protein_pos": protein_pos + 1,  # Convert to 1-indexed
```

Script 04 converts back:
```python
protein_pos = int(variant_row["protein_pos"]) - 1  # Convert to 0-indexed
```

pLDDT extraction converts back:
```python
get_plddt_at_position(Path(variant_row["pdb_path"]), protein_pos + 1)
```

✅ **Consistent throughout**

---

### 12. ✅ Variant ID Uniqueness
**Status:** VERIFIED CORRECT

```python
"variant_id": record.ID[0] if record.ID else f"{record.CHROM}_{record.POS}_{record.REF}_{alt_allele}"
```

Includes alt_allele, ensuring multi-allelic sites get unique IDs.

---

### 13. ✅ HDF5 Group Name Sanitization
**Status:** VERIFIED CORRECT

```python
variant_group_name = variant_id.replace("/", "_").replace(":", "_")
```

Removes characters that would break HDF5 group names.

---

## Recommended Action Plan

### Before Running:

1. **MUST DO:**
   - [x] ~~Verify distance units (Issue #2)~~ - CONFIRMED CORRECT (nm)
   - [x] ~~Add length assertion for variant predictions (Issue #9)~~ - FIXED
   - [x] ~~Fix community detection error handling (Issue #5)~~ - FIXED

2. **SHOULD DO:**
   - [ ] Test coordinate mapping on known variants (Issue #3)
   - [ ] Implement canonical transcript filtering (Issue #4)
   - [x] ~~Fix HDF5 attribute serialization (Issue #6)~~ - FIXED

3. **NICE TO HAVE:**
   - [ ] Improve exception handling with tracebacks (Issue #7)
   - [ ] Add stop-gain flagging (Issue #10)

### After Step 1:

4. **CHECK:**
   - [ ] Inspect `variant_dataset.csv` for sample sizes
   - [ ] Verify pathogenicity distribution is reasonable
   - [ ] Spot-check 5-10 known variants for correct mapping
   - [ ] Check for duplicate variants (same gene+position+mutation)

### During Pipeline:

5. **MONITOR:**
   - [ ] Watch logs for warnings/errors
   - [ ] Check checkpoint files for progress
   - [ ] Verify intermediate file sizes are reasonable

---

## Test Recommendations

### Unit Tests for Critical Functions:

```python
def test_snv_mapping():
    """Test that known SNV maps to correct amino acid"""
    # BRAF V600E: chr7:140753336 A>T
    # Should map to protein position 600, V→E
    pass

def test_negative_strand():
    """Test negative strand gene mapping"""
    # PTEN: chromosome 10 negative strand
    # Pick known variant and verify mapping
    pass

def test_distance_units():
    """Verify ca_dist predictions are in expected units"""
    pass

def test_community_detection_edge_cases():
    """Test small/disconnected networks"""
    pass
```

---

## Documentation Gaps

Add to README or config:

1. **Limitations:**
   - SNVs only (no indels)
   - Protein-coding regions only
   - Requires AlphaFold structure
   - Uses WT structure for variants

2. **Assumptions:**
   - Distance units in nanometers
   - pLDDT > 70 is "high quality"
   - Terminus buffer of 5 residues is appropriate

3. **Interpretation notes:**
   - Results specific to single residue changes
   - Structure-based predictions may not capture all functional effects
   - Centrality changes assume network topology matters for function
