#!/usr/bin/env python3
"""
Pre-flight verification script for ClinVar analysis pipeline.
Checks all prerequisites before submitting jobs.
"""

import os
import sys
from pathlib import Path

# Color codes for output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'
BOLD = '\033[1m'

def check(condition, message, warning=False):
    """Print check result with color coding."""
    if condition:
        print(f"  {GREEN}✓{RESET} {message}")
        return True
    else:
        color = YELLOW if warning else RED
        symbol = "⚠" if warning else "✗"
        print(f"  {color}{symbol}{RESET} {message}")
        return not warning  # Returns True for warnings (non-critical)

def main():
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}ClinVar Pipeline Pre-Flight Verification{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")

    all_good = True

    # 1. Check Python version
    print(f"{BOLD}1. Python Environment{RESET}")
    py_version = sys.version_info
    all_good &= check(
        py_version >= (3, 11),
        f"Python version {py_version.major}.{py_version.minor}.{py_version.micro} >= 3.11"
    )

    # 2. Check required packages
    print(f"\n{BOLD}2. Required Packages{RESET}")
    required_packages = [
        "vcfpy", "pyensembl", "biotite", "h5py", "networkx",
        "sklearn", "scipy", "pandas", "numpy", "torch",
        "loguru", "omegaconf", "tqdm", "matplotlib", "seaborn"
    ]

    for package in required_packages:
        try:
            __import__(package)
            check(True, f"{package} installed")
        except ImportError:
            all_good &= check(False, f"{package} NOT installed")

    # 3. Check RocketSHP installation
    print(f"\n{BOLD}3. RocketSHP{RESET}")
    try:
        from rocketshp import RocketSHP, load_sequence, load_structure
        check(True, "RocketSHP package importable")

        # Try loading model (will download if needed)
        try:
            import torch
            device = torch.device("cpu")
            model = RocketSHP.load_from_checkpoint("latest")
            check(True, "RocketSHP model 'latest' accessible")
        except Exception as e:
            all_good &= check(False, f"RocketSHP model loading failed: {e}")
    except ImportError as e:
        all_good &= check(False, f"RocketSHP not importable: {e}")

    # 4. Check files and directories
    print(f"\n{BOLD}4. Files and Directories{RESET}")

    # Config file
    config_path = Path("configs/clinvar_analysis_config.yml")
    all_good &= check(config_path.exists(), f"Config file exists: {config_path}")

    # ClinVar VCF
    clinvar_vcf = Path("/mnt/home/ssledzieski/database/clinvar/clinvar.vcf.gz")
    vcf_exists = clinvar_vcf.exists()
    all_good &= check(vcf_exists, f"ClinVar VCF exists: {clinvar_vcf}")
    if vcf_exists:
        size_mb = clinvar_vcf.stat().st_size / (1024**2)
        check(size_mb > 100, f"  VCF size: {size_mb:.1f} MB", warning=True)

    # Output directories
    output_dir = Path("data/processed/clinvar")
    check(True, f"Output directory: {output_dir}", warning=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    reports_dir = Path("reports/figures/clinvar")
    check(True, f"Reports directory: {reports_dir}", warning=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # AlphaFold directory
    alphafold_dir = Path("/mnt/home/ssledzieski/database/alphafold")
    alphafold_dir.mkdir(parents=True, exist_ok=True)
    check(True, f"AlphaFold cache directory: {alphafold_dir}", warning=True)

    # 5. Check environment variables
    print(f"\n{BOLD}5. Environment Variables{RESET}")

    # Check for .env file
    env_file = Path(".env")
    if env_file.exists():
        check(True, ".env file exists")

        # Try to load and check HF_TOKEN
        try:
            from dotenv import load_dotenv
            load_dotenv()
            hf_token = os.getenv("HF_TOKEN")
            if hf_token:
                check(True, "HF_TOKEN set in .env")
            else:
                all_good &= check(False, "HF_TOKEN not set in .env (required for ESM3)")
        except Exception as e:
            check(False, f"Error loading .env: {e}", warning=True)
    else:
        all_good &= check(False, ".env file not found (needed for HF_TOKEN)")

    # Neptune token (optional)
    neptune_token = os.getenv("NEPTUNE_API_TOKEN")
    check(
        neptune_token is not None,
        "NEPTUNE_API_TOKEN set (optional, for experiment tracking)",
        warning=True
    )

    # 6. Check disk space
    print(f"\n{BOLD}6. Disk Space{RESET}")
    import shutil
    try:
        usage = shutil.disk_usage(Path.cwd())
        free_gb = usage.free / (1024**3)
        check(
            free_gb > 50,
            f"Available disk space: {free_gb:.1f} GB (need ~50GB for pipeline)"
        )
    except Exception as e:
        check(False, f"Could not check disk space: {e}", warning=True)

    # 7. Check pyensembl
    print(f"\n{BOLD}7. Pyensembl Genome Data{RESET}")
    try:
        import pyensembl
        genome = pyensembl.EnsemblRelease(113, species="human")

        # Check if already downloaded
        try:
            genome.index()
            check(True, "Ensembl Release 113 already downloaded and indexed")
        except:
            check(
                True,
                "Ensembl Release 113 will be downloaded on first run (~10GB, may take 1-2 hours)",
                warning=True
            )
    except Exception as e:
        check(False, f"Pyensembl check failed: {e}", warning=True)

    # 8. Check cluster modules (if on SLURM)
    print(f"\n{BOLD}8. Cluster Environment{RESET}")
    if "SLURM_JOB_ID" in os.environ or shutil.which("sbatch"):
        check(shutil.which("sbatch") is not None, "SLURM available")

        # Check if uv module exists
        result = os.system("module avail uv 2>&1 | grep -q uv")
        check(result == 0, "UV module available", warning=True)

        result = os.system("module avail cuda 2>&1 | grep -q cuda")
        check(result == 0, "CUDA module available", warning=True)
    else:
        check(True, "Not running on SLURM cluster", warning=True)

    # Summary
    print(f"\n{BOLD}{'='*60}{RESET}")
    if all_good:
        print(f"{GREEN}{BOLD}✓ All critical checks passed!{RESET}")
        print(f"\n{GREEN}Ready to submit pipeline with:{RESET}")
        print(f"  cd scripts/04_downstream/clinvar")
        print(f"  ./submit_all_pipeline.sh")
    else:
        print(f"{RED}{BOLD}✗ Some critical checks failed!{RESET}")
        print(f"\n{RED}Please address the issues above before submitting.{RESET}")
        sys.exit(1)

    print(f"{BOLD}{'='*60}{RESET}\n")

    # Additional recommendations
    print(f"{YELLOW}Recommendations:{RESET}")
    print(f"  1. Review PRE_FLIGHT_CHECKLIST.md for full details")
    print(f"  2. After Step 1, check variant_dataset.csv for sample sizes")
    print(f"  3. Monitor logs in slurm_logs/ directory")
    print(f"  4. Use ./check_pipeline_status.sh to track progress")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Verification cancelled by user{RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n{RED}Unexpected error: {e}{RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
