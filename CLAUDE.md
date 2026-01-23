# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RocketSHP is a deep learning tool for ultra-fast prediction of protein dynamics and flexibility from amino acid sequences and/or protein structures. It predicts multiple dynamics-related properties simultaneously:
- Root-Mean-Square Fluctuations (RMSF)
- Generalized Correlation Coefficients with Linear Mutual Information (GCC-LMI)
- Structural Heterogeneity Profiles (SHP)
- CA distance predictions

The project leverages ESM3 for frozen protein sequence/structure embeddings and trains a transformer-based architecture with multiple prediction heads.

## Development Setup

```bash
# Create environment
mamba create -n rocketshp python=3.11
mamba activate rocketshp

# Install package in editable mode
pip install -e .

# Update dependencies
make requirements

# Code formatting and linting
make lint          # Run ruff check and format on rocketshp, scripts, notebooks
make format        # Format code only
```

Python version: **3.11** (strictly required per pyproject.toml)

**IMPORTANT**: This project uses `uv` for dependency management. Always run Python commands using `uv run python` instead of `python` directly. For example:
- `uv run python script.py` instead of `python script.py`
- `uv run python -c "..."` for inline Python commands

## Key Commands

### Training
```bash
# Train model with config file
rocketshp_train <run_id> --config configs/<config_file>.yml --dataset [atlas|mdcath]

# Example
rocketshp_train my_experiment --config configs/default_config.yml --dataset atlas

# Debug mode (detailed tensor/module info)
rocketshp_train my_experiment --debug

# Hyperparameter sweep training
python rocketshp/interface/train_sweep.py
```

### Inference
```bash
# Predict from PDB file
rocketshp_predict <path_to_pdb> <output_prefix>

# Example
rocketshp_predict example/kras_afdb.pdb KRAS_predictions
```

### Code Quality
```bash
# Run linting (ruff check + format)
make lint

# Format only
make format

# Clean compiled files
make clean
```

No test suite is currently configured in this repository.

## Architecture

### Core Model Components

1. **Feature Extraction** (`rocketshp/esm3.py`, `rocketshp/features.py`)
   - ESM3 model loaded from HuggingFace (uses forked version: `git+https://github.com/samsledje/esm.git`)
   - Sequence embeddings: 1536-dim frozen ESM3 embeddings
   - Structure embeddings: ESM3 VQVAE quantized tokens (4096 codebook size) or encoded embeddings (1024-dim)
   - Requires HuggingFace token in `.env` file as `HF_TOKEN`

2. **Model Architecture** (`rocketshp/modeling/architectures.py`)
   - `JointStructAndSequenceEncoder`: Combines sequence and structure inputs
     - `SeqEncoder`: Linear projection of ESM3 embeddings to d_model
     - `StructEncoder`: Embedding layer for quantized structure tokens OR linear projection for encoded embeddings
   - `TransformerStack`: ESM3's transformer layers (n_layers, n_heads, d_model)
   - Four prediction heads:
     - RMSF: Linear head for per-residue flexibility
     - GCC-LMI: Pairwise correlation with sigmoid activation
     - CA Distance: Pairwise distance prediction
     - SHP: Categorical distribution over structure tokens (cross-entropy loss)

3. **Training Wrapper** (`rocketshp/modeling/pt_lightning.py`)
   - PyTorch Lightning module with multi-task loss
   - Loss balancing via alpha parameters (rmsf_alpha, gcc_lmi_alpha, ca_dist_alpha, shp_alpha)
   - Neptune logging integration for experiment tracking
   - Optional GradNorm for adaptive loss weighting

### Data Pipeline

**Datasets** (`rocketshp/data/`):
- `mdcath.py`: MDCATH molecular dynamics dataset
  - 5 temperatures: [320, 348, 379, 413, 450]K
  - 5 replicates per temperature
  - Processed data stored in HDF5: `data/processed/mdcath/mdcath_processed.h5`
  - Foldseek clustering for train/val/test splits
- `atlas.py`: ATLAS dataset (larger molecular dynamics dataset)
- `utils.py`: Base `MDDataset` and `MDDataModule` classes for PyTorch Lightning

**Data Format**:
- Features stored in HDF5 files with keys:
  - `seq_feats`: ESM3 sequence embeddings [N x 1536]
  - `struct_feats`: Structure tokens [N] or embeddings [N x 1024]
  - `rmsf`: Target RMSF values [N]
  - `gcc_lmi`: Target correlation matrix [N x N]
  - `ca_dist`: CA distance matrix [N x N]
  - `shp`: Structure heterogeneity tokens [N]

### Configuration System

**Config Files** (`configs/*.yml`):
- YAML-based configuration using OmegaConf
- Key parameters:
  - Model: `d_model`, `n_heads`, `n_layers`, `embedding_dim`, `struct_stage`
  - Training: `batch_size`, `lr`, `max_epochs`, `precision`
  - Data: `crop_size`, `train_pct`, `val_pct`, `num_data_workers`
  - Loss weights: `rmsf_alpha`, `gcc_lmi_alpha`, `ca_dist_alpha`, `shp_alpha`
- Default config: `configs/default_config.yml`
- Specialized configs for different model variants (mini, large, seq-only)

### Model Variants

Three primary model variants with different tradeoffs:
1. **v1** (latest/full): Uses both sequence and structure features - best accuracy
2. **v1_seq**: Sequence-only model - works without PDB structures
3. **v1_mini**: Lightweight model (1.5M parameters) - faster but less accurate

Models are loaded via `RocketSHP.load_from_checkpoint("variant_name")` which downloads from HuggingFace Hub if not cached locally.

## Project Structure

```
rocketshp/
├── rocketshp/                    # Main package
│   ├── modeling/                 # Model architectures and training
│   │   ├── architectures.py      # Core model classes
│   │   └── pt_lightning.py       # Lightning wrapper
│   ├── data/                     # Dataset loaders
│   │   ├── mdcath.py
│   │   ├── atlas.py
│   │   └── utils.py
│   ├── interface/                # CLI entry points
│   │   ├── train.py              # rocketshp_train command
│   │   ├── train_sweep.py        # Hyperparameter sweeps
│   │   └── predict.py            # rocketshp_predict command
│   ├── structure/                # 3D structure processing
│   │   └── protein_chain.py      # ProteinChain class
│   ├── esm3.py                   # ESM3 model loading
│   ├── features.py               # Feature extraction (load_sequence, load_structure)
│   ├── trajectory.py             # MD trajectory analysis
│   ├── network.py                # Allosteric network analysis
│   ├── metrics.py                # Evaluation metrics
│   ├── plot.py                   # Visualization utilities
│   ├── config.py                 # Paths and constants
│   └── utils.py                  # General utilities
├── scripts/                      # Analysis workflows
│   ├── 01_preprocess/            # Data preprocessing (atlas, mdcath)
│   ├── 02_train/                 # Training scripts
│   ├── 03_evaluate/              # Model evaluation
│   └── 04_downstream/            # Applications (KRAS, ClinVar, etc.)
├── configs/                      # Training configurations
├── data/                         # Data directories (raw, processed, interim)
├── models/                       # Saved model checkpoints
├── notebooks/                    # Jupyter analysis notebooks
└── checkpoints/                  # Training checkpoints
```

## Important Implementation Details

### ESM3 Integration
- Uses a **forked version** of ESM3: `git+https://github.com/samsledje/esm.git`
- ESM3 embeddings are **frozen** during training (not fine-tuned)
- Structure encoding supports two modes:
  - `struct_stage="quantized"`: Uses VQVAE discrete tokens (4096 codebook)
  - `struct_stage="encoded"`: Uses continuous structure embeddings (1024-dim)

### Device Handling
- Models support GPU/CPU via device parameter: `device = torch.device("cuda:0")` or `"cpu"`
- Device passthrough recently improved in commits
- Multi-GPU support via PyTorch Lightning's `devices` parameter

### Experiment Tracking
- Neptune.ai integration for logging (requires `NEPTUNE_API_TOKEN` in `.env`)
- CSV logging as fallback
- Logs stored in `logs/` directory with run_id subdirectories

### Data Preprocessing
Scripts in `scripts/01_preprocess/` handle:
- Computing ESM3 sequence embeddings
- Computing ESM3 structure embeddings (quantized or encoded)
- Calculating target dynamics properties from MD trajectories
- Building HDF5 datasets for efficient training

### Structural Heterogeneity (SHP)
- SHP represents discrete conformational states using structure tokens
- Trained via cross-entropy loss over ESM3's structure vocabulary
- Can be used to build allosteric networks via `rocketshp.network` module

## Environment Variables

Required in `.env` file:
```bash
HF_TOKEN=<huggingface_token>          # Required for ESM3 model download
NEPTUNE_API_TOKEN=<neptune_token>     # Optional, for experiment tracking
LOG_LEVEL=INFO                        # Optional, logging verbosity
```

## Common Workflows

### Adding a New Model Configuration
1. Copy `configs/default_config.yml` to `configs/<new_name>.yml`
2. Modify parameters (architecture, loss weights, etc.)
3. Train with: `rocketshp_train experiment_name --config configs/<new_name>.yml`

### Running Downstream Analysis
Scripts in `scripts/04_downstream/` contain application-specific analyses:
- `weng_KRAS/`: KRAS mutation studies
- `clinvar/`: Clinical variant annotation
- `na_channel/`: Ion channel dynamics
- `swissprot_inference/`, `human_proteome/`: Large-scale predictions

These scripts typically:
1. Load a trained model checkpoint
2. Generate predictions on specific protein sets
3. Perform domain-specific analysis and visualization

### Working with Protein Structures
The `rocketshp.structure.protein_chain.ProteinChain` class wraps biotite structure objects:
- Handles PDB file parsing
- Provides coordinate normalization
- Extracts residue-level features
- Integrates with ESM3 structure encoder

## Dependencies Note

Critical dependencies with version constraints:
- Python: 3.11 (strict)
- PyTorch: 2.0+ (implicitly via Lightning and ESM3)
- Lightning: 2.4.0+
- NumPy: <2.0 (compatibility constraint)
- ESM3: Custom fork from `samsledje/esm`
- MDAnalysis: 2.9.0+ (for trajectory analysis)

The forked ESM3 version is maintained separately and may diverge from the official EvolutionaryScale ESM repository.

## Coding Best Practices

**CRITICAL - Error Handling:**
- **NEVER use bare `except:` statements**. Always catch specific exceptions.
- Bare except blocks silence all errors including KeyboardInterrupt and SystemExit
- Good: `except (ValueError, TypeError) as e:`
- Bad: `except:`
- If you must catch all exceptions, use `except Exception as e:` and log the error

**CRITICAL - File Operations:**
- **NEVER run `rm` or `rm -rf` commands** without explicit user permission
- If files need to be deleted, ask the user to remove them
- User will verify the request before deleting

**General:**
- Use specific exception types when possible
- Always log or re-raise exceptions after handling
- Include error context in log messages
