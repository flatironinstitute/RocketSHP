<div align="center">

# 🚀 PLANET-MD

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

<img src="logo.png" width="300">

</div>

PLANET-MD enables ultra-fast prediction of protein dynamics and flexibility from amino acid sequences and/or protein structures. Trained on thousands of molecular dynamics trajectories, it predicts multiple dynamics-related features simultaneously:

- Root-Mean-Square Fluctuations (RMSF)
- Generalized Correlation Coefficients with Linear Mutual Information (GCC-LMI)
- Structural Heterogeneity Profiles (SHP)

This approach bridges the gap between static structural biology and dynamic functional understanding, providing a computational tool that complements experimental approaches at unprecedented speed and scale.


## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/flatironinstitute/PLANET-MD.git
cd PLANET-MD

# Create and activate a conda environment
mamba create -n planet_md python=3.11
mamba activate planet_md

# Install package
pip install -e .

# Check installation
python -c "import planet_md; print('Success!')"
```

## 🚀 Quick Start

With a `.pdb` file from the command line:

```bash
planet_md_predict example/kras_afdb.pdb PLANET_MD_KRAS_Predictions
```

Python interface:

```python
import torch
from torch.nn.functional import softmax
from planet_md import PlanetMD, load_sequence, load_structure
from planet_md.plot import plot_predictions

# Set compute device
device = torch.device("cuda:0")

# Load the model (will download if not present)
model = PlanetMD.load_from_checkpoint("v1_seq").to(device)

# Predict dynamics from sequence only
sequence = "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKCDLPSRTVDTKQAQDLARSYGIPFIETSAKTRQRVEDAFYTLVREIRQYRLKKISKEEKTPGCVKIKKCIIM"
seq_features = load_sequence(sequence, device=device)
with torch.no_grad():
    dynamics_pred = model({"seq_feats": seq_features, "struct_feats": None})

# Access prediction results
rmsf = dynamics_pred["rmsf"].squeeze().cpu().numpy()
gcc_lmi = dynamics_pred["gcc_lmi"].squeeze().cpu().numpy()
ca_dist = dynamics_pred["ca_dist"].squeeze().cpu().numpy()
shp = softmax(dynamics_pred["shp"].squeeze(), dim=1).cpu().numpy()

# Visualize results
plot_predictions(rmsf, gcc_lmi, shp, "KRAS", "planet_md_kras.png")
```

## 💻 Model Variants

PLANET-MD comes in three variants:

1. **PLANET-MD** - Full model that uses both sequence and structure information
2. **PLANET-MD-seq** - Sequence-only model trained with the same architecture
3. **PLANET-MD-mini** - Lightweight sequence-only model (1.5M parameters)

```python
# Load different model variants
full_model = PlanetMD.load_from_checkpoint("latest")  # Latest is also 'v1'
seq_model = PlanetMD.load_from_checkpoint("v1_seq")
mini_model = PlanetMD.load_from_checkpoint("v1_mini")
```

### 📈 Using a Structure for Improved Predictions

```python
import torch
from torch.nn.functional import softmax
from planet_md import PlanetMD, load_sequence, load_structure
from biotite.structure.io import pdb
from biotite.structure import to_sequence

# Set compute device
device = torch.device("cuda:0")

# Load the model
model = PlanetMD.load_from_checkpoint("latest").to(device)

# Load structure file (PDB)
structure_file = "example/kras_afdb.pdb"
structure = pdb.PDBFile.read(structure_file).get_structure()
struct_features = load_structure(structure, device=device)

# Get sequence from structure
sequence = str(to_sequence(structure)[0][0])
seq_features = load_sequence(sequence, device=device)

# Predict dynamics with both sequence and structure
with torch.no_grad():
    dynamics_pred = model({
        "seq_feats": seq_features,
        "struct_feats": struct_features,
    })
```

## 🕸️ Building an Allosteric Network

```python
from planet_md.network import build_allosteric_network, cluster_network, calculate_centrality, plot_network_clusters

# Build network from GCC-LMI predictions and distance mask
gcc_lmi = dynamics_pred["gcc_lmi"].squeeze().cpu().numpy()
ca_dist = dynamics_pred["ca_dist"].squeeze().cpu().numpy()  # Predicted CA distances
network = build_allosteric_network(gcc_lmi, ca_dist, distance_cutoff=10.0)

# Apply clustering to identify communities
communities = cluster_network(network, k=5)

# Calculate betweenness centrality
centralities = calculate_centrality(network)
betweenness_centrality = centralities["betweenness"]
closeness_centrality = centralities["closeness"]
degree_centrality = centralities["degree"]

# Visualize network
from planet_md.network import plot_network_clusters
plot_network_clusters(network, communities, output_path="planet_md_kras_network.png")
```

## 🔎 Model Details

PLANET-MD leverages ESM3 for sequence and structure representations, feeding these through a transformer encoder to predict multiple dynamics properties:

- **Architecture**: Transformer encoder with specialized prediction heads
- **Sequence Input**: ESM3 embeddings (frozen)
- **Structure Input**: ESM3 structure embeddings (optional, three variants available)
- **Output Heads**:
  - RMSF: Linear prediction of per-residue flexibility
  - GCC-LMI: Pairwise correlation estimation with sigmoid activation
  - SHP: Categorical distribution over structure tokens representing different conformational states

For more technical details, please refer to our paper.

## 📝 Citation

If you use PLANET-MD in your research, please cite:

```bibtex
@article {sledzieski2026planetmd,
	author = {Sledzieski, Samuel and Hanson, Sonya M.},
	title = {PLANET-MD: Ultra-fast Proteome-scale Prediction of Allosteric Networks in Proteins},
	elocation-id = {2025.06.12.659353},
	year = {2026},
	doi = {10.1101/2025.06.12.659353},
	publisher = {Cold Spring Harbor Laboratory},
	journal = {bioRxiv}
}
```

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/flatironinstitute/PLANET-MD?tab=MIT-1-ov-file#readme) file for details.
