# RocketSHP

🚧 This is pre-release code that is under active development 🚧

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Machine learning models of protein structure have transformed our understanding of protein function and enabled advances in modeling biological systems, drug discovery and design, and protein engineering. However, static structures are only one piece of the puzzle. Proteins are flexible molecules which are known to adopt diverse conformations, and these dynamics play a crucial role in their eventual function. While molecular dynamics simulations offer one high-powered approach to estimating these ensembles, they are presently too computationally expensive to apply at the scale of the whole proteome, let alone to isoforms, sequence and structural variants, or designed proteins.

To address this shortcoming, we introduce RocketSHP, a super-fast method for estimating protein flexibility and dynamics, requiring only amino acid sequence and, optionally, a static structure. Trained on over 100,000 trajectories from publicly-available dynamics data, we represent each ensemble as a compressed structural heterogeneity profile (SHP) using recent advances in structure quantization. We then train a protein language model to reconstruct these SHPs, along with the RMSF and autocorrelation of pairwise distances from the trajectory. This provides not only a single measure of flexibility but a higher dimensional measure of local motion for each residue. RocketSHP enables ultra-high throughput measures of conformational flexibility,  shedding light on the dynamic dimension of the "dark proteome". We anticipate that RocketSHP will enable advancements in dynamics-dependent tasks such as modeling protein interactions, engineering protein switches, or estimating pathogenicity of coding variants.

## Installation

```
pip3 install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip3 install .
```

## Data Preparation

```
...
```

## Training

```
rocketshp_train {JOB NAME} --config configs/default_config.yml
```

or on a SLURM cluster

```
sbatch scripts/02_train/submit_default_config.sbatch
```