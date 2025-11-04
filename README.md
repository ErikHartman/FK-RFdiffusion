# FK-RFdiffusion

![](rfd_fk.png)

Feynman-Kac guided protein design using RFdiffusion. This package implements particle filtering to optimize design objectives during the diffusion process.

## Overview

FK-RFdiffusion extends [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion) with Feynman-Kac particle filtering for guided protein design. Instead of blind sampling, it guides the generative process toward desired properties like:

- **Binding affinity** (interface ΔG)
- **Secondary structure**
- **Sequence properties**

The method uses multiple particles that are resampled based on reward functions evaluated during the diffusion trajectory.

## Installation

### 1. Set up RFdiffusion Environment

First, follow the [RFdiffusion installation instructions](https://github.com/RosettaCommons/RFdiffusion) to set up the base environment. 

### 2. Clone this repository with submodules

```bash
git clone --recursive https://github.com/YOUR_USERNAME/fk-rfdiffusion.git
cd fk-rfdiffusion
```

If you already cloned without `--recursive`, initialize submodules:

```bash
git submodule update --init --recursive
```

### 3. Install RFdiffusion

```bash
cd externals/RFdiffusion
pip install -e . --no-deps

# Install SE(3) Transformer
cd env/SE3Transformer
pip install -r requirements.txt
pip install .
cd ../../..
```

### 4a. Install additional dependencies for FK-RFdiffusion

```bash
pip install pydssp biopython
```

### 4b. Install PyRosetta

### 5. Download RFdiffusion weights

```bash
cd externals/RFdiffusion
mkdir -p models && cd models
wget http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt

cd ../../..
```

### 6. Install ProteinMPNN

```bash
cd externals/ProteinMPNN
pip install -e .
cd ../..
```

## Quick Start

### Binder Design with Interface Energy Guidance

```python
from fk_rfdiffusion.run_inference_guided import run_feynman_kac_design

run_feynman_kac_design(
    contigs=["A1-50/0 20"],           # Target residues A1-50, then 20-residue binder
    target_structure="target.pdb",     # Your target protein
    reward_function="interface_dG",    # Optimize binding energy
    num_designs=10,
    n_particles=20,                    # Number of parallel particles
    resampling_frequency=5,            # Resample every 5 steps
    guidance_start_timestep=30,        # Start guiding at timestep 30
    output_prefix="./designs/binder"
)
```

### Unconditional Design with Secondary Structure Guidance

```python
run_feynman_kac_design(
    contigs=["50"],                    # Design a 50-residue protein
    reward_function="alpha_helix_ss",  # Favor alpha helical structure
    num_designs=5,
    n_particles=15,
    output_prefix="./designs/helix"
)
```


## Available Reward Functions (current)

- `interface_dG` - Binding energy (lower is better)
- `alpha_helix_ss` - Alpha helix secondary structure content
- `beta_sheet_ss` - Beta sheet secondary structure content
- `loop_ss` - Loop/coil secondary structure content
- `sequence_hydrophobic` - Hydrophobic residue content
- `sequence_charged_positive` - Positive charge content
- `sequence_charged_negative` - Negative charge content

See `fk_rfdiffusion/feynman_kac/reward/configs/presets.yaml` for full configuration options.

## Key Parameters

- `n_particles` - Number of parallel particles (more = better exploration, slower)
- `resampling_frequency` - How often to resample particles (lower = more guidance)
- `guidance_start_timestep` - When to start applying guidance (try 20-50)
- `potential_mode` - How to compute weights:
  - `"difference"` (default) - Based on reward improvement
  - `"immediate"` - Based on current reward
  - `"sum"` - Based on cumulative future reward
  - `"max"` - Based on maximum future reward
- `tau` - Temperature parameter for particle selection (auto-set per reward if None)

## Advanced Usage

### Custom Reward Functions

See `fk_rfdiffusion/feynman_kac/reward/` for examples of implementing custom reward functions.

### Multiple Sequence Evaluation

```python
run_feynman_kac_design(
    contigs=["A1-50/0 20"],
    target_structure="target.pdb",
    reward_function="interface_dG",
    n_sequences=5,              # Generate 5 sequences per structure
    aggregation_mode="mean",    # Average their rewards
    ...
)
```

## Citation

If you use this code, please cite:

- The RFdiffusion paper
- The ProteinMPNN paper
- This preprint

## License

MIT

## Troubleshooting

### ImportError: No module named 'rfdiffusion'

Make sure RFdiffusion is installed in editable mode: `pip install -e externals/RFdiffusion --no-deps`

### CUDA errors

Ensure your PyTorch and DGL installations match your CUDA version. Check with:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Reward function errors

Some reward functions require PyRosetta. See reward function documentation for dependencies.
