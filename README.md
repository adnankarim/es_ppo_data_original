# BBBC021 Ablation Study: ES vs PPO for Cellular Morphology Prediction

A comprehensive ablation study comparing **Evolution Strategies (ES)** and **Proximal Policy Optimization (PPO)** for training conditional diffusion models on cellular morphology prediction using the BBBC021 dataset.

## Overview

This project implements a conditional diffusion model (DDPM) that learns to predict how chemical perturbations affect cellular morphology. The model uses:

- **Condition A (Control Image)**: Source state - which specific cells to transform
- **Condition B (Perturbation Embedding)**: Target effect - which drug/perturbation to apply

The model learns a single function capable of generating any perturbation outcome by taking the specific drug identity (via Morgan fingerprint embedding) as input.

## Dataset: BBBC021

The [BBBC021 dataset](https://bbbc.broadinstitute.org/BBBC021) contains:
- **97,504 images** of MCF-7 breast cancer cells
- **113 chemical compounds** at 8 concentrations
- **3 channels**: DNA (DAPI), F-actin (Phalloidin), β-tubulin
- **26 Mode-of-Action (MoA) classes**
- **Image size**: 96×96 pixels (after preprocessing)

## Features

- ✅ **Conditional Diffusion Models** with dual conditioning (control image + perturbation embedding)
- ✅ **Batch-aware sampling** to avoid batch effect artifacts
- ✅ **Morgan fingerprint encoding** for chemical compound representation
- ✅ **Comprehensive ablation study** comparing ES vs PPO training methods
- ✅ **Real-time plotting** with 16-metric dashboard (updated every epoch)
- ✅ **Information-theoretic metrics** (MI, Entropy, KL Divergence)
- ✅ **Image quality metrics** (FID, MSE, MAE, SSIM)

## Installation

### Requirements

```bash
pip install torch torchvision
pip install numpy pandas matplotlib seaborn
pip install pillow scipy
pip install wandb  # Optional: for experiment tracking
pip install rdkit  # Optional: for Morgan fingerprints (fallback available)
```

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Resp
```

2. Prepare the BBBC021 dataset:
```bash
python prepare_bbbc021_data.py --output-dir ./data/bbbc021
```

For testing without real data, use synthetic mode:
```bash
python prepare_bbbc021_data.py --output-dir ./data/bbbc021 --synthetic-only
```

## Usage

### Running the Ablation Study

```bash
python run_bbbc021_ablation.py --data-dir data/bbbc021
```

### Command Line Arguments

**Data:**
- `--data-dir`: Directory containing BBBC021 data (default: `./data/bbbc021`)
- `--metadata-file`: Metadata CSV file (default: `metadata.csv`)

**Training:**
- `--ddpm-epochs`: DDPM pretraining epochs (default: 100)
- `--coupling-epochs`: Coupling training epochs (default: 30)
- `--warmup-epochs`: Warmup epochs before ES (default: 10)

**ES Ablation:**
- `--es-sigma-values`: ES sigma values (default: `0.001 0.005 0.01`)
- `--es-lr-values`: ES learning rate values (default: `0.0001 0.0005 0.001`)

**PPO Ablation:**
- `--ppo-kl-values`: PPO KL weight values (default: `0.3 0.5 0.7`)
- `--ppo-clip-values`: PPO clip values (default: `0.05 0.1 0.2`)
- `--ppo-lr-values`: PPO learning rate values (default: `2e-5 5e-5 1e-4`)

**Output:**
- `--output-dir`: Output directory (default: `bbbc021_ablation_results`)
- `--use-wandb`: Enable WandB logging
- `--seed`: Random seed (default: 42)
- `--no-reuse-pretrained`: Train pretrained model from scratch

### Example

```bash
# Full ablation study with custom parameters
python run_bbbc021_ablation.py \
    --data-dir data/bbbc021 \
    --ddpm-epochs 50 \
    --coupling-epochs 20 \
    --warmup-epochs 5 \
    --es-sigma-values 0.001 0.01 \
    --es-lr-values 0.0001 0.001 \
    --use-wandb \
    --seed 42
```

## Architecture

### Model Components

1. **Morgan Fingerprint Encoder**: Converts SMILES strings to 1024-bit binary vectors
2. **Perturbation Encoder**: Maps fingerprints to 256-dim embeddings
3. **Conditional U-Net**: 
   - Input: Concatenated [noisy_image, control_image] (6 channels)
   - Conditions: Time embedding + Perturbation embedding
   - Output: Predicted noise

### Training Phases

1. **Phase 1: DDPM Pretraining**
   - Trains unconditional DDPM on control images only
   - Learns basic image generation capabilities
   - Saves checkpoint for initialization

2. **Phase 2: Warmup (ES only)**
   - Initializes conditional model from pretrained weights
   - Uses gradient descent to stabilize before ES
   - Runs for `warmup_epochs` epochs

3. **Phase 3: Coupling Training**
   - **ES Mode**: Evolution Strategies (parameter perturbation + fitness evaluation)
   - **PPO Mode**: Proximal Policy Optimization (reconstruction loss + KL penalty)

### Batch-Aware Sampling

**Critical**: Control and perturbed images must come from the same experimental batch to avoid learning batch effects instead of perturbation effects. The `BatchPairedDataLoader` ensures this constraint.

## Results & Output

### Directory Structure

```
bbbc021_ablation_results/
├── pretrained_models/
│   └── ddpm_pretrained.pt
├── run_YYYYMMDD_HHMMSS/
│   ├── models/          # Model checkpoints
│   ├── plots/           # Visualization plots
│   │   ├── pretraining/
│   │   ├── ES_config_0/
│   │   └── PPO_config_0/
│   ├── logs/            # Training logs
│   ├── SUMMARY.txt      # Best configuration summary
│   └── all_results.json # All results in JSON format
```

### Plotting

The script generates comprehensive plots after every epoch:

- **16-metric dashboard** (4×4 grid):
  - Primary metrics: Loss, KL Divergence, Correlation, MAE
  - Image quality: FID, MSE, SSIM, Mutual Information
  - Information theory: Entropies, Conditional Entropies, KL components
  - Learned statistics: Mean, Std Dev vs targets
  - Text summary with latest values

- **Phase visualization**: Vertical dashed line separates warmup from training phases

- **CSV export**: All metrics saved to `{method}_metrics.csv`

### Metrics

**Image Quality:**
- **FID** (Fréchet Inception Distance): Lower is better
- **MSE** (Mean Squared Error): Lower is better
- **MAE** (Mean Absolute Error): Lower is better
- **SSIM** (Structural Similarity Index): Higher is better

**Information Theoretic:**
- **Mutual Information**: Measures dependency between real and generated
- **KL Divergence**: Measures distribution mismatch
- **Entropy**: Measures information content
- **Correlation**: Pearson correlation coefficient

## How It Works

### Conditional Diffusion

The model learns to transform control images into perturbed images:

1. **Input**: Control image (source) + Morgan fingerprint (drug ID)
2. **Process**: 
   - Forward diffusion: Add noise to perturbed image
   - Model predicts noise given: noisy_image + control_image + fingerprint
   - Backward diffusion: Remove predicted noise to reconstruct
3. **Output**: Generated perturbed image

### Training Process

```
For each batch:
  1. Sample perturbed image (e.g., Taxol-treated cell)
  2. Find control image from SAME batch
  3. Feed to model: Control + Noisy_Perturbed + Fingerprint
  4. Model predicts noise → Loss = MSE(predicted_noise, actual_noise)
  5. Update parameters (ES or PPO)
```

Because batches contain mixed perturbations (Taxol, Nocodazole, DMSO, etc.), the model must rely on the fingerprint to know which transformation to apply.

## Project Structure

```
Resp/
├── prepare_bbbc021_data.py    # Data preparation script
├── run_bbbc021_ablation.py     # Main ablation study script
├── data/
│   └── bbbc021/
│       ├── images/             # Cell images
│       └── metadata.csv        # Image metadata
├── bbbc021_ablation_results/  # Output directory
└── README.md                   # This file
```

## Citation

If you use this code, please cite:

- BBBC021 Dataset: [Broad Bioimage Benchmark Collection](https://bbbc.broadinstitute.org/BBBC021)
- DDPM: Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
- DDMEC: (Reference to your paper if applicable)

## License

[Specify your license here]

## Author

Adapted for CellFlux BBBC021 - December 2024

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch sizes in config
2. **RDKit not available**: Script will use hash-based fingerprint fallback
3. **Missing data**: Use `--synthetic-only` flag for testing
4. **WandB errors**: Disable with `--no-use-wandb` or install wandb

### Performance Tips

- Use GPU for training (automatically detected)
- Reduce `num_eval_samples` for faster evaluation
- Use `--reuse-pretrained` to skip pretraining if model exists
- Adjust `num_sampling_steps` for faster/slower generation

## Contributing

Adnan KARIM
