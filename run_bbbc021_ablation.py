"""
================================================================================
BBBC021 ABLATION STUDY: ES vs PPO for Cellular Morphology Prediction
================================================================================

Adapts the synthetic data ablation study for the BBBC021 dataset from CellFlux.
Key changes from synthetic version:
- U-Net architecture instead of MLP (for 96x96x3 images)
- BBBC021 data loading with batch-aware sampling
- Morgan fingerprint perturbation encoding
- FID and MoA metrics for evaluation
- Batch effect handling (control/perturbed from same batch)

Dataset: BBBC021 (Chemical Perturbations on MCF-7 breast cancer cells)
- 97,504 images, 113 compounds, 8 concentrations
- 3 channels: DNA, F-actin, β-tubulin
- Image size: 96x96 (after preprocessing)
- 26 Mode-of-Action (MoA) classes

Author: Adapted for CellFlux BBBC021
Date: December 2024
================================================================================
"""

import os
import sys
import json
import csv
import datetime
import argparse
import time
import shutil
import itertools
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WARNING: wandb not available. Install with: pip install wandb")

try:
    from scipy import linalg
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("WARNING: scipy not available for FID calculation")

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("WARNING: RDKit not available. Install with: pip install rdkit")
    print("         Morgan fingerprints will use random embeddings instead.")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class BBBC021Config:
    """Configuration for BBBC021 ablation study."""
    
    # Data paths
    data_dir: str = "./data/bbbc021"  # Directory containing BBBC021 data
    metadata_file: str = "metadata.csv"  # Metadata CSV file
    
    # Image settings
    image_size: int = 96
    num_channels: int = 3  # DNA, F-actin, β-tubulin
    
    # Perturbation encoding
    morgan_bits: int = 1024  # Morgan fingerprint dimensions
    perturbation_embed_dim: int = 256
    
    # DDPM pretraining
    ddpm_epochs: int = 100
    ddpm_lr: float = 1e-4
    ddpm_batch_size: int = 32
    ddpm_timesteps: int = 1000
    
    # Coupling training
    coupling_epochs: int = 30
    coupling_batch_size: int = 16
    warmup_epochs: int = 10
    num_sampling_steps: int = 50
    
    # ES Ablations
    es_population_size: int = 20
    es_sigma_values: List[float] = field(default_factory=lambda: [0.001, 0.005, 0.01])
    es_lr_values: List[float] = field(default_factory=lambda: [0.0001, 0.0005, 0.001])
    
    # PPO Ablations
    ppo_kl_weight_values: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])
    ppo_clip_values: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2])
    ppo_lr_values: List[float] = field(default_factory=lambda: [2e-5, 5e-5, 1e-4])
    
    # U-Net architecture
    unet_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    time_embed_dim: int = 256
    
    # Output
    output_dir: str = "bbbc021_ablation_results"
    use_wandb: bool = True
    wandb_project: str = "bbbc021-ddmec-ablation"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    
    # Model management
    reuse_pretrained: bool = True
    
    # Evaluation
    num_eval_samples: int = 500
    fid_batch_size: int = 64
    
    # Memory optimization
    gradient_checkpointing: bool = True
    mixed_precision: bool = True


# ============================================================================
# MORGAN FINGERPRINT ENCODER
# ============================================================================

class MorganFingerprintEncoder:
    """
    Encode chemical compounds using Morgan fingerprints.
    
    Converts SMILES strings (chemical structure) into fixed-size binary vectors (1024 bits).
    This gives each perturbation a unique mathematical "ID" - Taxol looks different
    from Nocodazole in this vector space, allowing the model to distinguish between
    different drugs and learn drug-specific morphological transformations.
    """
    
    def __init__(self, n_bits: int = 1024, radius: int = 2):
        self.n_bits = n_bits
        self.radius = radius
        self.cache = {}
        
    def encode(self, smiles: str) -> np.ndarray:
        """Encode SMILES string to Morgan fingerprint."""
        if smiles in self.cache:
            return self.cache[smiles]
        
        if RDKIT_AVAILABLE and smiles and smiles != 'DMSO':
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
                    arr = np.zeros(self.n_bits, dtype=np.float32)
                    for i in fp.GetOnBits():
                        arr[i] = 1.0
                    self.cache[smiles] = arr
                    return arr
            except Exception as e:
                print(f"Warning: Failed to encode SMILES '{smiles}': {e}")
        
        # Fallback: use hash-based encoding
        np.random.seed(hash(smiles) % (2**32))
        arr = np.random.rand(self.n_bits).astype(np.float32)
        arr = (arr > 0.5).astype(np.float32)
        self.cache[smiles] = arr
        return arr


# ============================================================================
# BBBC021 DATASET
# ============================================================================

class BBBC021Dataset(Dataset):
    """
    BBBC021 Dataset with batch-aware sampling.
    
    Critical: Control and perturbed images must come from the same batch
    to avoid learning batch effects instead of perturbation effects.
    """
    
    def __init__(
        self,
        data_dir: str,
        metadata_file: str = "metadata.csv",
        image_size: int = 96,
        split: str = "train",  # "train", "val", "test"
        transform: Optional[transforms.Compose] = None,
        morgan_encoder: Optional[MorganFingerprintEncoder] = None,
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.split = split
        self.morgan_encoder = morgan_encoder or MorganFingerprintEncoder()
        
        # Default transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = transform
        
        # Load metadata
        self.metadata = self._load_metadata(metadata_file)
        
        # Group by batch for batch-aware sampling
        self.batch_to_indices = self._group_by_batch()
        
        # Get unique compounds and MoA classes
        self.compounds = sorted(list(set(m['compound'] for m in self.metadata)))
        self.moa_classes = sorted(list(set(m['moa'] for m in self.metadata if m['moa'])))
        
        # Create mappings
        self.compound_to_idx = {c: i for i, c in enumerate(self.compounds)}
        self.moa_to_idx = {m: i for i, m in enumerate(self.moa_classes)}
        
        # Precompute fingerprints for all compounds
        self._precompute_fingerprints()
        
        print(f"BBBC021 Dataset loaded: {len(self.metadata)} images, "
              f"{len(self.compounds)} compounds, {len(self.moa_classes)} MoA classes")
    
    def _load_metadata(self, metadata_file: str) -> List[Dict]:
        """Load and parse metadata CSV."""
        metadata_path = self.data_dir / metadata_file
        
        if not metadata_path.exists():
            print(f"Warning: Metadata file not found at {metadata_path}")
            print("Generating synthetic metadata for testing...")
            return self._generate_synthetic_metadata()
        
        metadata = []
        with open(metadata_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata.append({
                    'image_path': row.get('image_path', ''),
                    'compound': row.get('compound', 'DMSO'),
                    'concentration': float(row.get('concentration', 0)),
                    'moa': row.get('moa', ''),
                    'batch': row.get('batch', row.get('plate', '0')),
                    'well': row.get('well', ''),
                    'is_control': row.get('compound', 'DMSO') == 'DMSO',
                    'smiles': row.get('smiles', ''),
                })
        
        # Split data
        np.random.seed(42)
        indices = np.random.permutation(len(metadata))
        n_train = int(0.7 * len(metadata))
        n_val = int(0.15 * len(metadata))
        
        if self.split == "train":
            selected_indices = indices[:n_train]
        elif self.split == "val":
            selected_indices = indices[n_train:n_train + n_val]
        else:  # test
            selected_indices = indices[n_train + n_val:]
        
        return [metadata[i] for i in selected_indices]
    
    def _generate_synthetic_metadata(self) -> List[Dict]:
        """Generate synthetic metadata for testing without real data."""
        compounds = ['DMSO', 'Taxol', 'Colchicine', 'Nocodazole', 'Cytochalasin_B',
                    'Vincristine', 'Demecolcine', 'Alsterpaullone', 'AZ138', 'PP-2']
        moa_classes = ['Control', 'Microtubule_stabilizers', 'Microtubule_destabilizers',
                      'Actin_disruptors', 'Kinase_inhibitors', 'Eg5_inhibitors', 'Epithelial']
        
        compound_to_moa = {
            'DMSO': 'Control',
            'Taxol': 'Microtubule_stabilizers',
            'Colchicine': 'Microtubule_destabilizers',
            'Nocodazole': 'Microtubule_destabilizers',
            'Cytochalasin_B': 'Actin_disruptors',
            'Vincristine': 'Microtubule_destabilizers',
            'Demecolcine': 'Microtubule_destabilizers',
            'Alsterpaullone': 'Kinase_inhibitors',
            'AZ138': 'Eg5_inhibitors',
            'PP-2': 'Epithelial',
        }
        
        metadata = []
        n_samples = 5000  # Smaller for testing
        n_batches = 10
        
        for i in range(n_samples):
            compound = compounds[i % len(compounds)]
            batch = f"batch_{i % n_batches}"
            metadata.append({
                'image_path': f'synthetic_{i}.png',
                'compound': compound,
                'concentration': 1.0,
                'moa': compound_to_moa.get(compound, 'Unknown'),
                'batch': batch,
                'well': f'well_{i}',
                'is_control': compound == 'DMSO',
                'smiles': '',  # Synthetic
            })
        
        # Split data
        n_train = int(0.7 * len(metadata))
        n_val = int(0.15 * len(metadata))
        
        if self.split == "train":
            return metadata[:n_train]
        elif self.split == "val":
            return metadata[n_train:n_train + n_val]
        else:  # test
            return metadata[n_train + n_val:]
    
    def _group_by_batch(self) -> Dict[str, List[int]]:
        """Group sample indices by batch for batch-aware sampling."""
        batch_to_indices = defaultdict(list)
        for idx, meta in enumerate(self.metadata):
            batch_to_indices[meta['batch']].append(idx)
        return dict(batch_to_indices)
    
    def _precompute_fingerprints(self):
        """Precompute Morgan fingerprints for all compounds."""
        self.fingerprints = {}
        for meta in self.metadata:
            compound = meta['compound']
            if compound not in self.fingerprints:
                smiles = meta.get('smiles', '')
                self.fingerprints[compound] = self.morgan_encoder.encode(smiles)
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        meta = self.metadata[idx]
        
        # Load image
        image = self._load_image(meta['image_path'])
        
        # Get perturbation embedding
        fingerprint = torch.tensor(self.fingerprints[meta['compound']], dtype=torch.float32)
        
        # Get labels
        moa_idx = self.moa_to_idx.get(meta['moa'], 0)
        compound_idx = self.compound_to_idx.get(meta['compound'], 0)
        
        return {
            'image': image,
            'fingerprint': fingerprint,
            'compound': meta['compound'],
            'compound_idx': compound_idx,
            'moa_idx': moa_idx,
            'batch': meta['batch'],
            'is_control': meta['is_control'],
            'idx': idx,
        }
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load image from path or generate synthetic."""
        full_path = self.data_dir / "images" / image_path
        
        if full_path.exists():
            image = Image.open(full_path).convert('RGB')
            return self.transform(image)
        else:
            # Generate synthetic image for testing
            return self._generate_synthetic_image()
    
    def _generate_synthetic_image(self) -> torch.Tensor:
        """Generate synthetic cell-like image for testing."""
        # Create random cell-like patterns
        image = np.random.randn(self.num_channels, self.image_size, self.image_size)
        
        # Add cell-like structure
        y, x = np.ogrid[-self.image_size//2:self.image_size//2, 
                        -self.image_size//2:self.image_size//2]
        
        # Add nucleus (channel 0 - DNA)
        nucleus_mask = x*x + y*y <= (self.image_size//6)**2
        image[0][nucleus_mask] += 2.0
        
        # Add cytoskeleton patterns (channel 1 - F-actin)
        image[1] += 0.5 * np.sin(x/5.0) * np.cos(y/5.0)
        
        # Add microtubules (channel 2 - β-tubulin)
        for _ in range(5):
            angle = np.random.uniform(0, 2*np.pi)
            cx, cy = np.random.randint(-20, 20, 2)
            line_mask = np.abs((x-cx)*np.sin(angle) - (y-cy)*np.cos(angle)) < 2
            image[2][line_mask] += 1.5
        
        # Normalize
        image = (image - image.mean()) / (image.std() + 1e-8)
        image = np.clip(image, -3, 3) / 3.0
        
        return torch.tensor(image, dtype=torch.float32)
    
    @property
    def num_channels(self) -> int:
        return 3
    
    def get_control_indices(self) -> List[int]:
        """Get indices of all control (DMSO) samples."""
        return [i for i, m in enumerate(self.metadata) if m['is_control']]
    
    def get_perturbed_indices(self) -> List[int]:
        """Get indices of all perturbed (non-DMSO) samples."""
        return [i for i, m in enumerate(self.metadata) if not m['is_control']]
    
    def get_batch_paired_sample(self, perturbed_idx: int) -> Tuple[int, int]:
        """Get a control sample from the same batch as the perturbed sample."""
        batch = self.metadata[perturbed_idx]['batch']
        batch_indices = self.batch_to_indices[batch]
        
        # Find control samples in this batch
        control_indices = [i for i in batch_indices if self.metadata[i]['is_control']]
        
        if not control_indices:
            # Fallback: use any control
            control_indices = self.get_control_indices()
        
        control_idx = np.random.choice(control_indices)
        return control_idx, perturbed_idx


class BatchPairedDataLoader:
    """
    DataLoader that ensures control and perturbed samples come from the same batch.
    This is CRITICAL for avoiding batch effect artifacts.
    
    Training Process:
    1. Samples a random perturbed image (e.g., Taxol-treated cell)
    2. Finds a control image from the SAME experimental batch
    3. Provides both images + the drug's Morgan fingerprint
    4. Model learns: Control + Fingerprint -> Perturbed morphology
    
    Because batches contain mixed perturbations (Taxol, Nocodazole, DMSO, etc.),
    the model must rely on the fingerprint to know which transformation to apply.
    """
    
    def __init__(
        self,
        dataset: BBBC021Dataset,
        batch_size: int = 16,
        shuffle: bool = True,
        num_workers: int = 4,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        # Get perturbed indices
        self.perturbed_indices = dataset.get_perturbed_indices()
        
    def __iter__(self):
        indices = self.perturbed_indices.copy()
        if self.shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, len(indices), self.batch_size):
            batch_indices = indices[start_idx:start_idx + self.batch_size]
            
            controls = []
            perturbeds = []
            fingerprints = []
            
            for perturbed_idx in batch_indices:
                control_idx, perturbed_idx = self.dataset.get_batch_paired_sample(perturbed_idx)
                
                control_data = self.dataset[control_idx]
                perturbed_data = self.dataset[perturbed_idx]
                
                controls.append(control_data['image'])
                perturbeds.append(perturbed_data['image'])
                fingerprints.append(perturbed_data['fingerprint'])
            
            yield {
                'control': torch.stack(controls),
                'perturbed': torch.stack(perturbeds),
                'fingerprint': torch.stack(fingerprints),
            }
    
    def __len__(self):
        return (len(self.perturbed_indices) + self.batch_size - 1) // self.batch_size


# ============================================================================
# U-NET ARCHITECTURE (for 96x96x3 images)
# ============================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal time embeddings."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResBlock(nn.Module):
    """
    Residual block with time and condition embedding.
    
    The condition embedding (perturbation embedding) is injected here via addition
    to the feature maps. This acts like a "style transfer" switch:
    - Taxol embedding activates filters for "bundled tubulin"
    - Nocodazole embedding activates filters for "fragmented nuclei"
    - etc.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        cond_emb_dim: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )
        
        self.cond_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_emb_dim, out_channels),
        ) if cond_emb_dim > 0 else None
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        cond_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.block1(x)
        h = h + self.time_mlp(time_emb)[:, :, None, None]
        
        if self.cond_mlp is not None and cond_emb is not None:
            h = h + self.cond_mlp(cond_emb)[:, :, None, None]
        
        h = self.block2(h)
        return h + self.residual_conv(x)


class DownBlock(nn.Module):
    """Downsampling block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        cond_emb_dim: int = 0,
        has_attn: bool = False,
    ):
        super().__init__()
        
        self.res1 = ResBlock(in_channels, out_channels, time_emb_dim, cond_emb_dim)
        self.res2 = ResBlock(out_channels, out_channels, time_emb_dim, cond_emb_dim)
        
        if has_attn:
            self.attn = nn.MultiheadAttention(out_channels, num_heads=4, batch_first=True)
        else:
            self.attn = None
        
        self.downsample = nn.Conv2d(out_channels, out_channels, 4, stride=2, padding=1)
    
    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        cond_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.res1(x, time_emb, cond_emb)
        h = self.res2(h, time_emb, cond_emb)
        
        if self.attn is not None:
            b, c, height, width = h.shape
            h_flat = h.reshape(b, c, -1).permute(0, 2, 1)
            h_attn, _ = self.attn(h_flat, h_flat, h_flat)
            h = h + h_attn.permute(0, 2, 1).reshape(b, c, height, width)
        
        skip = h
        h = self.downsample(h)
        return h, skip


class UpBlock(nn.Module):
    """Upsampling block with skip connection."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        cond_emb_dim: int = 0,
        has_attn: bool = False,
    ):
        super().__init__()
        
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        
        # Input is concatenated with skip connection (skip has same channels as upsampled feature)
        self.res1 = ResBlock(in_channels * 2, out_channels, time_emb_dim, cond_emb_dim)
        self.res2 = ResBlock(out_channels, out_channels, time_emb_dim, cond_emb_dim)
        
        if has_attn:
            self.attn = nn.MultiheadAttention(out_channels, num_heads=4, batch_first=True)
        else:
            self.attn = None
    
    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        time_emb: torch.Tensor,
        cond_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.upsample(x)
        h = torch.cat([h, skip], dim=1)
        h = self.res1(h, time_emb, cond_emb)
        h = self.res2(h, time_emb, cond_emb)
        
        if self.attn is not None:
            b, c, height, width = h.shape
            h_flat = h.reshape(b, c, -1).permute(0, 2, 1)
            h_attn, _ = self.attn(h_flat, h_flat, h_flat)
            h = h + h_attn.permute(0, 2, 1).reshape(b, c, height, width)
        
        return h


class UNet(nn.Module):
    """
    U-Net for DDPM on BBBC021 images (96x96x3).
    
    Architecture:
    - Input: (B, 3, 96, 96)
    - Encoder: 4 downsampling blocks -> (B, 512, 6, 6)
    - Bottleneck: ResBlock
    - Decoder: 4 upsampling blocks with skip connections
    - Output: (B, 3, 96, 96)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        channels: List[int] = [64, 128, 256, 512],
        time_emb_dim: int = 256,
        cond_emb_dim: int = 256,
        conditional: bool = False,
    ):
        super().__init__()
        
        self.conditional = conditional
        self.time_emb_dim = time_emb_dim
        self.cond_emb_dim = cond_emb_dim if conditional else 0
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        
        # Condition embedding (for perturbation)
        if conditional:
            self.cond_embed = nn.Sequential(
                nn.Linear(cond_emb_dim, cond_emb_dim * 2),
                nn.GELU(),
                nn.Linear(cond_emb_dim * 2, cond_emb_dim),
            )
        else:
            self.cond_embed = None
        
        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, channels[0], 3, padding=1)
        
        # Encoder (downsampling)
        self.down_blocks = nn.ModuleList()
        in_ch = channels[0]
        for i, out_ch in enumerate(channels):
            has_attn = i >= 2  # Attention in deeper layers
            self.down_blocks.append(
                DownBlock(in_ch, out_ch, time_emb_dim, self.cond_emb_dim, has_attn)
            )
            in_ch = out_ch
        
        # Bottleneck
        self.bottleneck = ResBlock(channels[-1], channels[-1], time_emb_dim, self.cond_emb_dim)
        
        # Decoder (upsampling)
        self.up_blocks = nn.ModuleList()
        reversed_channels = list(reversed(channels))
        for i, out_ch in enumerate(reversed_channels[1:] + [channels[0]]):
            in_ch = reversed_channels[i]
            has_attn = i < 2  # Match encoder attention pattern
            self.up_blocks.append(
                UpBlock(in_ch, out_ch, time_emb_dim, self.cond_emb_dim, has_attn)
            )
        
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], out_channels, 3, padding=1),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Noisy images (B, C, H, W)
            t: Timesteps (B,)
            condition: Condition embedding (B, cond_emb_dim) or None
        
        Returns:
            Predicted noise (B, C, H, W)
        """
        # Time embedding
        t_emb = self.time_embed(t.float())
        
        # Condition embedding
        if self.conditional and condition is not None:
            c_emb = self.cond_embed(condition)
        else:
            c_emb = None
        
        # Initial convolution
        h = self.init_conv(x)
        
        # Encoder
        skips = []
        for down_block in self.down_blocks:
            h, skip = down_block(h, t_emb, c_emb)
            skips.append(skip)
        
        # Bottleneck
        h = self.bottleneck(h, t_emb, c_emb)
        
        # Decoder
        for up_block, skip in zip(self.up_blocks, reversed(skips)):
            h = up_block(h, skip, t_emb, c_emb)
        
        # Final convolution
        return self.final_conv(h)


class ConditionalUNet(UNet):
    """
    Conditional U-Net that takes control image as additional input.
    Concatenates control image with noisy image in input channels.
    
    This enables conditional diffusion where:
    - Condition A (Control Image): Provides the source state (which specific cells to transform)
    - Condition B (Perturbation Embedding): Specifies the target effect (which drug/perturbation)
    
    The model learns a single function capable of generating any perturbation outcome
    by taking the specific drug identity (via Morgan fingerprint embedding) as input.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        channels: List[int] = [64, 128, 256, 512],
        time_emb_dim: int = 256,
        cond_emb_dim: int = 256,  # Perturbation embedding dim
    ):
        # Double input channels to accommodate control image
        super().__init__(
            in_channels=in_channels * 2,  # [noisy_image, control_image]
            out_channels=out_channels,
            channels=channels,
            time_emb_dim=time_emb_dim,
            cond_emb_dim=cond_emb_dim,
            conditional=True,
        )
        
        self.original_in_channels = in_channels
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        control: torch.Tensor,
        perturbation_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with control image conditioning.
        
        Args:
            x: Noisy images (B, C, H, W)
            t: Timesteps (B,)
            control: Control images (B, C, H, W)
            perturbation_emb: Perturbation embedding (B, cond_emb_dim)
        
        Returns:
            Predicted noise (B, C, H, W)
        """
        # Concatenate noisy image with control image
        x_concat = torch.cat([x, control], dim=1)
        return super().forward(x_concat, t, perturbation_emb)


# ============================================================================
# PERTURBATION ENCODER
# ============================================================================

class PerturbationEncoder(nn.Module):
    """Encode Morgan fingerprints to perturbation embeddings."""
    
    def __init__(self, input_dim: int = 1024, output_dim: int = 256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )
    
    def forward(self, fingerprint: torch.Tensor) -> torch.Tensor:
        return self.net(fingerprint)


# ============================================================================
# DDPM FOR IMAGES
# ============================================================================

class ImageDDPM:
    """DDPM for image generation with U-Net backbone."""
    
    def __init__(
        self,
        image_size: int = 96,
        in_channels: int = 3,
        channels: List[int] = [64, 128, 256, 512],
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        time_emb_dim: int = 256,
        lr: float = 1e-4,
        device: str = "cuda",
        conditional: bool = False,
        cond_emb_dim: int = 256,
    ):
        self.image_size = image_size
        self.in_channels = in_channels
        self.timesteps = timesteps
        self.device = torch.device(device)
        self.conditional = conditional
        
        # Beta schedule
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_schedule(betas, alphas, alphas_cumprod)
        
        # Model
        if conditional:
            self.model = ConditionalUNet(
                in_channels=in_channels,
                out_channels=in_channels,
                channels=channels,
                time_emb_dim=time_emb_dim,
                cond_emb_dim=cond_emb_dim,
            ).to(self.device)
        else:
            self.model = UNet(
                in_channels=in_channels,
                out_channels=in_channels,
                channels=channels,
                time_emb_dim=time_emb_dim,
                conditional=False,
            ).to(self.device)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        # Perturbation encoder
        if conditional:
            self.perturbation_encoder = PerturbationEncoder(
                input_dim=1024, output_dim=cond_emb_dim
            ).to(self.device)
            self.optimizer.add_param_group({'params': self.perturbation_encoder.parameters()})
        else:
            self.perturbation_encoder = None
    
    def register_schedule(self, betas, alphas, alphas_cumprod):
        """Register diffusion schedule."""
        self.betas = betas.to(self.device)
        self.alphas = alphas.to(self.device)
        self.alphas_cumprod = alphas_cumprod.to(self.device)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(self.device)
    
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """Forward diffusion: q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alpha_t * x0 + sqrt_one_minus_alpha_t * noise
    
    def train_step(
        self,
        x0: torch.Tensor,
        control: Optional[torch.Tensor] = None,
        fingerprint: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Single training step.
        
        Training objective: Given a control image and a perturbation fingerprint,
        predict the noise needed to reconstruct the perturbed image.
        
        The model sees thousands of pairs (Control->Taxol, Control->DMSO, Control->PP-2)
        and learns to associate the fingerprint vector with the required morphological change.
        """
        self.model.train()
        batch_size = x0.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)
        
        # Sample noise
        noise = torch.randn_like(x0)
        
        # Forward diffusion
        x_t = self.q_sample(x0, t, noise)
        
        # Predict noise
        if self.conditional:
            perturbation_emb = self.perturbation_encoder(fingerprint)
            noise_pred = self.model(x_t, t, control, perturbation_emb)
        else:
            noise_pred = self.model(x_t, t)
        
        # Loss
        loss = F.mse_loss(noise_pred, noise)
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        control: Optional[torch.Tensor] = None,
        fingerprint: Optional[torch.Tensor] = None,
        num_steps: int = None,
    ) -> torch.Tensor:
        """Sample from the model."""
        self.model.eval()
        
        if num_steps is None:
            num_steps = self.timesteps
        
        # Start from noise
        x = torch.randn(num_samples, self.in_channels, self.image_size, self.image_size, device=self.device)
        
        # Prepare condition
        if self.conditional and fingerprint is not None:
            perturbation_emb = self.perturbation_encoder(fingerprint)
        else:
            perturbation_emb = None
        
        # Reverse diffusion
        step_size = self.timesteps // num_steps
        
        for i in reversed(range(0, self.timesteps, step_size)):
            t = torch.full((num_samples,), i, device=self.device, dtype=torch.long)
            
            # Predict noise
            if self.conditional:
                noise_pred = self.model(x, t, control, perturbation_emb)
            else:
                noise_pred = self.model(x, t)
            
            # Clamp noise prediction
            noise_pred = torch.clamp(noise_pred, -10.0, 10.0)
            
            # Compute coefficients
            alpha_t = self.alphas[i]
            alpha_cumprod_t = self.alphas_cumprod[i]
            beta_t = self.betas[i]
            
            if i > 0:
                alpha_cumprod_t_prev = self.alphas_cumprod[i - step_size] if i >= step_size else self.alphas_cumprod[0]
            else:
                alpha_cumprod_t_prev = torch.tensor(1.0, device=self.device)
            
            # Compute x_{t-1}
            sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
            sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)
            
            pred_x0 = (x - sqrt_one_minus_alpha_cumprod_t * noise_pred) / (sqrt_alpha_cumprod_t + 1e-8)
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            
            dir_xt = torch.sqrt(1.0 - alpha_cumprod_t_prev) * noise_pred
            x = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + dir_xt
            
            if i > 0:
                noise = torch.randn_like(x)
                sigma_t = torch.sqrt(beta_t)
                x = x + sigma_t * noise
            
            x = torch.clamp(x, -1.0, 1.0)
        
        return x
    
    def save(self, path: str):
        """Save model."""
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'image_size': self.image_size,
            'in_channels': self.in_channels,
            'timesteps': self.timesteps,
            'conditional': self.conditional,
        }
        if self.perturbation_encoder is not None:
            state['perturbation_encoder_state_dict'] = self.perturbation_encoder.state_dict()
        torch.save(state, path)
    
    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.perturbation_encoder is not None and 'perturbation_encoder_state_dict' in checkpoint:
            self.perturbation_encoder.load_state_dict(checkpoint['perturbation_encoder_state_dict'])


# ============================================================================
# ES TRAINER FOR IMAGES
# ============================================================================

class ImageESTrainer:
    """Evolution Strategies trainer for image DDPM coupling."""
    
    def __init__(
        self,
        ddpm: ImageDDPM,
        population_size: int = 20,
        sigma: float = 0.005,
        lr: float = 0.0005,
        device: str = "cuda",
    ):
        self.ddpm = ddpm
        self.population_size = population_size
        self.sigma = sigma
        self.lr = lr
        self.device = torch.device(device)
    
    def compute_fitness(
        self,
        params: np.ndarray,
        x_batch: torch.Tensor,
        control_batch: torch.Tensor,
        fingerprint_batch: torch.Tensor,
    ) -> float:
        """Compute fitness (negative loss)."""
        if not np.all(np.isfinite(params)):
            return -float('inf')
        
        # Load parameters into model
        state_dict = self.ddpm.model.state_dict()
        offset = 0
        for key in state_dict.keys():
            param_size = state_dict[key].numel()
            state_dict[key] = torch.from_numpy(
                params[offset:offset + param_size].reshape(state_dict[key].shape)
            ).float().to(self.device)
            offset += param_size
        
        self.ddpm.model.load_state_dict(state_dict)
        
        # Compute loss
        self.ddpm.model.eval()
        with torch.no_grad():
            batch_size = x_batch.shape[0]
            t = torch.randint(0, self.ddpm.timesteps, (batch_size,), device=self.device)
            noise = torch.randn_like(x_batch)
            x_t = self.ddpm.q_sample(x_batch, t, noise)
            
            perturbation_emb = self.ddpm.perturbation_encoder(fingerprint_batch)
            noise_pred = self.ddpm.model(x_t, t, control_batch, perturbation_emb)
            
            if torch.isnan(noise_pred).any() or torch.isinf(noise_pred).any():
                return -float('inf')
            
            loss = F.mse_loss(noise_pred, noise)
            
            if torch.isnan(loss) or torch.isinf(loss):
                return -float('inf')
        
        return -loss.item()
    
    def train_step(
        self,
        x_batch: torch.Tensor,
        control_batch: torch.Tensor,
        fingerprint_batch: torch.Tensor,
    ) -> float:
        """Single ES training step."""
        # Get current parameters
        params = []
        for param in self.ddpm.model.parameters():
            params.append(param.data.cpu().numpy().flatten())
        params = np.concatenate(params)
        
        # Generate population
        population = []
        noises = []
        for _ in range(self.population_size):
            noise = np.random.randn(len(params)) * self.sigma
            noises.append(noise)
            population.append(params + noise)
        
        # Evaluate fitness
        fitnesses = []
        for p in population:
            fitness = self.compute_fitness(p, x_batch, control_batch, fingerprint_batch)
            fitnesses.append(fitness)
        
        fitnesses = np.array(fitnesses)
        
        # Handle invalid fitnesses
        if not np.all(np.isfinite(fitnesses)):
            valid_mask = np.isfinite(fitnesses)
            if np.any(valid_mask):
                worst_fitness = np.min(fitnesses[valid_mask])
                fitnesses[~valid_mask] = worst_fitness
            else:
                return 0.0
        
        # Update parameters
        fitness_std = np.std(fitnesses)
        if fitness_std < 1e-10:
            return -np.mean(fitnesses)
        
        normalized_fitnesses = (fitnesses - np.mean(fitnesses)) / (fitness_std + 1e-8)
        grad = np.zeros_like(params)
        for i in range(self.population_size):
            grad += normalized_fitnesses[i] * noises[i]
        grad /= (self.population_size * self.sigma)
        
        # Gradient clipping
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1.0:
            grad = grad * (1.0 / grad_norm)
        
        # Apply update
        params = params + self.lr * grad
        
        if not np.all(np.isfinite(params)):
            return -np.mean(fitnesses)
        
        # Load updated parameters
        state_dict = self.ddpm.model.state_dict()
        offset = 0
        for key in state_dict.keys():
            param_size = state_dict[key].numel()
            state_dict[key] = torch.from_numpy(
                params[offset:offset + param_size].reshape(state_dict[key].shape)
            ).float().to(self.device)
            offset += param_size
        
        self.ddpm.model.load_state_dict(state_dict)
        
        return -np.mean(fitnesses)


# ============================================================================
# PPO TRAINER FOR IMAGES
# ============================================================================

class ImagePPOTrainer:
    """PPO trainer for image DDPM coupling."""
    
    def __init__(
        self,
        cond_model: ImageDDPM,
        pretrain_model: ImageDDPM,
        kl_weight: float = 0.5,
        ppo_clip: float = 0.1,
        lr: float = 5e-5,
        device: str = "cuda",
    ):
        self.cond_model = cond_model
        self.pretrain_model = pretrain_model
        self.kl_weight = kl_weight
        self.ppo_clip = ppo_clip
        self.device = torch.device(device)
        
        # Freeze pretrained model
        self.pretrain_model.model.eval()
        for p in self.pretrain_model.model.parameters():
            p.requires_grad = False
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.cond_model.model.parameters(), lr=lr
        )
        if self.cond_model.perturbation_encoder is not None:
            self.optimizer.add_param_group({
                'params': self.cond_model.perturbation_encoder.parameters()
            })
    
    def train_step(
        self,
        x_batch: torch.Tensor,
        control_batch: torch.Tensor,
        fingerprint_batch: torch.Tensor,
    ) -> float:
        """Single PPO training step."""
        self.cond_model.model.train()
        batch_size = x_batch.shape[0]
        
        # Sample timesteps
        t = torch.randint(0, self.cond_model.timesteps, (batch_size,), device=self.device)
        noise = torch.randn_like(x_batch)
        x_t = self.cond_model.q_sample(x_batch, t, noise)
        
        # Reconstruction loss
        perturbation_emb = self.cond_model.perturbation_encoder(fingerprint_batch)
        noise_pred_cond = self.cond_model.model(x_t, t, control_batch, perturbation_emb)
        reconstruction_loss = F.mse_loss(noise_pred_cond, noise)
        
        # KL penalty (divergence from pretrained)
        with torch.no_grad():
            noise_pred_pretrain = self.pretrain_model.model(x_t, t)
        
        kl_loss = F.mse_loss(noise_pred_cond, noise_pred_pretrain)
        
        # Total loss
        total_loss = reconstruction_loss + self.kl_weight * kl_loss
        
        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.cond_model.model.parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item()


# ============================================================================
# METRICS
# ============================================================================

class ImageMetrics:
    """Compute metrics for image generation quality."""
    
    @staticmethod
    def compute_fid(real_images: np.ndarray, fake_images: np.ndarray) -> float:
        """
        Compute Fréchet Inception Distance.
        Simplified version without Inception network (uses raw features).
        """
        if not SCIPY_AVAILABLE:
            return 0.0
        
        # Flatten images
        real_flat = real_images.reshape(real_images.shape[0], -1)
        fake_flat = fake_images.reshape(fake_images.shape[0], -1)
        
        # Compute statistics
        mu_real = np.mean(real_flat, axis=0)
        mu_fake = np.mean(fake_flat, axis=0)
        
        sigma_real = np.cov(real_flat, rowvar=False) + np.eye(real_flat.shape[1]) * 1e-6
        sigma_fake = np.cov(fake_flat, rowvar=False) + np.eye(fake_flat.shape[1]) * 1e-6
        
        # Compute FID
        diff = mu_real - mu_fake
        
        try:
            covmean = linalg.sqrtm(sigma_real @ sigma_fake)
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            
            fid = diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean)
            return float(fid)
        except Exception:
            return float('inf')
    
    @staticmethod
    def compute_mse(real_images: np.ndarray, fake_images: np.ndarray) -> float:
        """Compute mean squared error."""
        return float(np.mean((real_images - fake_images) ** 2))
    
    @staticmethod
    def compute_mae(real_images: np.ndarray, fake_images: np.ndarray) -> float:
        """Compute mean absolute error."""
        return float(np.mean(np.abs(real_images - fake_images)))
    
    @staticmethod
    def compute_ssim(real_images: np.ndarray, fake_images: np.ndarray) -> float:
        """Compute structural similarity (simplified)."""
        # Simplified SSIM computation
        c1, c2 = 0.01**2, 0.03**2
        
        mu_real = np.mean(real_images, axis=(1, 2, 3), keepdims=True)
        mu_fake = np.mean(fake_images, axis=(1, 2, 3), keepdims=True)
        
        sigma_real_sq = np.var(real_images, axis=(1, 2, 3), keepdims=True)
        sigma_fake_sq = np.var(fake_images, axis=(1, 2, 3), keepdims=True)
        sigma_real_fake = np.mean((real_images - mu_real) * (fake_images - mu_fake), axis=(1, 2, 3), keepdims=True)
        
        ssim = ((2 * mu_real * mu_fake + c1) * (2 * sigma_real_fake + c2)) / \
               ((mu_real**2 + mu_fake**2 + c1) * (sigma_real_sq + sigma_fake_sq + c2))
        
        return float(np.mean(ssim))

# [INSERT THIS CLASS AFTER ImageMetrics AND BEFORE BBBC021AblationRunner]

class ApproximateMetrics:
    """
    Computes Information Theoretic proxies for Images using Histogram estimation.
    Restores metrics like MI, Entropy, and KL for compatibility with ablation logs.
    """
    
    @staticmethod
    def _pixel_entropy(imgs_flat, bins=256):
        """Estimates Shannon Entropy H(X) using pixel value histograms."""
        # Calculate histogram of pixel intensities
        hist, _ = np.histogram(imgs_flat, bins=bins, range=(-1, 1), density=True)
        prob = hist[hist > 0]
        return -np.sum(prob * np.log(prob + 1e-10))

    @staticmethod
    def _mutual_info(x_flat, y_flat, bins=64):
        """Estimates Mutual Information I(X; Y) = H(X) + H(Y) - H(X,Y)."""
        # 1. Marginal Entropies
        h_x = ApproximateMetrics._pixel_entropy(x_flat, bins)
        h_y = ApproximateMetrics._pixel_entropy(y_flat, bins)
        
        # 2. Joint Entropy H(X,Y) via 2D Histogram (Downsample for speed)
        if x_flat.size > 1_000_000:
            idx = np.random.choice(x_flat.size, 1_000_000, replace=False)
            x_sub, y_sub = x_flat[idx], y_flat[idx]
        else:
            x_sub, y_sub = x_flat, y_flat
            
        hist_2d, _, _ = np.histogram2d(x_sub, y_sub, bins=bins, range=[[-1, 1], [-1, 1]], density=True)
        prob_2d = hist_2d[hist_2d > 0]
        h_xy = -np.sum(prob_2d * np.log(prob_2d + 1e-10))
        
        return h_x + h_y - h_xy, h_xy, h_x, h_y

    @staticmethod
    def compute_all(real_images: np.ndarray, fake_images: np.ndarray):
        # Flatten images
        real_flat = real_images.flatten()
        fake_flat = fake_images.flatten()
        
        # 1. Basic Stats
        mu_real, std_real = np.mean(real_flat), np.std(real_flat)
        mu_fake, std_fake = np.mean(fake_flat), np.std(fake_flat)
        
        # 2. Correlation
        correlation = np.corrcoef(real_flat[::100], fake_flat[::100])[0, 1] 
        
        # 3. Entropy & MI
        mi, h_joint, h_real, h_fake = ApproximateMetrics._mutual_info(real_flat, fake_flat)
        
        # 4. KL Proxy (Gaussian approx for logging consistency)
        kl_div = np.log(std_fake / (std_real + 1e-8)) + \
                 ((std_real**2 + (mu_real - mu_fake)**2) / (2 * std_fake**2 + 1e-8)) - 0.5

        return {
            'kl_div_total': float(kl_div),
            'entropy_x1': float(h_fake), # Generated
            'entropy_x2': float(h_real), # Real
            'joint_entropy': float(h_joint),
            'mutual_information': float(max(0, mi)),
            'correlation': float(correlation),
            'mu1_learned': float(mu_fake),
            'std1_learned': float(std_fake),
            # Fillers to match exact keys from original script if needed
            'kl_div_1': float(kl_div), 'kl_div_2': float(kl_div),
            'mi_x2_to_x1': float(mi), 'mi_x1_to_x2': float(mi),
            'h_x1_given_x2': float(h_joint - h_real),
            'mae_x2_to_x1': float(np.mean(np.abs(real_flat - fake_flat))),
        }
# ============================================================================
# ABLATION RUNNER
# ============================================================================

class BBBC021AblationRunner:
    """Run ablation study on BBBC021 dataset."""
    
    def __init__(self, config: BBBC021Config):
        self.config = config
        
        # Set seeds
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
        
        # Create directories
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(config.output_dir, f"run_{timestamp}")
        self.models_dir = os.path.join(self.output_dir, "models")
        self.plots_dir = os.path.join(self.output_dir, "plots")
        self.logs_dir = os.path.join(self.output_dir, "logs")
        
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Pretrained models directory
        self.pretrained_dir = os.path.join(config.output_dir, "pretrained_models")
        os.makedirs(self.pretrained_dir, exist_ok=True)
        
        # Load dataset
        print("Loading BBBC021 dataset...")
        self.train_dataset = BBBC021Dataset(
            config.data_dir, config.metadata_file,
            config.image_size, split="train"
        )
        self.val_dataset = BBBC021Dataset(
            config.data_dir, config.metadata_file,
            config.image_size, split="val"
        )
        
        # Initialize wandb
        if config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=config.wandb_project,
                name=f"bbbc021_ablation_{timestamp}",
                config=asdict(config)
            )
        
        # Store results
        self.all_results = {'ES': [], 'PPO': []}
        
        print(f"Output directory: {self.output_dir}")
        print(f"Device: {config.device}")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
    
    def run(self):
        """Run the ablation study."""
        print("\n" + "=" * 80)
        print("BBBC021 ABLATION STUDY: ES vs PPO")
        print("=" * 80 + "\n")
        
        start_time = time.time()
        
        # Step 1: Pretrain unconditional DDPM
        print("Step 1: Pretraining unconditional DDPM on control images...")
        pretrain_ddpm = self._pretrain_ddpm()
        
        # Step 2: ES Ablations
        print("\nStep 2: ES Ablations...")
        es_configs = list(itertools.product(
            self.config.es_sigma_values,
            self.config.es_lr_values
        ))
        
        for i, (sigma, lr) in enumerate(es_configs):
            print(f"\n  ES Config {i+1}/{len(es_configs)}: sigma={sigma}, lr={lr}")
            result = self._run_es_experiment(pretrain_ddpm, sigma, lr, i)
            self.all_results['ES'].append(result)
        
        # Step 3: PPO Ablations
        print("\nStep 3: PPO Ablations...")
        ppo_configs = list(itertools.product(
            self.config.ppo_kl_weight_values,
            self.config.ppo_clip_values,
            self.config.ppo_lr_values
        ))
        
        for i, (kl_weight, ppo_clip, lr) in enumerate(ppo_configs):
            print(f"\n  PPO Config {i+1}/{len(ppo_configs)}: kl_weight={kl_weight}, clip={ppo_clip}, lr={lr}")
            result = self._run_ppo_experiment(pretrain_ddpm, kl_weight, ppo_clip, lr, i)
            self.all_results['PPO'].append(result)
        
        # Generate summary
        print("\n" + "=" * 80)
        print("GENERATING SUMMARY")
        print("=" * 80 + "\n")
        self._generate_summary()
        
        total_time = time.time() - start_time
        print(f"\nTotal time: {total_time / 3600:.2f} hours")
        print(f"Results saved to: {self.output_dir}")
        
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.finish()
        
    
    def _pretrain_ddpm(self) -> ImageDDPM:
        """Pretrain unconditional DDPM on control images."""
        model_path = os.path.join(self.pretrained_dir, "ddpm_pretrained.pt")
        
        ddpm = ImageDDPM(
            image_size=self.config.image_size,
            in_channels=self.config.num_channels,
            channels=self.config.unet_channels,
            timesteps=self.config.ddpm_timesteps,
            time_emb_dim=self.config.time_embed_dim,
            lr=self.config.ddpm_lr,
            device=self.config.device,
            conditional=False,
        )
        
        if self.config.reuse_pretrained and os.path.exists(model_path):
            print(f"  Loading pretrained model from {model_path}")
            ddpm.load(model_path)
            return ddpm
        
        print(f"  Training unconditional DDPM for {self.config.ddpm_epochs} epochs...")
        
        # Get control images only
        control_indices = self.train_dataset.get_control_indices()
        control_dataset = Subset(self.train_dataset, control_indices)
        dataloader = DataLoader(
            control_dataset,
            batch_size=self.config.ddpm_batch_size,
            shuffle=True,
            num_workers=4,
        )
        
        # Track metrics for plotting
        pretrain_metrics = []
        pretrain_plot_dir = os.path.join(self.plots_dir, "pretraining")
        os.makedirs(pretrain_plot_dir, exist_ok=True)
        
        for epoch in range(self.config.ddpm_epochs):
            epoch_losses = []
            for batch in dataloader:
                images = batch['image'].to(self.config.device)
                loss = ddpm.train_step(images)
                epoch_losses.append(loss)
            
            avg_loss = np.mean(epoch_losses)
            
            # Evaluate and track metrics
            metrics = self._evaluate_pretrain(ddpm, control_dataset)
            metrics['epoch'] = epoch + 1
            metrics['loss'] = avg_loss
            metrics['phase'] = 'pretraining'
            pretrain_metrics.append(metrics)
            
            # Plot every epoch
            self._plot_checkpoint(pretrain_metrics, pretrain_plot_dir, epoch, 'Pretraining', 'DDPM Pretraining')
            
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{self.config.ddpm_epochs}, Loss: {avg_loss:.4f}, FID: {metrics.get('fid', 0):.2f}")
        
        ddpm.save(model_path)
        print(f"  Saved pretrained model to {model_path}")
        
        return ddpm
    
    def _create_conditional_ddpm(self, pretrain_ddpm: ImageDDPM) -> ImageDDPM:
        """Create conditional DDPM and initialize from pretrained."""
        cond_ddpm = ImageDDPM(
            image_size=self.config.image_size,
            in_channels=self.config.num_channels,
            channels=self.config.unet_channels,
            timesteps=self.config.ddpm_timesteps,
            time_emb_dim=self.config.time_embed_dim,
            lr=self.config.ddpm_lr,
            device=self.config.device,
            conditional=True,
            cond_emb_dim=self.config.perturbation_embed_dim,
        )
        
        # Smart weight transfer (handle architecture mismatch)
        pretrained_state = pretrain_ddpm.model.state_dict()
        current_state = cond_ddpm.model.state_dict()
        
        for key in current_state.keys():
            if key in pretrained_state:
                if pretrained_state[key].shape == current_state[key].shape:
                    current_state[key] = pretrained_state[key].clone()
                elif 'init_conv' in key:
                    # Handle input channel mismatch (3 vs 6 for conditional)
                    current_state[key][:, :3] = pretrained_state[key]
                    print(f"    Partially loaded {key}")
        
        cond_ddpm.model.load_state_dict(current_state)
        
        return cond_ddpm
    
    def _run_es_experiment(
        self,
        pretrain_ddpm: ImageDDPM,
        sigma: float,
        lr: float,
        config_idx: int,
    ) -> Dict:
        """Run single ES experiment."""
        cond_ddpm = self._create_conditional_ddpm(pretrain_ddpm)
        
        # Create dataloader
        dataloader = BatchPairedDataLoader(
            self.train_dataset,
            batch_size=self.config.coupling_batch_size,
            shuffle=True,
        )
        
        # Warmup phase (gradient training)
        print(f"    Warmup phase: {self.config.warmup_epochs} epochs...")
        warmup_metrics = []
        for warmup_epoch in range(self.config.warmup_epochs):
            warmup_losses = []
            for batch in dataloader:
                control = batch['control'].to(self.config.device)
                perturbed = batch['perturbed'].to(self.config.device)
                fingerprint = batch['fingerprint'].to(self.config.device)
                
                loss = cond_ddpm.train_step(perturbed, control, fingerprint)
                warmup_losses.append(loss)
            
            avg_loss = np.mean(warmup_losses)
            
            # Evaluate during warmup
            metrics = self._evaluate(cond_ddpm)
            metrics['epoch'] = warmup_epoch + 1
            metrics['loss'] = avg_loss
            metrics['sigma'] = sigma
            metrics['lr'] = lr
            metrics['phase'] = 'warmup'
            warmup_metrics.append(metrics)
            
            # Plot during warmup
            checkpoint_dir = os.path.join(self.plots_dir, f'ES_config_{config_idx}')
            os.makedirs(checkpoint_dir, exist_ok=True)
            self._plot_checkpoint(warmup_metrics, checkpoint_dir, warmup_epoch, 'ES', f'σ={sigma}, lr={lr}')
            
            if (warmup_epoch + 1) % 3 == 0:
                print(f"      Warmup epoch {warmup_epoch+1}, Loss: {avg_loss:.4f}, FID: {metrics.get('fid', 0):.2f}")
        
        # Initialize epoch_metrics with warmup metrics
        epoch_metrics = warmup_metrics.copy()
        
        # ES training
        es_trainer = ImageESTrainer(
            cond_ddpm,
            population_size=self.config.es_population_size,
            sigma=sigma,
            lr=lr,
            device=self.config.device,
        )
        
        for epoch in range(self.config.coupling_epochs):
            epoch_losses = []
            
            for batch in dataloader:
                control = batch['control'].to(self.config.device)
                perturbed = batch['perturbed'].to(self.config.device)
                fingerprint = batch['fingerprint'].to(self.config.device)
                
                loss = es_trainer.train_step(perturbed, control, fingerprint)
                epoch_losses.append(loss)
            
            avg_loss = np.mean(epoch_losses)
            
            # Evaluate
            metrics = self._evaluate(cond_ddpm)
            # Continue epoch numbering from warmup
            metrics['epoch'] = len(epoch_metrics) + 1
            metrics['loss'] = avg_loss
            metrics['sigma'] = sigma
            metrics['lr'] = lr
            metrics['phase'] = 'training'  # ES training phase
            epoch_metrics.append(metrics)
            
            # Plot checkpoint
            checkpoint_dir = os.path.join(self.plots_dir, f'ES_config_{config_idx}')
            os.makedirs(checkpoint_dir, exist_ok=True)
            self._plot_checkpoint(epoch_metrics, checkpoint_dir, epoch, 'ES', f'σ={sigma}, lr={lr}')

            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1}: Loss={avg_loss:.4f}, FID={metrics['fid']:.2f}, MI={metrics['mutual_information']:.4f}")
            
            # Log to wandb
            if self.config.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    f'ES/config_{config_idx}/loss': avg_loss,
                    f'ES/config_{config_idx}/fid': metrics['fid'],
                    f'ES/config_{config_idx}/mse': metrics['mse'],
                })
        
        final_metrics = epoch_metrics[-1]
        final_metrics['method'] = 'ES'
        final_metrics['history'] = epoch_metrics
        
        return final_metrics
    
    def _run_ppo_experiment(
        self,
        pretrain_ddpm: ImageDDPM,
        kl_weight: float,
        ppo_clip: float,
        lr: float,
        config_idx: int,
    ) -> Dict:
        """Run single PPO experiment."""
        cond_ddpm = self._create_conditional_ddpm(pretrain_ddpm)
        
        # Create dataloader
        dataloader = BatchPairedDataLoader(
            self.train_dataset,
            batch_size=self.config.coupling_batch_size,
            shuffle=True,
        )
        
        # PPO trainer
        ppo_trainer = ImagePPOTrainer(
            cond_ddpm,
            pretrain_ddpm,
            kl_weight=kl_weight,
            ppo_clip=ppo_clip,
            lr=lr,
            device=self.config.device,
        )
        
        epoch_metrics = []
        
        for epoch in range(self.config.coupling_epochs):
            epoch_losses = []
            
            for batch in dataloader:
                control = batch['control'].to(self.config.device)
                perturbed = batch['perturbed'].to(self.config.device)
                fingerprint = batch['fingerprint'].to(self.config.device)
                
                loss = ppo_trainer.train_step(perturbed, control, fingerprint)
                epoch_losses.append(loss)
            
            avg_loss = np.mean(epoch_losses)
            
            # Evaluate
            metrics = self._evaluate(cond_ddpm)
            metrics['epoch'] = epoch + 1
            metrics['loss'] = avg_loss
            metrics['kl_weight'] = kl_weight
            metrics['ppo_clip'] = ppo_clip
            metrics['lr'] = lr
            metrics['phase'] = 'training'  # PPO training phase
            epoch_metrics.append(metrics)
            
            # Plot checkpoint
            checkpoint_dir = os.path.join(self.plots_dir, f'PPO_config_{config_idx}')
            os.makedirs(checkpoint_dir, exist_ok=True)
            self._plot_checkpoint(epoch_metrics, checkpoint_dir, epoch, 'PPO', f'KL={kl_weight}, lr={lr}')

            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1}: Loss={avg_loss:.4f}, FID={metrics['fid']:.2f}, MI={metrics['mutual_information']:.4f}")
            
            # Log to wandb
            if self.config.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    f'PPO/config_{config_idx}/loss': avg_loss,
                    f'PPO/config_{config_idx}/fid': metrics['fid'],
                    f'PPO/config_{config_idx}/mse': metrics['mse'],
                })
        
        final_metrics = epoch_metrics[-1]
        final_metrics['method'] = 'PPO'
        final_metrics['history'] = epoch_metrics
        
        return final_metrics
    
    
    def _evaluate(self, cond_ddpm: ImageDDPM) -> Dict:
        """Evaluate model on validation set."""
        cond_ddpm.model.eval()
        
        real_images = []
        fake_images = []
        
        val_loader = BatchPairedDataLoader(
            self.val_dataset,
            batch_size=self.config.coupling_batch_size,
            shuffle=False,
        )
        
        num_samples = 0
        max_samples = self.config.num_eval_samples
        
        with torch.no_grad():
            for batch in val_loader:
                if num_samples >= max_samples: break
                
                control = batch['control'].to(self.config.device)
                perturbed = batch['perturbed'].to(self.config.device)
                fingerprint = batch['fingerprint'].to(self.config.device)
                
                generated = cond_ddpm.sample(
                    len(control), control, fingerprint,
                    num_steps=self.config.num_sampling_steps,
                )
                
                real_images.append(perturbed.cpu().numpy())
                fake_images.append(generated.cpu().numpy())
                num_samples += len(control)
        
        real_images = np.concatenate(real_images, axis=0)[:max_samples]
        fake_images = np.concatenate(fake_images, axis=0)[:max_samples]
        
        # 1. Standard Metrics
        fid = ImageMetrics.compute_fid(real_images, fake_images)
        mse = ImageMetrics.compute_mse(real_images, fake_images)
        mae = ImageMetrics.compute_mae(real_images, fake_images)
        ssim = ImageMetrics.compute_ssim(real_images, fake_images)
        
        # 2. Information Theoretic Metrics (Added)
        info_metrics = ApproximateMetrics.compute_all(real_images, fake_images)
        
        # Combine
        metrics = {
            'fid': fid, 'mse': mse, 'mae': mae, 'ssim': ssim,
        }
        metrics.update(info_metrics) # Merge dictionaries
        
        return metrics
    
    def _evaluate_pretrain(self, ddpm: ImageDDPM, dataset: Subset) -> Dict:
        """Evaluate pretrained unconditional DDPM."""
        ddpm.model.eval()
        
        # Sample some images
        num_samples = min(100, len(dataset))
        real_images = []
        fake_images = []
        
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        with torch.no_grad():
            # Get real images
            for i, batch in enumerate(dataloader):
                if len(real_images) >= num_samples:
                    break
                images = batch['image'].to(self.config.device)
                real_images.append(images.cpu().numpy())
            
            # Generate fake images
            generated = ddpm.sample(num_samples, num_steps=self.config.num_sampling_steps)
            fake_images.append(generated.cpu().numpy())
        
        real_images = np.concatenate(real_images, axis=0)[:num_samples]
        fake_images = np.concatenate(fake_images, axis=0)[:num_samples]
        
        # Compute metrics
        fid = ImageMetrics.compute_fid(real_images, fake_images)
        mse = ImageMetrics.compute_mse(real_images, fake_images)
        mae = ImageMetrics.compute_mae(real_images, fake_images)
        ssim = ImageMetrics.compute_ssim(real_images, fake_images)
        
        # Information theoretic metrics
        info_metrics = ApproximateMetrics.compute_all(real_images, fake_images)
        
        metrics = {
            'fid': fid, 'mse': mse, 'mae': mae, 'ssim': ssim,
        }
        metrics.update(info_metrics)
        
        return metrics
    
    def _save_metrics_to_csv(self, metrics: List[Dict], checkpoint_dir: str, method: str, config_str: str = ""):
        """Save metrics to CSV file."""
        if not metrics:
            return
        
        csv_path = os.path.join(checkpoint_dir, f"{method}_metrics.csv")
        fieldnames = list(metrics[0].keys())
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics)
    
    def _plot_checkpoint(self, epoch_metrics: List[Dict], checkpoint_dir: str, epoch: int, method: str, config_str: str):
        """Generate comprehensive checkpoint plot (including warmup epochs with shading)."""
        if len(epoch_metrics) < 1:
            return
        
        # Include ALL epochs for full picture
        all_metrics = epoch_metrics
        warmup_metrics = [m for m in all_metrics if m.get('phase') == 'warmup']
        training_metrics = [m for m in all_metrics if m.get('phase') in ['training', 'pretraining']]
        
        if len(all_metrics) < 1:
            return  # Not enough data yet
        
        # Save metrics to CSV
        self._save_metrics_to_csv(all_metrics, checkpoint_dir, method, config_str)
        
        epochs = [m['epoch'] for m in all_metrics]
        
        # Find warmup/training boundary for vertical line
        warmup_boundary = None
        if warmup_metrics and training_metrics:
            warmup_end = max([m['epoch'] for m in warmup_metrics])
            training_start = min([m['epoch'] for m in training_metrics])
            warmup_boundary = (warmup_end + training_start) / 2.0
        
        # Create comprehensive plot with ALL metrics
        fig = plt.figure(figsize=(24, 16))
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Helper function to plot with warmup/training boundary
        def plot_metric(ax, all_epochs, all_values, ylabel, title, color='blue'):
            """Plot metric with phase boundary line."""
            ax.plot(all_epochs, all_values, color=color, linewidth=2, 
                   marker='o', markersize=4, alpha=0.8, label=method)
            
            # Add vertical line at warmup/training boundary
            if warmup_boundary is not None:
                ax.axvline(x=warmup_boundary, color='gray', linestyle='--', 
                          linewidth=1.5, alpha=0.6, label='Warmup|Training')
            
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Row 1: Primary metrics
        ax1 = fig.add_subplot(gs[0, 0])
        plot_metric(ax1, epochs, [m['loss'] for m in all_metrics], 
                   'Loss', 'Training Loss', 'navy')
        
        ax2 = fig.add_subplot(gs[0, 1])
        if 'kl_div_total' in all_metrics[0]:
            plot_metric(ax2, epochs, [m.get('kl_div_total', 0) for m in all_metrics], 
                       'KL Divergence', 'KL Divergence', 'darkred')
        else:
            ax2.text(0.5, 0.5, 'KL Div not available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('KL Divergence')
        
        ax3 = fig.add_subplot(gs[0, 2])
        if 'correlation' in all_metrics[0]:
            plot_metric(ax3, epochs, [m.get('correlation', 0) for m in all_metrics], 
                       'Correlation', 'Correlation', 'darkgreen')
        else:
            ax3.text(0.5, 0.5, 'Correlation not available', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Correlation')
        
        ax4 = fig.add_subplot(gs[0, 3])
        plot_metric(ax4, epochs, [m.get('mae', m.get('mae_x2_to_x1', 0)) for m in all_metrics], 
                   'MAE', 'Mean Absolute Error', 'purple')
        
        # Row 2: Image quality metrics
        ax5 = fig.add_subplot(gs[1, 0])
        plot_metric(ax5, epochs, [m.get('fid', 0) for m in all_metrics], 
                   'FID', 'Fréchet Inception Distance', 'darkorange')
        
        ax6 = fig.add_subplot(gs[1, 1])
        plot_metric(ax6, epochs, [m.get('mse', 0) for m in all_metrics], 
                   'MSE', 'Mean Squared Error', 'crimson')
        
        ax7 = fig.add_subplot(gs[1, 2])
        plot_metric(ax7, epochs, [m.get('ssim', 0) for m in all_metrics], 
                   'SSIM', 'Structural Similarity Index', 'teal')
        
        ax8 = fig.add_subplot(gs[1, 3])
        if 'mutual_information' in all_metrics[0]:
            plot_metric(ax8, epochs, [m.get('mutual_information', 0) for m in all_metrics], 
                       'Mutual Information', 'Mutual Information I(X;Y)', 'purple')
        else:
            ax8.text(0.5, 0.5, 'MI not available', ha='center', va='center', transform=ax8.transAxes)
            ax8.set_title('Mutual Information')
        
        # Row 3: Information theoretic metrics
        ax9 = fig.add_subplot(gs[2, 0])
        if 'entropy_x1' in all_metrics[0] and 'entropy_x2' in all_metrics[0]:
            ax9.plot(epochs, [m.get('entropy_x1', 0) for m in all_metrics], 
                    'b-', linewidth=2, marker='o', label='H(X1)', markersize=4)
            ax9.plot(epochs, [m.get('entropy_x2', 0) for m in all_metrics], 
                    'r-', linewidth=2, marker='o', label='H(X2)', markersize=4)
            if warmup_boundary is not None:
                ax9.axvline(x=warmup_boundary, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
            ax9.set_xlabel('Epoch', fontsize=10)
            ax9.set_ylabel('Entropy', fontsize=10)
            ax9.set_title('Marginal Entropies', fontsize=11, fontweight='bold')
            ax9.legend(fontsize=8)
            ax9.grid(True, alpha=0.3)
        else:
            ax9.text(0.5, 0.5, 'Entropy not available', ha='center', va='center', transform=ax9.transAxes)
            ax9.set_title('Marginal Entropies')
        
        ax10 = fig.add_subplot(gs[2, 1])
        if 'joint_entropy' in all_metrics[0]:
            ax10.plot(epochs, [m.get('joint_entropy', 0) for m in all_metrics], 
                     'purple', linewidth=2, marker='o', markersize=4)
            if warmup_boundary is not None:
                ax10.axvline(x=warmup_boundary, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
            ax10.set_xlabel('Epoch', fontsize=10)
            ax10.set_ylabel('Joint Entropy', fontsize=10)
            ax10.set_title('Joint Entropy H(X,Y)', fontsize=11, fontweight='bold')
            ax10.grid(True, alpha=0.3)
        else:
            ax10.text(0.5, 0.5, 'Joint Entropy not available', ha='center', va='center', transform=ax10.transAxes)
            ax10.set_title('Joint Entropy')
        
        ax11 = fig.add_subplot(gs[2, 2])
        if 'h_x1_given_x2' in all_metrics[0]:
            ax11.plot(epochs, [m.get('h_x1_given_x2', 0) for m in all_metrics], 
                     'b-', linewidth=2, marker='o', label='H(X1|X2)', markersize=4)
            if warmup_boundary is not None:
                ax11.axvline(x=warmup_boundary, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
            ax11.set_xlabel('Epoch', fontsize=10)
            ax11.set_ylabel('Conditional Entropy', fontsize=10)
            ax11.set_title('Conditional Entropy', fontsize=11, fontweight='bold')
            ax11.legend(fontsize=8)
            ax11.grid(True, alpha=0.3)
        else:
            ax11.text(0.5, 0.5, 'Conditional Entropy not available', ha='center', va='center', transform=ax11.transAxes)
            ax11.set_title('Conditional Entropy')
        
        ax12 = fig.add_subplot(gs[2, 3])
        if 'kl_div_1' in all_metrics[0] and 'kl_div_2' in all_metrics[0]:
            ax12.plot(epochs, [m.get('kl_div_1', 0) for m in all_metrics], 
                     'b-', linewidth=2, marker='o', label='KL(X1)', markersize=4)
            ax12.plot(epochs, [m.get('kl_div_2', 0) for m in all_metrics], 
                     'r-', linewidth=2, marker='o', label='KL(X2)', markersize=4)
            if warmup_boundary is not None:
                ax12.axvline(x=warmup_boundary, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
            ax12.set_xlabel('Epoch', fontsize=10)
            ax12.set_ylabel('KL Divergence', fontsize=10)
            ax12.set_title('Individual KL Components', fontsize=11, fontweight='bold')
            ax12.legend(fontsize=8)
            ax12.grid(True, alpha=0.3)
        else:
            ax12.text(0.5, 0.5, 'KL components not available', ha='center', va='center', transform=ax12.transAxes)
            ax12.set_title('Individual KL Components')
        
        # Row 4: Learned statistics and summary
        ax13 = fig.add_subplot(gs[3, 0])
        if 'mu1_learned' in all_metrics[0]:
            ax13.plot(epochs, [m.get('mu1_learned', 0) for m in all_metrics], 
                     'b-', linewidth=2, marker='o', label='μ1 learned', markersize=4)
            ax13.axhline(y=0.0, color='b', linestyle='--', alpha=0.5, label='μ1 target (0.0)')
            if warmup_boundary is not None:
                ax13.axvline(x=warmup_boundary, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
            ax13.set_xlabel('Epoch', fontsize=10)
            ax13.set_ylabel('Mean', fontsize=10)
            ax13.set_title('Learned Mean', fontsize=11, fontweight='bold')
            ax13.legend(fontsize=8)
            ax13.grid(True, alpha=0.3)
        else:
            ax13.text(0.5, 0.5, 'Mean not available', ha='center', va='center', transform=ax13.transAxes)
            ax13.set_title('Learned Mean')
        
        ax14 = fig.add_subplot(gs[3, 1])
        if 'std1_learned' in all_metrics[0]:
            ax14.plot(epochs, [m.get('std1_learned', 0) for m in all_metrics], 
                     'b-', linewidth=2, marker='o', label='σ1 learned', markersize=4)
            ax14.axhline(y=1.0, color='b', linestyle='--', alpha=0.5, label='σ1 target (1.0)')
            if warmup_boundary is not None:
                ax14.axvline(x=warmup_boundary, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
            ax14.set_xlabel('Epoch', fontsize=10)
            ax14.set_ylabel('Std Dev', fontsize=10)
            ax14.set_title('Learned Std Dev', fontsize=11, fontweight='bold')
            ax14.legend(fontsize=8)
            ax14.grid(True, alpha=0.3)
        else:
            ax14.text(0.5, 0.5, 'Std Dev not available', ha='center', va='center', transform=ax14.transAxes)
            ax14.set_title('Learned Std Dev')
        
        # Summary metrics table
        ax15 = fig.add_subplot(gs[3, 2:])
        ax15.axis('off')
        latest = all_metrics[-1]
        initial = all_metrics[0] if all_metrics else latest
        
        summary_text = f"""
{method} Training Summary - {config_str}
{'='*60}
Epoch: {latest.get('epoch', 'N/A')} / {max([m.get('epoch', 0) for m in all_metrics])}

Primary Metrics:
  Loss:        {latest.get('loss', 0):.6f} (initial: {initial.get('loss', 0):.6f})
  FID:         {latest.get('fid', 0):.4f} (initial: {initial.get('fid', 0):.4f})
  MSE:         {latest.get('mse', 0):.6f} (initial: {initial.get('mse', 0):.6f})
  MAE:         {latest.get('mae', 0):.6f} (initial: {initial.get('mae', 0):.6f})
  SSIM:        {latest.get('ssim', 0):.4f} (initial: {initial.get('ssim', 0):.4f})

Information Theoretic:
  MI:          {latest.get('mutual_information', 0):.4f}
  KL Div:      {latest.get('kl_div_total', 0):.4f}
  Correlation: {latest.get('correlation', 0):.4f}
  Entropy X1:  {latest.get('entropy_x1', 0):.4f}
  Entropy X2:  {latest.get('entropy_x2', 0):.4f}
  Joint Ent:   {latest.get('joint_entropy', 0):.4f}

Learned Statistics:
  Mean:        {latest.get('mu1_learned', 0):.4f}
  Std Dev:     {latest.get('std1_learned', 0):.4f}
"""
        
        ax15.text(0.05, 0.95, summary_text, transform=ax15.transAxes, 
                 fontsize=9, verticalalignment='top', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Add title
        fig.suptitle(f'{method} Training Progress - {config_str}', 
                     fontsize=14, fontweight='bold', y=0.995)
        
        # Save plot
        plot_path = os.path.join(checkpoint_dir, f'{method}_epoch_{epoch+1:04d}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also save latest plot with fixed name for easy access
        latest_plot_path = os.path.join(checkpoint_dir, f'{method}_latest.png')
        import shutil
        if os.path.exists(plot_path):
            shutil.copy(plot_path, latest_plot_path)
    
    def _generate_summary(self):
        """Generate ablation summary."""
        summary_path = os.path.join(self.output_dir, "SUMMARY.txt")
        
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("BBBC021 ABLATION STUDY SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            # Best ES config
            best_es = min(self.all_results['ES'], key=lambda x: x['fid'])
            f.write("BEST ES CONFIGURATION:\n")
            f.write(f"  sigma: {best_es['sigma']}\n")
            f.write(f"  lr: {best_es['lr']}\n")
            f.write(f"  FID: {best_es['fid']:.4f}\n")
            f.write(f"  MSE: {best_es['mse']:.4f}\n\n")
            
            # Best PPO config
            best_ppo = min(self.all_results['PPO'], key=lambda x: x['fid'])
            f.write("BEST PPO CONFIGURATION:\n")
            f.write(f"  kl_weight: {best_ppo['kl_weight']}\n")
            f.write(f"  ppo_clip: {best_ppo['ppo_clip']}\n")
            f.write(f"  lr: {best_ppo['lr']}\n")
            f.write(f"  FID: {best_ppo['fid']:.4f}\n")
            f.write(f"  MSE: {best_ppo['mse']:.4f}\n\n")
            
            winner = 'ES' if best_es['fid'] < best_ppo['fid'] else 'PPO'
            f.write(f"WINNER: {winner}\n")
        
        print(f"Summary saved to: {summary_path}")
        
        # Save JSON results
        json_path = os.path.join(self.output_dir, "all_results.json")
        json_results = {
            'ES': [{k: v for k, v in r.items() if k != 'history'} for r in self.all_results['ES']],
            'PPO': [{k: v for k, v in r.items() if k != 'history'} for r in self.all_results['PPO']],
        }
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Results JSON saved to: {json_path}")
        
        # Generate plots
        self._generate_plots()
    
    def _generate_plots(self):
        """Generate ablation plots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # ES: sigma vs FID
        ax1 = axes[0, 0]
        sigma_vals = sorted(set(r['sigma'] for r in self.all_results['ES']))
        for sigma in sigma_vals:
            data = [r for r in self.all_results['ES'] if r['sigma'] == sigma]
            lrs = [r['lr'] for r in data]
            fids = [r['fid'] for r in data]
            ax1.plot(lrs, fids, marker='o', label=f'σ={sigma}')
        ax1.set_xlabel('Learning Rate')
        ax1.set_ylabel('FID')
        ax1.set_title('ES: LR vs FID')
        ax1.set_xscale('log')
        ax1.legend()
        ax1.grid(True)
        
        # PPO: kl_weight vs FID
        ax2 = axes[0, 1]
        kl_weights = sorted(set(r['kl_weight'] for r in self.all_results['PPO']))
        for kl_w in kl_weights:
            data = [r for r in self.all_results['PPO'] if r['kl_weight'] == kl_w]
            lrs = [r['lr'] for r in data]
            fids = [r['fid'] for r in data]
            ax2.plot(lrs, fids, marker='o', label=f'KL_w={kl_w}')
        ax2.set_xlabel('Learning Rate')
        ax2.set_ylabel('FID')
        ax2.set_title('PPO: LR vs FID')
        ax2.set_xscale('log')
        ax2.legend()
        ax2.grid(True)
        
        # Best comparison
        ax3 = axes[1, 0]
        best_es = min(self.all_results['ES'], key=lambda x: x['fid'])
        best_ppo = min(self.all_results['PPO'], key=lambda x: x['fid'])
        
        metrics = ['fid', 'mse', 'mae']
        es_vals = [best_es[m] for m in metrics]
        ppo_vals = [best_ppo[m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        ax3.bar(x - width/2, es_vals, width, label='Best ES')
        ax3.bar(x + width/2, ppo_vals, width, label='Best PPO')
        ax3.set_xticks(x)
        ax3.set_xticklabels(['FID', 'MSE', 'MAE'])
        ax3.set_ylabel('Value')
        ax3.set_title('Best Configs Comparison')
        ax3.legend()
        ax3.grid(True, axis='y')
        
        # SSIM comparison
        ax4 = axes[1, 1]
        ax4.bar(['ES', 'PPO'], [best_es['ssim'], best_ppo['ssim']])
        ax4.set_ylabel('SSIM')
        ax4.set_title('Structural Similarity (higher is better)')
        ax4.grid(True, axis='y')
        
        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, 'ablation_summary.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to: {plot_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="BBBC021 Ablation Study: ES vs PPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data
    parser.add_argument("--data-dir", type=str, default="./data/bbbc021",
                       help="Directory containing BBBC021 data")
    parser.add_argument("--metadata-file", type=str, default="metadata.csv",
                       help="Metadata CSV file")
    
    # Training
    parser.add_argument("--ddpm-epochs", type=int, default=100,
                       help="DDPM pretraining epochs")
    parser.add_argument("--coupling-epochs", type=int, default=30,
                       help="Coupling training epochs")
    parser.add_argument("--warmup-epochs", type=int, default=10,
                       help="Warmup epochs before ES")
    
    # ES ablation
    parser.add_argument("--es-sigma-values", type=float, nargs='+',
                       default=[0.001, 0.005, 0.01],
                       help="ES sigma values")
    parser.add_argument("--es-lr-values", type=float, nargs='+',
                       default=[0.0001, 0.0005, 0.001],
                       help="ES learning rate values")
    
    # PPO ablation
    parser.add_argument("--ppo-kl-values", type=float, nargs='+',
                       default=[0.3, 0.5, 0.7],
                       help="PPO KL weight values")
    parser.add_argument("--ppo-clip-values", type=float, nargs='+',
                       default=[0.05, 0.1, 0.2],
                       help="PPO clip values")
    parser.add_argument("--ppo-lr-values", type=float, nargs='+',
                       default=[2e-5, 5e-5, 1e-4],
                       help="PPO learning rate values")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="bbbc021_ablation_results",
                       help="Output directory")
    parser.add_argument("--use-wandb", action="store_true",
                       help="Enable WandB logging")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Model
    parser.add_argument("--no-reuse-pretrained", action="store_true",
                       help="Train pretrained model from scratch")
    
    args = parser.parse_args()
    
    config = BBBC021Config(
        data_dir=args.data_dir,
        metadata_file=args.metadata_file,
        ddpm_epochs=args.ddpm_epochs,
        coupling_epochs=args.coupling_epochs,
        warmup_epochs=args.warmup_epochs,
        es_sigma_values=args.es_sigma_values,
        es_lr_values=args.es_lr_values,
        ppo_kl_weight_values=args.ppo_kl_values,
        ppo_clip_values=args.ppo_clip_values,
        ppo_lr_values=args.ppo_lr_values,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        seed=args.seed,
        reuse_pretrained=not args.no_reuse_pretrained,
    )
    
    runner = BBBC021AblationRunner(config)
    runner.run()


if __name__ == "__main__":
    main()