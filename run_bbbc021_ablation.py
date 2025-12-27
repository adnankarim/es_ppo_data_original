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
import glob
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
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    print("WARNING: torchmetrics not found. FID will be approximate. Install with: pip install torchmetrics")

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("WARNING: RDKit not available. Install with: pip install rdkit")
    print("         Morgan fingerprints will use random embeddings instead.")

try:
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("WARNING: sklearn not available. MoA metrics will be disabled.")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class BBBC021Config:
    """Configuration for BBBC021 ablation study."""
    
    # Experiment Mode
    mode: str = "ablation"  # "ablation" or "single"
    method: str = "PPO"     # "ES" or "PPO" (for single mode)
    
    # Resume / IDs
    resume_exp_id: Optional[str] = None  # If set, resume this specific run folder
    resume_pretrain: bool = False        # If True, continue training the global base model
    skip_optimizer_on_resume: bool = False  # If True, don't load optimizer state (use new LR)
    
    # Data paths
    # For IMPA dataset on Windows, use: r"C:\Users\dell\Downloads\IMPA_reproducibility\IMPA_reproducibility\datasets\bbbc021_all\bbbc021_all"
    data_dir: str = "./data/bbbc021_all"  # Directory containing BBBC021 data (supports nested .npy files)
    metadata_file: str = "metadata/bbbc021_df_all.csv"  # Metadata CSV file (or "metadata/bbbc021_df_all.csv" for IMPA)
    
    # Global Model Management
    global_model_dir: str = "./global_pretrained_models"
    force_pretrain: bool = False  # Ignore global model, train local fresh
    
    # Image settings
    image_size: int = 96
    num_channels: int = 3  # DNA, F-actin, β-tubulin
    
    # Perturbation encoding
    morgan_bits: int = 1024  # Morgan fingerprint dimensions
    perturbation_embed_dim: int = 256
    
    # DDPM pretraining
    ddpm_epochs: int = 500  # Increased from 100 - DDPMs need more epochs to converge (FID < 200)
    # Reduced LR for better convergence on stalling models
    ddpm_lr: float = 1e-4  # Reduced from 3e-4 - prevents "bouncing" around local minima
    # Reduced batch size for better gradient stability (can use grad_accumulation if needed)
    ddpm_batch_size: int = 512  # Reduced from 1024 - better gradient stability
    ddpm_timesteps: int = 1000
    
    # Coupling training
    coupling_epochs: int = 30
    # Optimized for 144GB GPU (GH200) - handles control+perturbed images
    coupling_batch_size: int = 512  # Increased from 256
    warmup_epochs: int = 10
    num_sampling_steps: int = 50
    
    # ES Ablations
    es_population_size: int = 50  # Increased default for better gradient estimation
    es_sigma_values: List[float] = field(default_factory=lambda: [0.001, 0.005, 0.01])
    es_lr_values: List[float] = field(default_factory=lambda: [0.0001, 0.0005, 0.001])
    
    # PPO Ablations
    ppo_kl_weight_values: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])
    ppo_clip_values: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2])
    ppo_lr_values: List[float] = field(default_factory=lambda: [2e-5, 5e-5, 1e-4])
    
    # --- Single Config Params (Used if mode='single') ---
    single_es_sigma: float = 0.005
    single_es_lr: float = 0.0005
    single_ppo_kl: float = 0.4  # Lower KL weight for better convergence
    single_ppo_clip: float = 0.08  # Tighter clip for stability
    single_ppo_lr: float = 3e-5  # Lower LR for stability and convergence
    
    # U-Net architecture
    # Pixel-Space Settings: 96x96 needs significant depth (Report Section 1.3)
    unet_channels: List[int] = field(default_factory=lambda: [128, 256, 512, 512])
    time_embed_dim: int = 256
    
    # Output
    output_dir: str = "bbbc021_ablation_results"
    use_wandb: bool = True
    wandb_project: str = "bbbc021-ddmec-ablation"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    
    # Model management
    reuse_pretrained: bool = True
    
    # Biological constraints
    enable_bio_loss: bool = False  # Enable DNA preservation loss in PPO
    
    # Evaluation
    num_eval_samples: int = 5000
    fid_batch_size: int = 64
    
    # Memory optimization
    gradient_checkpointing: bool = True
    mixed_precision: bool = True
    
    # Optimization
    num_workers: int = 32 if os.name != 'nt' else 0
    pin_memory: bool = True


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
        """
        Load metadata with Robust Substring Splitting for Windows/IMPA.
        """
        metadata_path = self.data_dir / metadata_file
        
        if not metadata_path.exists():
            print(f"Warning: Metadata file not found at {metadata_path}")
            print("Generating synthetic metadata for testing...")
            return self._generate_synthetic_metadata()
        
        # CellFlux OOD Compound List
        ood_compounds = {
            'docetaxel', 'AZ841', 'cytochalasin D', 'simvastatin', 
            'cyclohexamide', 'latrunculin B', 'epothilone B', 'lactacystin',
            'Docetaxel', 'Cytochalasin D', 'Simvastatin', 
            'Cyclohexamide', 'Latrunculin B', 'Epothilone B', 'Lactacystin'
        }

        metadata = []
        # Debugging set to see what batches we actually find
        found_batches = set()

        with open(metadata_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 1. Basic Info
                compound = row.get('compound') or row.get('CPD_NAME') or 'DMSO'
                moa = row.get('moa') or row.get('ANNOT') or row.get('MOA') or ''
                # Get batch and strip whitespace
                batch_name = str(row.get('batch', row.get('BATCH', row.get('Plate', row.get('PLATE', '0'))))).strip()
                found_batches.add(batch_name)

                # 2. Determine Split (Robust Substring Logic)
                row_split = row.get('split') or row.get('SPLIT') or ''
                
                if not row_split:
                    # Check for keywords in the batch name (case insensitive)
                    b_upper = batch_name.upper()
                    
                    # Test Set: Week 10 or Batch 09
                    if 'WEEK10' in b_upper or 'WEEK 10' in b_upper or 'BATCH_09' in b_upper:
                        row_split = 'test'
                    # Validation Set: Week 8, 9 or Batch 07, 08
                    elif any(x in b_upper for x in ['WEEK8', 'WEEK 8', 'WEEK9', 'WEEK 9', 'BATCH_07', 'BATCH_08']):
                        row_split = 'val'
                    # Training Set: Everything else
                    else:
                        row_split = 'train'
                else:
                    # Normalize split value to lowercase
                    row_split = row_split.lower()

                # 3. Filtering
                if row_split != self.split:
                    continue

                # Filter by OOD (Training only)
                is_ood = compound in ood_compounds
                if self.split == 'train' and is_ood:
                    continue

                # 4. Path Construction
                filename = row.get('image_path') or row.get('SAMPLE_KEY') or row.get('file_path') or ''
                if filename and '/' not in filename and '\\' not in filename:
                    week = row.get('week') or row.get('Week') or ''
                    plate = row.get('plate') or row.get('Plate') or row.get('PLATE') or ''
                    if week and plate:
                         full_image_path = os.path.join(week, plate, filename)
                    else:
                         full_image_path = filename
                else:
                    full_image_path = filename

                # 5. Control Logic
                is_control_val = row.get('is_control', '')
                if is_control_val != '':
                    if isinstance(is_control_val, str):
                        is_control = is_control_val.lower() in ('true', '1', 'yes')
                    else:
                        is_control = bool(is_control_val)
                else:
                    is_control = (compound.upper() == 'DMSO')

                metadata.append({
                    'image_path': full_image_path,
                    'compound': compound,
                    'concentration': float(row.get('concentration', row.get('DOSE', row.get('CONCENTRATION', 0)))),
                    'moa': moa,
                    'batch': batch_name,
                    'is_control': is_control,
                    'smiles': row.get('smiles', row.get('SMILES', '')),
                })
        
        # Debug print only once per dataset initialization
        if len(metadata) > 0 and self.split == 'train':
            print(f"DEBUG: Found batches in CSV: {sorted(list(found_batches))[:20]}...")  # Show first 20 to avoid spam

        # --- SAFETY FALLBACK FOR EMPTY SPLITS ---
        # If we asked for validation but got nothing (due to batch naming mismatch),
        # force a random split so the code doesn't crash.
        if self.split == 'val' and len(metadata) == 0:
            print("WARNING: No validation batches found matching naming convention.")
            print("         Generating random validation split (10% of data).")
            
            # Re-read everything ignoring split rules
            all_data = []
            with open(metadata_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Basic parsing (copy from above or simplify)
                    compound = row.get('compound') or row.get('CPD_NAME') or 'DMSO'
                    filename = row.get('image_path') or row.get('SAMPLE_KEY') or ''
                    
                    # Skip OOD for safety
                    if compound in ood_compounds: 
                        continue
                    
                    # Quick path fix
                    if filename and '/' not in filename and '\\' not in filename:
                         week = row.get('week') or row.get('Week') or ''
                         plate = row.get('plate') or row.get('Plate') or row.get('PLATE') or ''
                         if week and plate: 
                             filename = os.path.join(week, plate, filename)

                    all_data.append({
                        'image_path': filename,
                        'compound': compound,
                        'concentration': float(row.get('concentration', row.get('DOSE', row.get('CONCENTRATION', 0)))),
                        'moa': row.get('moa') or row.get('ANNOT') or row.get('MOA') or '',
                        'batch': 'fallback_batch',
                        'is_control': (compound.upper() == 'DMSO'),
                        'smiles': row.get('smiles', row.get('SMILES', '')),
                    })
            
            # Deterministic Random Split
            np.random.seed(42)
            indices = np.random.permutation(len(all_data))
            n_val = int(0.1 * len(all_data))
            val_indices = indices[:n_val]
            metadata = [all_data[i] for i in val_indices]

        return metadata
    
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
        """[FIXED] Load IMPA 96x96 uint8 crops with Global Scaling."""
        if not image_path.endswith('.npy'):
            image_path += '.npy'
            
        full_path = self.data_dir / image_path
        
        if full_path.exists():
            try:
                # 1. Load data (uint8 0-255)
                img_array = np.load(str(full_path)) 
                
                # 2. To Float Tensor & Channel First: [3, 96, 96]
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
                
                # 3. GLOBAL SCALING (Fixes the Variance Trap)
                # Maps [0, 255] -> [-1, 1]
                img_tensor = (img_tensor / 127.5) - 1.0
                
                return img_tensor
            except Exception as e:
                return self._generate_synthetic_image()
        else:
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
        num_workers: int = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        # Windows compatibility: default to 0 on Windows to avoid frozen process bug
        # Optimized for 144GB GPU - increase workers to prevent CPU bottleneck
        self.num_workers = num_workers if num_workers is not None else (0 if os.name == 'nt' else 32)
        
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
            
            moa_indices = []
            compounds = []
            
            for perturbed_idx in batch_indices:
                control_idx, perturbed_idx = self.dataset.get_batch_paired_sample(perturbed_idx)
                
                control_data = self.dataset[control_idx]
                perturbed_data = self.dataset[perturbed_idx]
                
                controls.append(control_data['image'])
                perturbeds.append(perturbed_data['image'])
                fingerprints.append(perturbed_data['fingerprint'])
                moa_indices.append(perturbed_data['moa_idx'])
                compounds.append(perturbed_data['compound'])  # Compound name (string)
            
            yield {
                'control': torch.stack(controls),
                'perturbed': torch.stack(perturbeds),
                'fingerprint': torch.stack(fingerprints),
                'moa_idx': torch.tensor(moa_indices),
                'compound': compounds,  # List of compound names (strings)
                'compound': compounds,  # List of compound names
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
    """
    Evolution Strategies trainer for image DDPM coupling.
    Now includes Biological Consistency Loss (DNA Preservation).
    """
    
    def __init__(
        self,
        ddpm: ImageDDPM,
        population_size: int = 50,  # Increased default for better gradient estimation
        sigma: float = 0.005,
        lr: float = 0.0005,
        device: str = "cuda",
        use_bio_loss: bool = False,  # [NEW] Flag for DNA preservation
    ):
        self.ddpm = ddpm
        self.population_size = population_size
        self.sigma = sigma
        self.lr = lr
        self.device = torch.device(device)
        self.use_bio_loss = use_bio_loss
    
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
            
            # 1. Base MSE Loss (Reconstruction)
            loss = F.mse_loss(noise_pred, noise)
            
            # 2. [NEW] Biological Consistency Loss (DNA Preservation)
            # Channel 0 is DAPI (DNA/Nucleus). We enforce that the nucleus noise 
            # matches the ground truth more strictly than the cytoskeleton.
            if self.use_bio_loss:
                dna_loss = F.mse_loss(noise_pred[:, 0, :, :], noise[:, 0, :, :])
                loss = loss + 0.1 * dna_loss  # 0.1 weight matches PPO settings
            
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
        use_bio_loss: bool = False,
    ):
        self.cond_model = cond_model
        self.pretrain_model = pretrain_model
        self.kl_weight = kl_weight
        self.ppo_clip = ppo_clip
        self.device = torch.device(device)
        self.use_bio_loss = use_bio_loss  # Store the flag
        
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
            if self.pretrain_model.conditional:
                # Encode fingerprint using the PRETRAINED encoder (frozen)
                # We must use the pretrained model's encoder, not the active one
                pt_emb = self.pretrain_model.perturbation_encoder(fingerprint_batch)
                
                # Pass control and embedding to the reference model
                noise_pred_pretrain = self.pretrain_model.model(x_t, t, control_batch, pt_emb)
            else:
                # Fallback for old unconditional models
                noise_pred_pretrain = self.pretrain_model.model(x_t, t)
        
        kl_loss = F.mse_loss(noise_pred_cond, noise_pred_pretrain)
        
        # Base loss
        total_loss = reconstruction_loss + self.kl_weight * kl_loss
        
        # Conditional Biological Consistency Loss (MoA Regularization)
        # Ensures DNA channel (channel 0 = DAPI) is preserved - prevents hallucinating new cell locations
        # This ensures the model doesn't just change everything, but preserves the source DNA
        if self.use_bio_loss:
            # FIX: DAPI (DNA) is Index 0 in BBBC021Dataset, not Index 2.
            # Channel order: 0=DAPI (DNA), 1=Phalloidin (Actin), 2=Tubulin
            # We enforce that DNA noise prediction matches ground truth to anchor the nucleus.
            # This prevents the model from moving nuclei while allowing cytoskeleton changes.
            dna_preservation = F.mse_loss(noise_pred_cond[:, 0, :, :], noise[:, 0, :, :])
            total_loss += 0.1 * dna_preservation
        
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
    """
    [FIXED] Metrics suite matching CellFlux (ICML 2025) protocols.
    - Fixes 'dtype=torch.uint8' crash by casting tensors before Inception.
    - Uses InceptionV3 features for MoA Accuracy (Deep Proxy).
    - Supports Conditional FID (per-compound).
    - Scaled KID calculation.
    - Extracts features ONCE and reuses for FID, FIDc, and MoA.
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.inception = None
        self.fid_metric = None
        if TORCHMETRICS_AVAILABLE:
            # Initialize InceptionV3 once to save memory
            from torchmetrics.image.fid import FrechetInceptionDistance
            self.fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
            # Access the internal inception model for feature extraction
            self.inception = self.fid_metric.inception
            self.inception.eval()

    @torch.no_grad()
    def get_features(self, images: np.ndarray) -> torch.Tensor:
        """Extract deep features (2048-dim) from images using InceptionV3."""
        if self.inception is None:
            return None
        
        # 1. Convert to Float Tensor on Device
        images_t = torch.tensor(images).float().to(self.device)
        
        # 2. Normalize to [0, 1] Range
        # If images are [-1, 1] (standard DDPM output), map to [0, 1]
        if images_t.min() < 0:
            images_t = (images_t + 1.0) / 2.0
        images_t = torch.clamp(images_t, 0.0, 1.0)
        
        features = []
        batch_size = 32
        
        for i in range(0, len(images_t), batch_size):
            batch = images_t[i:i+batch_size]
            
            # 3. Resize to 299x299 (Inception Standard)
            if batch.shape[-1] != 299 or batch.shape[-2] != 299:
                batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
            
            # 4. [CRITICAL FIX] Convert to uint8 [0, 255]
            # The internal torch-fidelity module explicitly checks for uint8
            batch_uint8 = (batch * 255).to(torch.uint8)
            
            # 5. Extract Features
            feats = self.inception(batch_uint8)
            features.append(feats.cpu())
            
        return torch.cat(features)

    @torch.no_grad()
    def compute_fid_from_features(self, real_features: torch.Tensor, fake_features: torch.Tensor) -> float:
        """Calculate FID using pre-computed features (Fast)."""
        if real_features is None or fake_features is None:
            return 0.0
        # Manual FID calculation using Gaussian statistics
        mu1, sigma1 = self._compute_stats(real_features.numpy())
        mu2, sigma2 = self._compute_stats(fake_features.numpy())
        return self._calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    @staticmethod
    def _compute_stats(features):
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    @staticmethod
    def _calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of FID calculation."""
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            covmean = np.eye(sigma1.shape[0])
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        tr_covmean = np.trace(covmean)
        return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

    @staticmethod
    @torch.no_grad()
    def compute_kid(real_images: np.ndarray, fake_images: np.ndarray, device='cuda') -> float:
        """Computes KID scaled by 1000 (Target: ~1.62)."""
        if not TORCHMETRICS_AVAILABLE:
            return 0.0
        
        kid = KernelInceptionDistance(subset_size=100, normalize=True).to(device)
        real_t = torch.tensor(real_images).float().to(device)
        fake_t = torch.tensor(fake_images).float().to(device)
        
        # Normalize
        if real_t.min() < 0:
            real_t = (real_t + 1.0) / 2.0
            fake_t = (fake_t + 1.0) / 2.0
            
        kid.update(torch.clamp(real_t, 0, 1), real=True)
        kid.update(torch.clamp(fake_t, 0, 1), real=False)
        
        kid_mean, _ = kid.compute()
        return float(kid_mean.item()) * 1000.0

    @staticmethod
    def compute_deep_moa_accuracy(real_features: np.ndarray, fake_features: np.ndarray, labels: np.ndarray) -> float:
        """
        1-NN Accuracy using Deep Inception Features.
        This is a 'Strong Proxy' for the CellFlux ResNet evaluation.
        """
        if not SKLEARN_AVAILABLE or len(labels) == 0:
            return 0.0
        
        knn = KNeighborsClassifier(n_neighbors=1, metric='cosine')
        # Train on Real
        knn.fit(real_features, labels)
        # Test on Fake
        preds = knn.predict(fake_features)
        
        return float(np.mean(preds == labels))

    @staticmethod
    def compute_pixel_change(real_controls: np.ndarray, fake_perturbed: np.ndarray) -> float:
        """
        Calculates L1 Pixel Difference. 
        Detects 'Lazy Models' that just output the input image.
        Target: > 0.02 (Small change) but < 0.2 (Total destruction).
        """
        diff = np.abs(real_controls - fake_perturbed)
        return float(np.mean(diff))

    @staticmethod
    def compute_mse(real_images: np.ndarray, fake_images: np.ndarray) -> float:
        return float(np.mean((real_images - fake_images) ** 2))
    
    @staticmethod
    def compute_mae(real_images: np.ndarray, fake_images: np.ndarray) -> float:
        return float(np.mean(np.abs(real_images - fake_images)))
    
    @staticmethod
    def compute_ssim(real_images: np.ndarray, fake_images: np.ndarray) -> float:
        # Simplified SSIM for 3-channel images
        c1, c2 = 0.01**2, 0.03**2
        mu_real = np.mean(real_images, axis=(1, 2, 3), keepdims=True)
        mu_fake = np.mean(fake_images, axis=(1, 2, 3), keepdims=True)
        sigma_real_sq = np.var(real_images, axis=(1, 2, 3), keepdims=True)
        sigma_fake_sq = np.var(fake_images, axis=(1, 2, 3), keepdims=True)
        sigma_real_fake = np.mean((real_images - mu_real) * (fake_images - mu_fake), axis=(1, 2, 3), keepdims=True)
        ssim = ((2 * mu_real * mu_fake + c1) * (2 * sigma_real_fake + c2)) / \
               ((mu_real**2 + mu_fake**2 + c1) * (sigma_real_sq + sigma_fake_sq + c2))
        return float(np.mean(ssim))
    
    # Legacy static method for backward compatibility (used in _nscb_benchmark)
    @staticmethod
    def compute_fid(real_images: np.ndarray, fake_images: np.ndarray, device='cuda') -> float:
        """Legacy method: Computes FID using InceptionV3 features (creates new metric each time)."""
        if not TORCHMETRICS_AVAILABLE:
            return 0.0

        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        
        # Convert to Tensor [0, 1] range
        real_t = torch.tensor(real_images).float().to(device)
        fake_t = torch.tensor(fake_images).float().to(device)
        
        if real_t.min() < 0:
            real_t = (real_t + 1.0) / 2.0
            fake_t = (fake_t + 1.0) / 2.0
            
        real_t = torch.clamp(real_t, 0.0, 1.0)
        fake_t = torch.clamp(fake_t, 0.0, 1.0)
        
        fid.update(real_t, real=True)
        fid.update(fake_t, real=False)
        
        return float(fid.compute().item())

# [INSERT THIS CLASS AFTER ImageMetrics AND BEFORE BBBC021AblationRunner]

class ApproximateMetrics:
    """
    Computes Information Theoretic proxies + Biological Fidelity Metrics.
    Uses Histogram-based estimation for entropy and MI.
    CellFlux (ICML 2025) methodology for morphological profile comparison.
    """
    
    @staticmethod
    def _get_bio_features(imgs: np.ndarray) -> np.ndarray:
        """Extracts Morphological Profile (Mean + Std per channel)."""
        if imgs.ndim == 3:
            imgs = imgs[np.newaxis, ...]
        if len(imgs.shape) != 4:
            return np.zeros((imgs.shape[0], 6))

        # Explicitly compute per-channel statistics
        means = np.mean(imgs, axis=(2, 3))  # Shape (N, 3)
        stds = np.std(imgs, axis=(2, 3))    # Shape (N, 3)
        
        return np.hstack([means, stds])

    @staticmethod
    def compute_all(real_images: np.ndarray, fake_images: np.ndarray):
        """
        Compute biological fidelity metrics with histogram-based information-theoretic estimations.
        
        Returns:
            Dictionary with profile_similarity, morpho_drift, and populated info-theoretic metrics
        """
        # Flatten
        real_flat = real_images.flatten()
        fake_flat = fake_images.flatten()
        
        # 1. Standard Biological Metrics
        feat_real = ApproximateMetrics._get_bio_features(real_images)
        feat_fake = ApproximateMetrics._get_bio_features(fake_images)
        
        dot = np.sum(feat_real * feat_fake, axis=1)
        norm_r = np.linalg.norm(feat_real, axis=1)
        norm_f = np.linalg.norm(feat_fake, axis=1)
        profile_sim = np.mean(dot / (norm_r * norm_f + 1e-8))
        drift = np.mean(np.linalg.norm(feat_real - feat_fake, axis=1))
        
        mu_fake = np.mean(fake_flat)
        std_fake = np.std(fake_flat)
        correlation = np.corrcoef(real_flat[::100], fake_flat[::100])[0, 1] if len(real_flat) > 100 else 0.0

        # 2. Information Theoretic Estimations (Histogram Method)
        bins = 50
        hist_range = (-1, 1)

        # Marginal PDF P(X) and P(Y)
        p_real, _ = np.histogram(real_flat, bins=bins, range=hist_range, density=True)
        p_fake, _ = np.histogram(fake_flat, bins=bins, range=hist_range, density=True)
        
        # Normalize and add epsilon
        p_real = p_real / (p_real.sum() + 1e-10) + 1e-10
        p_fake = p_fake / (p_fake.sum() + 1e-10) + 1e-10

        # Marginal Entropies H(X), H(Y)
        h_real = -np.sum(p_real * np.log(p_real))
        h_fake = -np.sum(p_fake * np.log(p_fake))

        # Joint Probability P(X,Y) approximation
        # We assume independence for the joint histogram estimation to save speed,
        # or use 2D histogram on a subset of data
        subset_size = min(len(real_flat), 100000) # Use subset for speed
        H, _, _ = np.histogram2d(real_flat[:subset_size], fake_flat[:subset_size], bins=bins, range=[hist_range, hist_range], density=True)
        p_joint = H / (H.sum() + 1e-10) + 1e-10
        
        # Joint Entropy H(X,Y)
        h_joint = -np.sum(p_joint * np.log(p_joint))

        # Mutual Information I(X;Y) = H(X) + H(Y) - H(X,Y)
        mi = h_real + h_fake - h_joint
        
        # KL Divergence D(P||Q)
        kl_div = np.sum(p_real * np.log(p_real / p_fake))

        # Conditional Entropy H(X|Y) = H(X,Y) - H(Y)
        h_cond = h_joint - h_fake

        return {
            'profile_similarity': float(profile_sim),
            'morpho_drift': float(drift),
            'correlation': float(correlation),
            'mu1_learned': float(mu_fake),
            'std1_learned': float(std_fake),
            
            # Populated Metrics
            'kl_div_total': float(kl_div),
            'mutual_information': float(max(0.0, mi)), # MI >= 0
            'entropy_x1': float(h_real), 
            'entropy_x2': float(h_fake),
            'joint_entropy': float(h_joint),
            'h_x1_given_x2': float(max(0.0, h_cond)),
            
            # Directed metrics (approximated as symmetric or simple diffs for viz)
            'kl_div_1': float(kl_div * 0.5), # Split for visualization
            'kl_div_2': float(kl_div * 0.5),
            'mi_x2_to_x1': float(max(0.0, mi)),
            'mi_x1_to_x2': float(max(0.0, mi)),
            
            'mae_x2_to_x1': float(np.mean(np.abs(real_flat - fake_flat))),
        }
# ============================================================================
# ABLATION RUNNER
# ============================================================================

class BBBC021AblationRunner:
    """Run ablation study on BBBC021 dataset."""
    
    def __init__(self, config: BBBC021Config):
        self.config = config
        
        # [NEW] Session Cache: Stores the specific CSV path chosen for this run
        # Prevents creating new files for every epoch; locks to one file per session.
        self._session_csv_paths = {}
        
        # Set seeds
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
        
        # 1. Handle Run Directory & Resume
        if config.resume_exp_id:
            # Look for existing directory
            candidates = glob.glob(os.path.join(config.output_dir, f"run_{config.resume_exp_id}*"))
            if not candidates:
                raise ValueError(f"Could not find experiment to resume with ID: {config.resume_exp_id}")
            self.output_dir = candidates[0]
            print(f"RESUMING Experiment from: {self.output_dir}")
        else:
            # Create new directory
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(config.output_dir, f"run_{timestamp}_{config.mode}")
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"STARTING New Experiment at: {self.output_dir}")
        
        self.models_dir = os.path.join(self.output_dir, "models")
        self.plots_dir = os.path.join(self.output_dir, "plots")
        self.logs_dir = os.path.join(self.output_dir, "logs")
        
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Pretrained models directory
        self.pretrained_dir = os.path.join(config.output_dir, "pretrained_models")
        os.makedirs(self.pretrained_dir, exist_ok=True)
        
        # Global model directory
        os.makedirs(config.global_model_dir, exist_ok=True)
        
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
            if config.resume_exp_id:
                run_id = config.resume_exp_id
                resume_mode = "allow"
            else:
                # Generate a unique ID
                try:
                    run_id = wandb.util.generate_id()
                except:
                    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                resume_mode = None
            wandb.init(
                project=config.wandb_project,
                id=run_id,
                resume=resume_mode,
                name=f"bbbc021_{config.mode}_{run_id}",
                config=asdict(config)
            )
        
        # Store results
        self.all_results = {'ES': [], 'PPO': []}
        
        print(f"Output directory: {self.output_dir}")
        print(f"Device: {config.device}")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
    
    def _load_checkpoint_if_exists(self, model, optimizer, filename, skip_optimizer=False):
        """Attempts to load a checkpoint. Returns epoch number.
        
        Args:
            model: Model to load weights into
            optimizer: Optimizer to load state into (optional)
            filename: Checkpoint filename to load
            skip_optimizer: If True, don't load optimizer state (useful when changing LR)
        """
        # Check global first for pretraining
        if "pretrain" in filename and not self.config.force_pretrain:
            global_path = os.path.join(self.config.global_model_dir, "ddpm_base_latest.pt")
            if os.path.exists(global_path):
                print(f"Loading GLOBAL pretrain checkpoint: {global_path}")
                ckpt = torch.load(global_path, map_location=self.config.device)
                model.model.load_state_dict(ckpt['model_state_dict'])
                # Only load optimizer if we are RESUMING pretraining and not skipping
                if self.config.resume_pretrain and optimizer and 'optimizer_state_dict' in ckpt and not skip_optimizer:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                    return ckpt.get('epoch', 0)
                elif skip_optimizer:
                    print("  Note: Skipping optimizer state (new learning rate will be used)")
                # If not resuming, check if model is complete
                if ckpt.get('epoch', 0) >= self.config.ddpm_epochs:
                    return self.config.ddpm_epochs  # Return max epochs so we skip training
                return 0  # Model exists but incomplete, will train from start

        # Check local run dir
        local_path = os.path.join(self.models_dir, filename)
        if os.path.exists(local_path):
            print(f"Loading LOCAL checkpoint: {local_path}")
            ckpt = torch.load(local_path, map_location=self.config.device)
            model.model.load_state_dict(ckpt['model_state_dict'])
            if optimizer and 'optimizer_state_dict' in ckpt and ckpt['optimizer_state_dict'] is not None and not skip_optimizer:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            elif skip_optimizer:
                print("  Note: Skipping optimizer state (new learning rate will be used)")
            return ckpt.get('epoch', 0)
        
        return 0

    def _save_checkpoint(self, model, optimizer, epoch, filename, is_global=False):
        """Save checkpoint for model and optimizer."""
        state = {
            'epoch': epoch,
            'model_state_dict': model.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None
        }
        if model.perturbation_encoder:
            state['perturbation_encoder_state_dict'] = model.perturbation_encoder.state_dict()
            
        # Save to local run
        torch.save(state, os.path.join(self.models_dir, filename))
        
        # Save to global if applicable
        if is_global:
            torch.save(state, os.path.join(self.config.global_model_dir, "ddpm_base_latest.pt"))
    
    def run(self):
        """Run the ablation study or single experiment."""
        print("\n" + "=" * 80)
        if self.config.mode == "ablation":
            print("BBBC021 ABLATION STUDY: ES vs PPO")
        else:
            print(f"BBBC021 SINGLE EXPERIMENT: {self.config.method}")
        print("=" * 80 + "\n")
        
        start_time = time.time()
        
        # Step 1: Pretrain unconditional DDPM
        print("Step 1: Pretraining unconditional DDPM on control images...")
        
        # [UPDATED] Unpack both model and metrics
        pretrain_ddpm, best_pretrain_metrics = self._pretrain_ddpm()
        
        # [NEW] Print Final Pretraining Baseline
        print("\n" + "=" * 60)
        print("FINAL PRETRAINING BASELINE (Stage 1 & 2 Complete)")
        print("=" * 60)
        
        # Try to load from latest CSV first (more reliable for resumed runs)
        csv_metrics = self._load_latest_metrics_from_csv(self.plots_dir, 'Pretraining')
        
        if csv_metrics:
            # Use CSV metrics (most recent data)
            print(f"Best Epoch: {int(csv_metrics.get('epoch', 0))}")
            print(f"Best FID:   {csv_metrics.get('fid', 0.0):.2f}")
            print(f"Baseline KL: {csv_metrics.get('kl_div_total', 0.0):.4f}")
            print(f"Final Loss:  {csv_metrics.get('loss', 0.0):.4f}")
        elif best_pretrain_metrics['fid'] != float('inf'):
            # Fallback to in-memory metrics (current session)
            print(f"Best Epoch: {best_pretrain_metrics['epoch']}")
            print(f"Best FID:   {best_pretrain_metrics['fid']:.2f}")
            print(f"Baseline KL: {best_pretrain_metrics['kl']:.4f}")
            print(f"Final Loss:  {best_pretrain_metrics['loss']:.4f}")
        else:
            print("Metrics not recorded (Pretraining skipped or < 10 epochs run).")
            print("Assuming converged model loaded from checkpoint.")
        print("=" * 60 + "\n")
        
        # Step 2: Run experiments based on mode
        if self.config.mode == "ablation":
            # Step 2: ES Ablations
            print("\nStep 2: ES Ablations...")
            es_configs = list(itertools.product(
                self.config.es_sigma_values,
                self.config.es_lr_values
            ))
            
            for i, (sigma, lr) in enumerate(es_configs):
                print(f"\n  ES Config {i+1}/{len(es_configs)}: sigma={sigma}, lr={lr}")
                # Check if this config is already complete
                final_path = os.path.join(self.models_dir, f'ES_config_{i}_final.pt')
                if os.path.exists(final_path) and self.config.resume_exp_id:
                    print(f"    Config {i} already complete. Skipping.")
                    # Load and evaluate to get metrics
                    cond_ddpm = self._create_conditional_ddpm(pretrain_ddpm)
                    cond_ddpm.load(final_path)
                    result = self._evaluate(cond_ddpm)
                    result['method'] = 'ES'
                    result['sigma'] = sigma
                    result['lr'] = lr
                    self.all_results['ES'].append(result)
                else:
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
                # Check if this config is already complete
                final_path = os.path.join(self.models_dir, f'PPO_config_{i}_final.pt')
                if os.path.exists(final_path) and self.config.resume_exp_id:
                    print(f"    Config {i} already complete. Skipping.")
                    # Load and evaluate to get metrics
                    cond_ddpm = self._create_conditional_ddpm(pretrain_ddpm)
                    cond_ddpm.load(final_path)
                    result = self._evaluate(cond_ddpm)
                    result['method'] = 'PPO'
                    result['kl_weight'] = kl_weight
                    result['ppo_clip'] = ppo_clip
                    result['lr'] = lr
                    self.all_results['PPO'].append(result)
                else:
                    result = self._run_ppo_experiment(pretrain_ddpm, kl_weight, ppo_clip, lr, i)
                    self.all_results['PPO'].append(result)
        else:
            # Single experiment mode
            print(f"\nStep 2: Running Single {self.config.method} Experiment...")
            if self.config.method == "PPO":
                result = self._run_ppo_experiment(
                    pretrain_ddpm, 
                    self.config.single_ppo_kl,
                    self.config.single_ppo_clip,
                    self.config.single_ppo_lr,
                    0  # config_idx
                )
                self.all_results['PPO'].append(result)
            elif self.config.method == "ES":
                result = self._run_es_experiment(
                    pretrain_ddpm,
                    self.config.single_es_sigma,
                    self.config.single_es_lr,
                    0  # config_idx
                )
                self.all_results['ES'].append(result)
        
        # Generate summary (only for ablation mode)
        if self.config.mode == "ablation":
            print("\n" + "=" * 80)
            print("GENERATING SUMMARY")
            print("=" * 80 + "\n")
            self._generate_summary()
            
            # Run NSCB benchmark on best models
            print("\n" + "=" * 80)
            print("RUNNING NSCB BENCHMARK (Final Results)")
            print("=" * 80 + "\n")
            self._run_final_benchmarks()
        
        # [NEW] Generate Interpolation Video
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80)
        
        # Generate interpolation using the best model found
        if self.config.mode == "ablation":
            best_model_path = os.path.join(self.models_dir, f'PPO_config_0_final.pt')
        else:
            # Single mode: use method-specific model
            best_model_path = os.path.join(self.models_dir, f'{self.config.method}_config_0_final.pt')
        
        if os.path.exists(best_model_path):
            viz_model = self._create_conditional_ddpm(pretrain_ddpm) # Re-init architecture
            viz_model.load(best_model_path)
            self.generate_interpolation(viz_model, start_compound='DMSO', end_compound='Taxol')
        else:
            print("Warning: Best model not found. Skipping interpolation visualization.")
        
        total_time = time.time() - start_time
        print(f"\nTotal time: {total_time / 3600:.2f} hours")
        print(f"Results saved to: {self.output_dir}")
        
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.finish()
    
    def _get_gpu_stats(self) -> Dict[str, float]:
        """Get GPU memory usage statistics."""
        stats = {
            'gpu_mem_allocated_mb': 0.0,
            'gpu_mem_reserved_mb': 0.0,
            'gpu_mem_max_mb': 0.0
        }
        
        if torch.cuda.is_available():
            # Current memory allocated to tensors
            stats['gpu_mem_allocated_mb'] = torch.cuda.memory_allocated() / 1024**2
            # Total memory reserved by PyTorch (cached)
            stats['gpu_mem_reserved_mb'] = torch.cuda.memory_reserved() / 1024**2
            # Peak memory usage since last reset
            stats['gpu_mem_max_mb'] = torch.cuda.max_memory_allocated() / 1024**2
            
        return stats
    
    def _pretrain_ddpm(self) -> Tuple[ImageDDPM, Dict]:
        """[FIXED] Conditional Marginal Pretraining with INCREMENTAL Logic."""
        # 1. Initialize Conditional Model
        ddpm = ImageDDPM(
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
        
        # Try to load checkpoint
        start_epoch = self._load_checkpoint_if_exists(ddpm, ddpm.optimizer, "ddpm_pretrain_latest.pt", skip_optimizer=self.config.skip_optimizer_on_resume)
        
        # --- NEW INCREMENTAL LOGIC ---
        # The argument --ddpm-epochs now means "How many epochs to train THIS SESSION"
        epochs_to_train = self.config.ddpm_epochs
        target_epoch = start_epoch + epochs_to_train
        
        # Track best metrics
        best_metrics = {'fid': float('inf'), 'kl': 0.0, 'loss': 0.0, 'epoch': 0}

        if epochs_to_train <= 0:
            print(f"  [Incremental] Epochs set to {epochs_to_train}. Skipping training.")
            return ddpm, best_metrics

        print(f"  [Incremental] Resuming from Epoch {start_epoch}. Training for {epochs_to_train} epochs.")
        print(f"  [Incremental] Target Epoch: {target_epoch}")

        # 2. Use Full Dataset
        dataloader = DataLoader(
            self.train_dataset, 
            batch_size=self.config.ddpm_batch_size,
            shuffle=True,
            num_workers=32 if os.name != 'nt' else 0,
            pin_memory=True
        )

        for epoch in range(start_epoch, target_epoch):
            # Reset peak memory stats at start of epoch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            epoch_losses = []
            
            # 1. Train
            for batch in dataloader:
                images = batch['image'].to(self.config.device)
                fingerprint = batch['fingerprint'].to(self.config.device)
                
                dummy_control = torch.zeros_like(images)
                
                loss = ddpm.train_step(x0=images, control=dummy_control, fingerprint=fingerprint)
                epoch_losses.append(loss)
            
            avg_loss = np.mean(epoch_losses)
            
            # Get GPU stats
            gpu_stats = self._get_gpu_stats()

            # 2. Evaluate (Every 10 epochs)
            if (epoch + 1) % 10 == 0:
                metrics = self._evaluate_pretrain(ddpm, self.train_dataset)
                fid_score = metrics.get('fid', 0.0)
                kl_score = metrics.get('kl_div_total', 0.0)
                mi_score = metrics.get('mutual_information', 0.0)
                num_samples_used = metrics.get('num_eval_samples', self.config.num_eval_samples)
                
                print(f"    Epoch {epoch+1}/{target_epoch} | Loss: {avg_loss:.4f} | FID ({num_samples_used}): {fid_score:.2f} | KL: {kl_score:.4f} | MI: {mi_score:.4f} | GPU: {gpu_stats['gpu_mem_max_mb']:.0f}MB")
                
                # Update Best Metrics
                if fid_score < best_metrics['fid']:
                    best_metrics = {'fid': fid_score, 'kl': kl_score, 'loss': avg_loss, 'epoch': epoch + 1}

                # Save metrics to CSV
                metrics['epoch'] = epoch + 1
                metrics['loss'] = avg_loss
                metrics['phase'] = 'pretraining'
                self._save_metrics_to_csv([metrics], self.plots_dir, 'Pretraining')
            else:
                print(f"    Epoch {epoch+1}/{target_epoch} | Loss: {avg_loss:.4f}")

            # 3. Checkpoint
            self._save_checkpoint(ddpm, ddpm.optimizer, epoch + 1, "ddpm_pretrain_latest.pt", is_global=True)
            
        return ddpm, best_metrics
    
    def _create_conditional_ddpm(self, pretrain_ddpm: ImageDDPM) -> ImageDDPM:
        """
        Create conditional DDPM for coupling phase and initialize from pretrained.
        
        Note: Since pretraining is now conditional (Report Section 4), both models
        have the same architecture, so weight transfer is straightforward.
        """
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
        
        # Weight transfer: Both models are now conditional, so architectures match
        pretrained_state = pretrain_ddpm.model.state_dict()
        current_state = cond_ddpm.model.state_dict()
        
        for key in current_state.keys():
            if key in pretrained_state:
                if pretrained_state[key].shape == current_state[key].shape:
                    current_state[key] = pretrained_state[key].clone()
                else:
                    # Shouldn't happen now that both are conditional, but keep for safety
                    print(f"    Warning: Shape mismatch for {key}, skipping transfer")
        
        cond_ddpm.model.load_state_dict(current_state)
        
        # Also transfer perturbation encoder if available
        if pretrain_ddpm.perturbation_encoder is not None and cond_ddpm.perturbation_encoder is not None:
            cond_ddpm.perturbation_encoder.load_state_dict(
                pretrain_ddpm.perturbation_encoder.state_dict()
            )
        
        return cond_ddpm
    
    def _run_es_experiment(
        self,
        pretrain_ddpm: ImageDDPM,
        sigma: float,
        lr: float,
        config_idx: int,
    ) -> Dict:
        """Run single ES experiment with INCREMENTAL Resume capability."""
        cond_ddpm = self._create_conditional_ddpm(pretrain_ddpm)
        
        checkpoint_name = f'ES_config_{config_idx}_latest.pt'
        # ES doesn't have a standard optimizer in the trainer, so we pass None
        start_epoch = self._load_checkpoint_if_exists(cond_ddpm, None, checkpoint_name, skip_optimizer=self.config.skip_optimizer_on_resume)
        
        # --- INCREMENTAL LOGIC ---
        epochs_to_train = self.config.coupling_epochs
        
        # Logic: If we are still in warmup, finish warmup first, THEN do the requested ES epochs
        # If we are past warmup, just do the requested ES epochs
        
        run_warmup = False
        warmup_start = 0
        
        if start_epoch < self.config.warmup_epochs:
            run_warmup = True
            warmup_start = start_epoch
            # If resuming during warmup, we finish warmup, then run the full requested ES epochs
            es_start_epoch = self.config.warmup_epochs
            es_target_epoch = es_start_epoch + epochs_to_train
            print(f"  [Incremental] Resuming during Warmup (Epoch {start_epoch}). Will finish warmup then run {epochs_to_train} ES epochs.")
        else:
            # Past warmup, just add epochs
            es_start_epoch = start_epoch
            es_target_epoch = start_epoch + epochs_to_train
            print(f"  [Incremental] Resuming ES (Warmup complete). Running {epochs_to_train} additional epochs.")
        
        # Create dataloader
        dataloader = BatchPairedDataLoader(
            self.train_dataset,
            batch_size=self.config.coupling_batch_size,
            shuffle=True,
        )
        
        warmup_metrics = []
        
        # --- WARMUP PHASE ---
        if run_warmup:
            print(f"    Warmup phase: {warmup_start} to {self.config.warmup_epochs} epochs...")
            for warmup_epoch in range(warmup_start, self.config.warmup_epochs):
                # Reset peak memory stats at start of epoch
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                
                warmup_losses = []
                for batch in dataloader:
                    control = batch['control'].to(self.config.device)
                    perturbed = batch['perturbed'].to(self.config.device)
                    fingerprint = batch['fingerprint'].to(self.config.device)
                    
                    loss = cond_ddpm.train_step(perturbed, control, fingerprint)
                    warmup_losses.append(loss)
                
                avg_loss = np.mean(warmup_losses)
                
                # Get GPU stats
                gpu_stats = self._get_gpu_stats()
                
                # Evaluate during warmup
                metrics = self._evaluate(cond_ddpm)
                metrics['epoch'] = warmup_epoch + 1
                metrics['loss'] = avg_loss
                metrics['sigma'] = sigma
                metrics['lr'] = lr
                metrics['phase'] = 'warmup'
                # Add GPU stats to metrics
                metrics.update(gpu_stats)
                warmup_metrics.append(metrics)
                
                # Plot during warmup
                checkpoint_dir = os.path.join(self.plots_dir, f'ES_config_{config_idx}')
                os.makedirs(checkpoint_dir, exist_ok=True)
                self._plot_checkpoint(warmup_metrics, checkpoint_dir, warmup_epoch, 'ES', f'σ={sigma}, lr={lr}')
                
                if (warmup_epoch + 1) % 3 == 0:
                    print(f"      Warmup epoch {warmup_epoch+1}, Loss: {avg_loss:.4f}, FID: {metrics.get('fid', 0):.2f}")
                
                # Save checkpoint during warmup
                current_epoch = warmup_epoch + 1
                self._save_checkpoint(cond_ddpm, None, current_epoch, checkpoint_name)
        
        # Initialize epoch_metrics with warmup metrics
        epoch_metrics = warmup_metrics.copy()
        
        # ES training
        es_trainer = ImageESTrainer(
            cond_ddpm,
            population_size=self.config.es_population_size,
            sigma=sigma,
            lr=lr,
            device=self.config.device,
            use_bio_loss=self.config.enable_bio_loss,  # [ADDED] Pass the flag
        )
        
        print(f"  [Incremental] Starting ES Training: {es_start_epoch} -> {es_target_epoch}")

        for epoch in range(es_start_epoch, es_target_epoch):
            # Reset peak memory stats at start of epoch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            epoch_losses = []
            
            for batch in dataloader:
                control = batch['control'].to(self.config.device)
                perturbed = batch['perturbed'].to(self.config.device)
                fingerprint = batch['fingerprint'].to(self.config.device)
                
                loss = es_trainer.train_step(perturbed, control, fingerprint)
                epoch_losses.append(loss)
            
            avg_loss = np.mean(epoch_losses)
            
            # Get GPU stats
            gpu_stats = self._get_gpu_stats()
            
            metrics = self._evaluate(cond_ddpm)
            metrics['epoch'] = epoch + 1
            metrics['loss'] = avg_loss
            metrics['sigma'] = sigma
            metrics['lr'] = lr
            metrics['phase'] = 'training'  # ES training phase
            # Add GPU stats to metrics
            metrics.update(gpu_stats)
            epoch_metrics.append(metrics)
            
            # Plot checkpoint
            checkpoint_dir = os.path.join(self.plots_dir, f'ES_config_{config_idx}')
            os.makedirs(checkpoint_dir, exist_ok=True)
            self._plot_checkpoint(epoch_metrics, checkpoint_dir, epoch, 'ES', f'σ={sigma}, lr={lr}')

            if (epoch + 1) % 5 == 0:
                num_samples_used = metrics.get('num_eval_samples', self.config.num_eval_samples)
                
                # [PAPER COMPLIANT] Extract key metrics
                fid_cond = metrics.get('fid_conditional', 0.0)  # The competitive metric (Target: 56.8)
                kid = metrics.get('kid', 0.0)
                moa = metrics.get('moa_accuracy', 0.0)
                kl = metrics.get('kl_div_total', 0.0)
                mi = metrics.get('mutual_information', 0.0)

                print(f"    Epoch {epoch+1}: FID(All)={metrics['fid']:.2f} | FID(Cond)={fid_cond:.2f} | KID={kid:.2f} | MoA={moa*100:.1f}% | KL={kl:.4f} | MI={mi:.4f}")
            
            # Log to wandb
            if self.config.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    f'ES/config_{config_idx}/loss': avg_loss,
                    f'ES/config_{config_idx}/fid': metrics['fid'],
                    f'ES/config_{config_idx}/mse': metrics['mse'],
                    f'ES/config_{config_idx}/gpu_max_mb': gpu_stats['gpu_mem_max_mb'],
                })
            
            self._save_checkpoint(cond_ddpm, None, epoch + 1, checkpoint_name)
        
        final_metrics = epoch_metrics[-1] if epoch_metrics else {}
        final_metrics['method'] = 'ES'
        final_metrics['history'] = epoch_metrics
        
        # Generate latent space visualization
        self._plot_latent_clusters(cond_ddpm, 'ES', config_idx)
        
        # Save model for NSCB benchmark
        model_path = os.path.join(self.models_dir, f'ES_config_{config_idx}_final.pt')
        cond_ddpm.save(model_path)
        
        # Run NSCB benchmark
        nscb_results = self.run_nscb_benchmark(cond_ddpm)
        final_metrics.update(nscb_results)
        
        return final_metrics
    
    def _run_ppo_experiment(
        self,
        pretrain_ddpm: ImageDDPM,
        kl_weight: float,
        ppo_clip: float,
        lr: float,
        config_idx: int,
    ) -> Dict:
        """Run single PPO experiment with INCREMENTAL Resume capability."""
        cond_ddpm = self._create_conditional_ddpm(pretrain_ddpm)
        
        checkpoint_name = f'PPO_config_{config_idx}_latest.pt'
        
        ppo_trainer = ImagePPOTrainer(
            cond_ddpm,
            pretrain_ddpm,
            kl_weight=kl_weight,
            ppo_clip=ppo_clip,
            lr=lr,
            device=self.config.device,
            use_bio_loss=self.config.enable_bio_loss,
        )
        
        # Load state
        start_epoch = self._load_checkpoint_if_exists(cond_ddpm, ppo_trainer.optimizer, checkpoint_name, skip_optimizer=self.config.skip_optimizer_on_resume)
        
        # --- NEW INCREMENTAL LOGIC ---
        epochs_to_train = self.config.coupling_epochs
        target_epoch = start_epoch + epochs_to_train
        
        print(f"  [Incremental] Resuming PPO Config {config_idx} from Epoch {start_epoch}")
        print(f"  [Incremental] Training for {epochs_to_train} additional epochs (Target: {target_epoch})")
        
        # Create dataloader
        dataloader = BatchPairedDataLoader(
            self.train_dataset,
            batch_size=self.config.coupling_batch_size,
            shuffle=True,
        )
        
        epoch_metrics = []
        
        for epoch in range(start_epoch, target_epoch):
            # Reset peak memory stats at start of epoch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            epoch_losses = []
            
            for batch in dataloader:
                control = batch['control'].to(self.config.device)
                perturbed = batch['perturbed'].to(self.config.device)
                fingerprint = batch['fingerprint'].to(self.config.device)
                
                loss = ppo_trainer.train_step(perturbed, control, fingerprint)
                epoch_losses.append(loss)
            
            avg_loss = np.mean(epoch_losses)
            
            # Get GPU stats
            gpu_stats = self._get_gpu_stats()
            
            # Evaluate
            metrics = self._evaluate(cond_ddpm)
            metrics['epoch'] = epoch + 1
            metrics['loss'] = avg_loss
            metrics['kl_weight'] = kl_weight
            metrics['ppo_clip'] = ppo_clip
            metrics['lr'] = lr
            metrics['phase'] = 'training'  # PPO training phase
            # Add GPU stats to metrics
            metrics.update(gpu_stats)
            epoch_metrics.append(metrics)
            
            # Plot checkpoint
            checkpoint_dir = os.path.join(self.plots_dir, f'PPO_config_{config_idx}')
            os.makedirs(checkpoint_dir, exist_ok=True)
            self._plot_checkpoint(epoch_metrics, checkpoint_dir, epoch, 'PPO', f'KL={kl_weight}, lr={lr}')

            if (epoch + 1) % 5 == 0:
                num_samples_used = metrics.get('num_eval_samples', self.config.num_eval_samples)
                
                # [PAPER COMPLIANT] Extract key metrics
                fid_cond = metrics.get('fid_conditional', 0.0)  # The competitive metric (Target: 56.8)
                kid = metrics.get('kid', 0.0)
                moa = metrics.get('moa_accuracy', 0.0)
                kl = metrics.get('kl_div_total', 0.0)
                mi = metrics.get('mutual_information', 0.0)

                print(f"    Epoch {epoch+1}: Loss={avg_loss:.4f} | FID(All)={metrics['fid']:.2f} | FID(Cond)={fid_cond:.2f} | KID={kid:.2f} | MoA={moa*100:.1f}% | KL={kl:.4f} | MI={mi:.4f}")
            
            # Log to wandb
            if self.config.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    f'PPO/config_{config_idx}/loss': avg_loss,
                    f'PPO/config_{config_idx}/fid': metrics['fid'],
                    f'PPO/config_{config_idx}/mse': metrics['mse'],
                    f'PPO/config_{config_idx}/gpu_max_mb': gpu_stats['gpu_mem_max_mb'],
                })
            
            # Save Checkpoint
            self._save_checkpoint(cond_ddpm, ppo_trainer.optimizer, epoch + 1, checkpoint_name)
        
        final_metrics = epoch_metrics[-1] if epoch_metrics else {}
        final_metrics['method'] = 'PPO'
        final_metrics['history'] = epoch_metrics
        
        # Visuals & Benchmarks
        self._plot_latent_clusters(cond_ddpm, 'PPO', config_idx)
        model_path = os.path.join(self.models_dir, f'PPO_config_{config_idx}_final.pt')
        cond_ddpm.save(model_path)
        nscb_results = self.run_nscb_benchmark(cond_ddpm)
        final_metrics.update(nscb_results)
        
        return final_metrics
    
    def _nscb_benchmark(self, cond_ddpm: ImageDDPM, test_batch_name: str = None) -> Dict:
        """
        [FIXED] NSCB Evaluation using Deep Features (Paper Compliant).
        Tests on a 'Hold-out Batch' to measure Batch-Effect correction.
        """
        print("\n=== Running NSCB Benchmark (Batch Generalization) ===")
        cond_ddpm.model.eval()
        metrics_engine = ImageMetrics(device=self.config.device)
        
        # Load Test Data
        test_dataset = BBBC021Dataset(
            self.config.data_dir, 
            self.config.metadata_file, 
            self.config.image_size, 
            split="test"
        )
        
        if len(test_dataset) == 0:
             print("Warning: No test data found. Skipping NSCB.")
             return {}

        test_loader = BatchPairedDataLoader(
            test_dataset, 
            batch_size=self.config.coupling_batch_size, 
            shuffle=False
        )
        
        all_real = []
        all_fake = []
        all_moas = []
        
        # 1. Generate images
        with torch.no_grad():
            for batch in test_loader:
                control = batch['control'].to(self.config.device)
                perturbed = batch['perturbed'].to(self.config.device)
                fingerprint = batch['fingerprint'].to(self.config.device)
                
                # Generate
                gen = cond_ddpm.sample(len(control), control, fingerprint, num_steps=self.config.num_sampling_steps)
                
                all_real.append(perturbed.cpu().numpy())
                all_fake.append(gen.cpu().numpy())
                
                if 'moa_idx' in batch:
                    all_moas.extend(batch['moa_idx'].cpu().tolist())
        
        real_imgs = np.concatenate(all_real, axis=0)
        fake_imgs = np.concatenate(all_fake, axis=0)
        moas = np.array(all_moas)
        
        # 2. Extract Deep Features (InceptionV3)
        # This matches the SOTA standard. Old pixel-stats are too weak.
        print("  Extracting Deep Features for Final MoA Check...")
        real_feats = metrics_engine.get_features(real_imgs)
        fake_feats = metrics_engine.get_features(fake_imgs)
        
        # 3. Compute Deep MoA Accuracy
        moa_acc = 0.0
        if SKLEARN_AVAILABLE and len(moas) > 0 and real_feats is not None:
            moa_acc = metrics_engine.compute_deep_moa_accuracy(
                real_feats.numpy(), 
                fake_feats.numpy(), 
                moas
            )
            
        # 4. Compute Profile Similarity (Pixel-based is fine for this specific metric)
        info_metrics = ApproximateMetrics.compute_all(real_imgs, fake_imgs)
        
        print(f"  NSCB Result -> MoA Accuracy (Deep): {moa_acc*100:.2f}%")
        print(f"  NSCB Result -> Profile Similarity:  {info_metrics.get('profile_similarity', 0):.4f}")
        
        return {
            'nscb_moa_accuracy': moa_acc,
            'nscb_profile_similarity': info_metrics.get('profile_similarity', 0),
            'nscb_num_samples': len(real_imgs)
        }
    
    def _evaluate(self, cond_ddpm: ImageDDPM) -> Dict:
        """
        [PAPER COMPLIANT] Evaluation Routine.
        Calculates:
        1. Overall FID & KID (Global Distribution)
        2. Conditional FID (Per-compound Average) -> The key metric to beat (56.8)
        3. Deep MoA Accuracy (Using Inception Features) -> Target > 70%
        """
        cond_ddpm.model.eval()
        metrics_engine = ImageMetrics(device=self.config.device)
        
        # 1. Generate Data
        all_real = []
        all_fake = []
        all_compounds = []
        all_moas = []
        
        val_loader = BatchPairedDataLoader(
            self.val_dataset,
            batch_size=self.config.coupling_batch_size,
            shuffle=False
        )
        
        # Limit evaluation size for speed during training (full size for final bench)
        max_samples = self.config.num_eval_samples
        if len(self.val_dataset) < max_samples:
            max_samples = len(self.val_dataset)
        
        num_samples = 0
        
        print("  Generating samples for Full Paper Evaluation...")
        with torch.no_grad():
            for batch in val_loader:
                if num_samples >= max_samples: 
                    break
                
                control = batch['control'].to(self.config.device)
                perturbed = batch['perturbed'].to(self.config.device)  # Real Target
                fingerprint = batch['fingerprint'].to(self.config.device)
                
                # Generate
                generated = cond_ddpm.sample(
                    len(control), control, fingerprint, 
                    num_steps=self.config.num_sampling_steps
                )
                
                # Store (CPU)
                all_real.append(perturbed.cpu().numpy())
                all_fake.append(generated.cpu().numpy())
                if 'compound' in batch:
                    all_compounds.extend(batch['compound'])
                if 'moa_idx' in batch:
                    all_moas.extend(batch['moa_idx'].cpu().numpy().tolist())
                
                num_samples += len(control)

        # Concatenate
        real_imgs = np.concatenate(all_real, axis=0)[:max_samples]
        fake_imgs = np.concatenate(all_fake, axis=0)[:max_samples]
        compounds = np.array(all_compounds[:max_samples])
        moas = np.array(all_moas[:max_samples]) if len(all_moas) > 0 else np.array([])
        
        # --- 2. Extract Features (ONCE) ---
        # We extract 2048-dim features once and use them for FID, FIDc, and MoA
        print("  Extracting Inception features...")
        real_feats = metrics_engine.get_features(real_imgs)
        fake_feats = metrics_engine.get_features(fake_imgs)
        
        if real_feats is None or fake_feats is None:
            print("  Warning: Inception features not available, falling back to basic metrics")
            # Fallback to basic metrics
            mse = ImageMetrics.compute_mse(real_imgs, fake_imgs)
            return {
                'fid': 0.0,
                'fid_conditional': 0.0,
                'kid': 0.0,
                'moa_accuracy': 0.0,
                'mse': mse,
                'num_eval_samples': len(real_imgs)
            }
        
        # --- 3. Compute Metrics ---
        
        # A. Overall Metrics
        fid_overall = metrics_engine.compute_fid_from_features(real_feats, fake_feats)
        kid_overall = ImageMetrics.compute_kid(real_imgs, fake_imgs, device=self.config.device)
        mse = ImageMetrics.compute_mse(real_imgs, fake_imgs)
        ssim = ImageMetrics.compute_ssim(real_imgs, fake_imgs)
        
        # B. Conditional FID (FID_c) - The CellFlux Benchmark
        fid_conditional = 0.0
        if len(compounds) > 0 and len(np.unique(compounds)) > 1:
            unique_cmps = np.unique(compounds)
            fids_c = []
            for cmp in unique_cmps:
                # Get indices for this compound
                idxs = np.where(compounds == cmp)[0]
                if len(idxs) < 5: 
                    continue  # Need minimum samples for stats
                
                # Compute FID just for this slice using pre-computed features
                r_sub = real_feats[idxs]
                f_sub = fake_feats[idxs]
                try:
                    fid_val = metrics_engine.compute_fid_from_features(r_sub, f_sub)
                    fids_c.append(fid_val)
                except Exception as e:
                    # Skip compounds that fail (e.g., too few samples for covariance)
                    pass
            
            fid_conditional = np.mean(fids_c) if fids_c else 0.0
            if len(fids_c) > 0:
                print(f"    Conditional FID: {fid_conditional:.2f} (avg over {len(fids_c)} compounds)")
        
        # C. Deep MoA Accuracy (Using Inception Features)
        moa_acc = 0.0
        if SKLEARN_AVAILABLE and len(moas) > 0:
            moa_acc = ImageMetrics.compute_deep_moa_accuracy(
                real_feats.numpy(), 
                fake_feats.numpy(), 
                moas
            )
        
        # D. Additional metrics
        pixel_change = ImageMetrics.compute_pixel_change(real_imgs, fake_imgs)  # Approximate
        info_metrics = ApproximateMetrics.compute_all(real_imgs, fake_imgs)

        metrics = {
            'fid': fid_overall,                    # Overall FID (Target: 18.7)
            'fid_conditional': fid_conditional,     # Conditional FID (Target: 56.8) <--- COMPARE THIS
            'kid': kid_overall,                    # KID (Target: 1.62)
            'moa_accuracy': moa_acc,               # MoA Accuracy (Target: > 70%) <--- COMPARE THIS
            'pixel_change': pixel_change,
            'mse': mse,
            'ssim': ssim,
            'num_eval_samples': len(real_imgs)
        }
        metrics.update(info_metrics)
        
        return metrics
    
    def _evaluate_pretrain(self, ddpm: ImageDDPM, dataset, max_samples: int = None) -> Dict:
        """
        [FIXED] Evaluate pretrained conditional DDPM.
        Uses exactly max_samples if provided, otherwise defaults to num_eval_samples (5000).
        Supports dynamic evaluation (QUICK: 500, FULL: 5000).
        """
        ddpm.model.eval()
        
        # 1. Use the config value (5000) if no specific limit is passed
        if max_samples is None:
            max_samples = self.config.num_eval_samples
        
        # 2. Enforce exactly max_samples (fail if dataset is too small)
        if len(dataset) < max_samples:
            raise ValueError(
                f"Dataset has only {len(dataset)} samples, but evaluation requires {max_samples} samples. "
                f"Please ensure the dataset has at least {max_samples} samples."
            )
        
        num_samples = max_samples  # Use exactly this many, no less
        real_images = []
        fake_images = []
        
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config.fid_batch_size,  # Use config batch size for safety
            shuffle=False,
            num_workers=16 if os.name != 'nt' else 0,
            pin_memory=True,
        )
        
        with torch.no_grad():
            for batch in dataloader:
                if len(real_images) >= num_samples:
                    break
                images = batch['image'].to(self.config.device)
                fingerprints = batch['fingerprint'].to(self.config.device)
                real_images.append(images.cpu().numpy())
                
                if ddpm.conditional:
                    dummy_control = torch.zeros_like(images)
                    generated = ddpm.sample(
                        len(images), 
                        control=dummy_control,
                        fingerprint=fingerprints,
                        num_steps=self.config.num_sampling_steps
                    )
                    fake_images.append(generated.cpu().numpy())
        
        # Flatten lists and truncate to exact num_samples
        real_images = np.concatenate(real_images, axis=0)[:num_samples]
        fake_images = np.concatenate(fake_images, axis=0)[:num_samples]
        
        # Verify we got exactly the right number
        actual_samples = len(real_images)
        if actual_samples != num_samples:
            print(f"WARNING: Expected {num_samples} samples but got {actual_samples}")
        
        # 3. Calculate FID using the new device-aware method
        fid = ImageMetrics.compute_fid(real_images, fake_images, device=self.config.device)
        
        # Calculate fast metrics
        mse = ImageMetrics.compute_mse(real_images, fake_images)
        mae = ImageMetrics.compute_mae(real_images, fake_images)
        ssim = ImageMetrics.compute_ssim(real_images, fake_images)
        info_metrics = ApproximateMetrics.compute_all(real_images, fake_images)
        
        metrics = {'fid': fid, 'mse': mse, 'mae': mae, 'ssim': ssim, 'num_eval_samples': actual_samples}
        metrics.update(info_metrics)
        
        return metrics
    
    def _load_latest_metrics_from_csv(self, checkpoint_dir: str, method: str) -> Optional[Dict]:
        """
        Load the latest metrics from the most recent CSV file.
        Returns the row with the best (lowest) FID score, or None if no CSV found.
        """
        # Find all CSV files for this method
        base_pattern = f"{method}_metrics"
        csv_files = []
        
        # Check for base file
        base_path = os.path.join(checkpoint_dir, f"{base_pattern}.csv")
        if os.path.exists(base_path):
            csv_files.append((0, base_path))  # 0 means base file (oldest)
        
        # Check for versioned files
        counter = 1
        while True:
            versioned_path = os.path.join(checkpoint_dir, f"{base_pattern}_{counter}.csv")
            if os.path.exists(versioned_path):
                csv_files.append((counter, versioned_path))
                counter += 1
            else:
                break
        
        if not csv_files:
            return None
        
        # Get the latest file (highest counter, or base if no versions)
        csv_files.sort(key=lambda x: x[0], reverse=True)
        latest_file = csv_files[0][1]
        
        # Read the CSV and find the best FID
        try:
            with open(latest_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                if not rows:
                    return None
                
                # Find row with best (lowest) FID
                best_row = None
                best_fid = float('inf')
                
                for row in rows:
                    try:
                        fid = float(row.get('fid', float('inf')))
                        if fid < best_fid:
                            best_fid = fid
                            best_row = row
                    except (ValueError, TypeError):
                        continue
                
                if best_row:
                    # Convert string values to appropriate types
                    result = {}
                    for key, value in best_row.items():
                        try:
                            # Try to convert to float first
                            result[key] = float(value)
                        except (ValueError, TypeError):
                            try:
                                # Try int
                                result[key] = int(value)
                            except (ValueError, TypeError):
                                # Keep as string
                                result[key] = value
                    
                    return result
        except Exception as e:
            print(f"Warning: Failed to read metrics from CSV {latest_file}: {e}")
        
        return None
    
    def _save_metrics_to_csv(self, metrics: List[Dict], checkpoint_dir: str, method: str, config_str: str = ""):
        """
        Save metrics to CSV with smart versioning. 
        Creates a NEW file for every new run session (e.g. _1.csv, _2.csv)
        to prevent mixing data or overwriting old runs.
        """
        if not metrics:
            return
        
        # 1. Session Key (Unique per stage/method in this folder)
        session_key = os.path.join(checkpoint_dir, method)

        # 2. Determine Filename (ONCE per runtime session)
        if session_key not in self._session_csv_paths:
            base_name = f"{method}_metrics.csv"
            file_path = os.path.join(checkpoint_dir, base_name)
            
            # If default file exists, increment counter until we find a free name
            if os.path.exists(file_path):
                counter = 1
                while True:
                    new_name = f"{method}_metrics_{counter}.csv"
                    new_path = os.path.join(checkpoint_dir, new_name)
                    if not os.path.exists(new_path):
                        file_path = new_path
                        print(f"  [Log] Creating NEW metrics file for this session: {os.path.basename(file_path)}")
                        break
                    counter += 1
            
            # Cache the path so we use the SAME file for the rest of this run
            self._session_csv_paths[session_key] = file_path

        final_path = self._session_csv_paths[session_key]
        
        # 3. Write Data (Append Mode)
        # We append because we might call this multiple times in one session (e.g. per epoch)
        file_exists = os.path.exists(final_path)
        mode = 'a' if file_exists else 'w'
        
        # Ensure 'epoch' is the first column if present
        keys = list(metrics[0].keys())
        if 'epoch' in keys:
            keys.remove('epoch')
            keys.insert(0, 'epoch')
            
        try:
            with open(final_path, mode, newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                if not file_exists:
                    writer.writeheader()
                writer.writerows(metrics)
        except Exception as e:
            print(f"Warning: Failed to save metrics to CSV: {e}")
    
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
  Profile Sim: {latest.get('profile_similarity', 0):.4f}

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
    
    def run_nscb_benchmark(self, cond_ddpm: ImageDDPM, test_batch_name: str = None) -> Dict:
        """
        Runs the specific NSCB (Not-Same-Compound-or-Batch) evaluation required by the paper.
        Tests on hold-out batches (e.g., Weeks 9-10) to measure batch generalization.
        
        Args:
            cond_ddpm: Trained conditional DDPM model
            test_batch_name: Specific batch to use for testing (e.g., 'Week10')
                           If None, uses test split from dataset
        
        Returns:
            Dictionary with NSCB metrics
        """
        print("\n=== Running NSCB Benchmark (Batch Generalization) ===")
        
        # Load Test Data
        test_dataset = BBBC021Dataset(
            self.config.data_dir, 
            self.config.metadata_file, 
            self.config.image_size, 
            split="test"
        )
        
        test_loader = BatchPairedDataLoader(
            test_dataset, 
            batch_size=self.config.coupling_batch_size, 
            shuffle=False
        )
        
        real_feats = []
        fake_feats = []
        moa_labels = []
        
        cond_ddpm.model.eval()
        
        with torch.no_grad():
            for batch in test_loader:
                control = batch['control'].to(self.config.device)
                fingerprint = batch['fingerprint'].to(self.config.device)
                
                # Get real images
                real_img = batch['perturbed'].cpu().numpy()
                
                # Generate
                gen = cond_ddpm.sample(
                    len(control), control, fingerprint,
                    num_steps=self.config.num_sampling_steps,
                )
                fake_img = gen.cpu().numpy()
                
                # Extract Bio Features (CellFlux methodology)
                real_feats.append(ApproximateMetrics._get_bio_features(real_img))
                fake_feats.append(ApproximateMetrics._get_bio_features(fake_img))
                
                # Get MoA labels
                if 'moa_idx' in batch:
                    moa_labels.extend(batch['moa_idx'].cpu().numpy().tolist())
        
        if not real_feats or not fake_feats:
            print("Warning: No data collected for NSCB benchmark")
            return {}
        
        real_feats = np.vstack(real_feats)
        fake_feats = np.vstack(fake_feats)
        
        # Compute MoA Accuracy using biological features
        if SKLEARN_AVAILABLE and len(moa_labels) > 0:
            moa_accuracy = ImageMetrics.compute_moa_accuracy(
                real_feats, fake_feats, np.array(moa_labels)
            )
            print(f"NSCB MoA Accuracy: {moa_accuracy*100:.2f}%")
        else:
            moa_accuracy = 0.0
            if not SKLEARN_AVAILABLE:
                print("Warning: sklearn not available, skipping MoA accuracy")
        
        # Compute profile similarity
        dot = np.sum(real_feats * fake_feats, axis=1)
        norm_r = np.linalg.norm(real_feats, axis=1)
        norm_f = np.linalg.norm(fake_feats, axis=1)
        profile_sim = np.mean(dot / (norm_r * norm_f + 1e-8))
        
        nscb_metrics = {
            'nscb_moa_accuracy': float(moa_accuracy),
            'nscb_profile_similarity': float(profile_sim),
            'nscb_num_samples': len(real_feats),
        }
        
        print(f"NSCB Profile Similarity: {profile_sim:.4f}")
        print("NSCB Benchmark Complete.")
        
        return nscb_metrics
    
    def _run_final_benchmarks(self):
        """Run NSCB benchmarks on best ES and PPO models."""
        if not self.all_results['ES'] or not self.all_results['PPO']:
            print("Warning: No results available for final benchmarks")
            return
        
        # Get best models
        best_es = min(self.all_results['ES'], key=lambda x: x['fid'])
        best_ppo = min(self.all_results['PPO'], key=lambda x: x['fid'])
        
        print("Best ES and PPO configurations identified.")
        print("Note: Full NSCB benchmark requires trained models to be saved and loaded.")
        print("The run_nscb_benchmark() method is available for use with saved models.")
        
        # Save NSCB benchmark instructions
        nscb_path = os.path.join(self.output_dir, "NSCB_BENCHMARK_INSTRUCTIONS.txt")
        with open(nscb_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("NSCB BENCHMARK INSTRUCTIONS\n")
            f.write("=" * 80 + "\n\n")
            f.write("To run the full NSCB (Not-Same-Compound-or-Batch) benchmark:\n\n")
            f.write("1. Ensure models are saved during training\n")
            f.write("2. Load the best ES and PPO models\n")
            f.write("3. Call runner.run_nscb_benchmark(model) for each model\n")
            f.write("4. Compare NSCB metrics between ES and PPO\n\n")
            f.write("Best ES Config: sigma={}, lr={}, FID={:.4f}\n".format(
                best_es.get('sigma', 'N/A'), best_es.get('lr', 'N/A'), best_es.get('fid', 0)))
            f.write("Best PPO Config: kl_weight={}, lr={}, FID={:.4f}\n".format(
                best_ppo.get('kl_weight', 'N/A'), best_ppo.get('lr', 'N/A'), best_ppo.get('fid', 0)))
        
        print(f"NSCB benchmark instructions saved to: {nscb_path}")
    
    def _plot_latent_clusters(self, cond_ddpm: ImageDDPM, method: str, config_idx: int):
        """
        Generate latent space visualization (UMAP/PCA) for biological feature alignment.
        
        This proves the ES vs PPO optimization actually moved the cells into the 
        correct biological cluster.
        """
        try:
            from sklearn.decomposition import PCA
            try:
                import umap
                USE_UMAP = True
            except ImportError:
                USE_UMAP = False
                print("  UMAP not available, using PCA for latent visualization")
        except ImportError:
            print("  sklearn not available, skipping latent space visualization")
            return
        
        cond_ddpm.model.eval()
        
        # Get sample images
        val_loader = BatchPairedDataLoader(
            self.val_dataset,
            batch_size=self.config.coupling_batch_size,
            shuffle=False,
        )
        
        real_features = []
        fake_features = []
        moa_labels = []
        
        num_samples = min(200, self.config.num_eval_samples)  # Limit for visualization
        
        with torch.no_grad():
            count = 0
            for batch in val_loader:
                if count >= num_samples:
                    break
                
                control = batch['control'].to(self.config.device)
                perturbed = batch['perturbed'].to(self.config.device)
                fingerprint = batch['fingerprint'].to(self.config.device)
                
                # Get real features (flatten images)
                real_feat = perturbed.cpu().numpy().reshape(len(perturbed), -1)
                real_features.append(real_feat)
                
                # Generate and get fake features
                generated = cond_ddpm.sample(
                    len(control), control, fingerprint,
                    num_steps=self.config.num_sampling_steps,
                )
                fake_feat = generated.cpu().numpy().reshape(len(generated), -1)
                fake_features.append(fake_feat)
                
                # Get MoA labels
                for idx in batch.get('moa_idx', []):
                    moa_labels.append(int(idx))
                
                count += len(control)
        
        if not real_features or not fake_features:
            return
        
        real_features = np.concatenate(real_features, axis=0)[:num_samples]
        fake_features = np.concatenate(fake_features, axis=0)[:num_samples]
        moa_labels = moa_labels[:num_samples]
        
        # Combine for dimensionality reduction
        combined_features = np.vstack([real_features, fake_features])
        
        # Project to 2D
        if USE_UMAP:
            reducer = umap.UMAP(n_components=2, random_state=42)
            proj = reducer.fit_transform(combined_features)
        else:
            pca = PCA(n_components=2, random_state=42)
            proj = pca.fit_transform(combined_features)
        
        # Split back
        proj_real = proj[:len(real_features)]
        proj_fake = proj[len(real_features):]
        
        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Color by MoA if available
        if len(set(moa_labels)) > 1:
            import matplotlib.cm as cm
            colors = cm.get_cmap('tab20')
            for moa in set(moa_labels):
                mask_real = [i for i, m in enumerate(moa_labels) if m == moa]
                mask_fake = [i for i, m in enumerate(moa_labels) if m == moa]
                if mask_real:
                    ax.scatter(proj_real[mask_real, 0], proj_real[mask_real, 1], 
                             c=[colors(moa / max(moa_labels))], alpha=0.6, 
                             marker='o', s=30, label=f'Real MoA{moa}')
                if mask_fake:
                    ax.scatter(proj_fake[mask_fake, 0], proj_fake[mask_fake, 1], 
                             c=[colors(moa / max(moa_labels))], alpha=0.6, 
                             marker='^', s=30, label=f'Generated MoA{moa}')
        else:
            ax.scatter(proj_real[:, 0], proj_real[:, 1], alpha=0.6, 
                      marker='o', s=30, label='Real', c='blue')
            ax.scatter(proj_fake[:, 0], proj_fake[:, 1], alpha=0.6, 
                      marker='^', s=30, label='Generated', c='red')
        
        ax.set_xlabel('Component 1', fontsize=12)
        ax.set_ylabel('Component 2', fontsize=12)
        ax.set_title(f'Biological Feature Alignment - {method}', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        checkpoint_dir = os.path.join(self.plots_dir, f'{method}_config_{config_idx}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        plot_path = os.path.join(checkpoint_dir, 'latent_space.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Latent space visualization saved to: {plot_path}")

    def generate_interpolation(self, model, start_compound='DMSO', end_compound='Taxol', steps=8):
        """
        Generates a latent interpolation video/plot between two compounds.
        """
        print(f"\n[Viz] Generating interpolation: {start_compound} -> {end_compound}")
        model.model.eval()
        
        # 1. Get SMILES strings from metadata
        start_smiles = None
        end_smiles = None
        for m in self.train_dataset.metadata:
            if m['compound'] == start_compound and m.get('smiles'):
                start_smiles = m['smiles']
            if m['compound'] == end_compound and m.get('smiles'):
                end_smiles = m['smiles']
        
        # Fallback: use compound name if SMILES not found (for DMSO, etc.)
        if not start_smiles:
            start_smiles = start_compound
        if not end_smiles:
            end_smiles = end_compound
        
        # Get Latent Vectors (Fingerprints)
        fp_start = torch.tensor(self.train_dataset.morgan_encoder.encode(start_smiles)).to(self.config.device)
        fp_end = torch.tensor(self.train_dataset.morgan_encoder.encode(end_smiles)).to(self.config.device)
        
        # 2. Get a real control image (Source)
        # Find first start_compound image in validation set
        ctrl_indices = [i for i, m in enumerate(self.val_dataset.metadata) if m['compound'] == start_compound]
        if not ctrl_indices: 
            print(f"Warning: Could not find {start_compound} in validation set. Skipping interpolation.")
            return # Skip if not found
        
        control_img = self.val_dataset[ctrl_indices[0]]['image'].unsqueeze(0).to(self.config.device)
        
        # 3. Interpolate and Generate
        interpolation_grid = []
        with torch.no_grad():
            for alpha in np.linspace(0, 1, steps):
                # Linear Interpolation in Latent Space (slerp is better, but lerp works for binary FPs)
                fp_interp = (1 - alpha) * fp_start + alpha * fp_end
                
                # Sample with the mixed embedding
                gen = model.sample(1, control_img, fp_interp.unsqueeze(0), num_steps=50)
                
                # Post-process for display ([-1, 1] -> [0, 1])
                img = gen.squeeze().cpu().permute(1, 2, 0).numpy()
                img = (img + 1.0) / 2.0
                img = np.clip(img, 0, 1)
                interpolation_grid.append(img)

        # 4. Save Plot
        fig, axes = plt.subplots(1, steps, figsize=(20, 3))
        for i, ax in enumerate(axes):
            ax.imshow(interpolation_grid[i])
            ax.axis('off')
            if i == 0: ax.set_title(start_compound, fontsize=10)
            if i == steps - 1: ax.set_title(end_compound, fontsize=10)
            
        save_path = os.path.join(self.plots_dir, f"interp_{start_compound}_to_{end_compound}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Interpolation saved to {save_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="BBBC021 Unified Runner: Ablation & Single Experiment with Resume",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mode Selection
    parser.add_argument("--mode", type=str, default="ablation", choices=["ablation", "single"],
                       help="Run mode: 'ablation' (loop through params) or 'single' (one config)")
    parser.add_argument("--method", type=str, default="PPO", choices=["PPO", "ES"],
                       help="Training method (for single mode)")
    parser.add_argument("--resume-id", type=str, default=None,
                       help="Experiment ID to resume (folder name suffix, e.g., '20241223_120000')")
    
    # Global Pretrain Flags
    parser.add_argument("--force-pretrain", action="store_true",
                       help="Ignore global model, train fresh pretrained model")
    parser.add_argument("--resume-pretrain", action="store_true",
                       help="Resume training the global base model")
    parser.add_argument("--skip-optimizer", action="store_true",
                       help="Skip loading optimizer state when resuming (use new learning rate)")
    
    # Data
    parser.add_argument("--data-dir", type=str, default="./data/bbbc021_all",
                       help="Directory containing BBBC021 data")
    parser.add_argument("--metadata-file", type=str, default="metadata/bbbc021_df_all.csv",
                       help="Metadata CSV file")
    
    # Training
    parser.add_argument("--ddpm-epochs", type=int, default=500,
                       help="DDPM pretraining epochs (recommended: 500-1000 for convergence)")
    parser.add_argument("--coupling-epochs", type=int, default=30,
                       help="Coupling training epochs")
    parser.add_argument("--warmup-epochs", type=int, default=10,
                       help="Warmup epochs before ES")
    parser.add_argument("--ddpm-batch-size", type=int, default=512,
                       help="DDPM batch size (reduced for gradient stability)")
    parser.add_argument("--ddpm-lr", type=float, default=1e-4,
                       help="DDPM learning rate (reduced for better convergence)")
    parser.add_argument("--coupling-batch-size", type=int, default=512,
                       help="Coupling batch size")
    
    # ES ablation
    parser.add_argument("--es-sigma-values", type=float, nargs='+',
                       default=[0.001, 0.005, 0.01],
                       help="ES sigma values")
    parser.add_argument("--es-lr-values", type=float, nargs='+',
                       default=[0.0001, 0.0005, 0.001],
                       help="ES learning rate values")
    parser.add_argument("--es-population-size", type=int, default=50,
                       help="Population size for Evolution Strategies (Higher=More Stable)")
    
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
    
    # Single Experiment Params (Only used if mode='single')
    parser.add_argument("--single-ppo-kl", type=float, default=0.5,
                       help="PPO KL weight for single experiment")
    parser.add_argument("--single-ppo-clip", type=float, default=0.1,
                       help="PPO clip value for single experiment")
    parser.add_argument("--single-ppo-lr", type=float, default=5e-5,
                       help="PPO learning rate for single experiment")
    parser.add_argument("--single-es-sigma", type=float, default=0.005,
                       help="ES sigma for single experiment")
    parser.add_argument("--single-es-lr", type=float, default=0.0005,
                       help="ES learning rate for single experiment")
    
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
    
    # Biological constraints
    parser.add_argument("--enable-bio-loss", action="store_true",
                       help="Enable DNA preservation loss in PPO phase (CellFlux methodology)")
    
    args = parser.parse_args()
    
    config = BBBC021Config(
        mode=args.mode,
        method=args.method,
        resume_exp_id=args.resume_id,
        resume_pretrain=args.resume_pretrain,
        force_pretrain=args.force_pretrain,
        skip_optimizer_on_resume=args.skip_optimizer,
        data_dir=args.data_dir,
        metadata_file=args.metadata_file,
        ddpm_epochs=args.ddpm_epochs,
        coupling_epochs=args.coupling_epochs,
        warmup_epochs=args.warmup_epochs,
        ddpm_batch_size=args.ddpm_batch_size,
        ddpm_lr=args.ddpm_lr,
        coupling_batch_size=args.coupling_batch_size,
        es_sigma_values=args.es_sigma_values,
        es_lr_values=args.es_lr_values,
        es_population_size=args.es_population_size,
        ppo_kl_weight_values=args.ppo_kl_values,
        ppo_clip_values=args.ppo_clip_values,
        ppo_lr_values=args.ppo_lr_values,
        single_ppo_kl=args.single_ppo_kl,
        single_ppo_clip=args.single_ppo_clip,
        single_ppo_lr=args.single_ppo_lr,
        single_es_sigma=args.single_es_sigma,
        single_es_lr=args.single_es_lr,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        seed=args.seed,
        reuse_pretrained=not args.no_reuse_pretrained,
        enable_bio_loss=args.enable_bio_loss,
    )
    
    runner = BBBC021AblationRunner(config)
    runner.run()


if __name__ == "__main__":
    main()