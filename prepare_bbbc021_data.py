"""
================================================================================
BBBC021 Data Preparation Script
================================================================================

Downloads, preprocesses, and organizes the BBBC021 dataset for the ablation study.

BBBC021 Dataset:
- 97,504 images of MCF-7 breast cancer cells
- 113 chemical compounds at 8 concentrations
- 3 channels: DAPI (DNA), Phalloidin (F-actin), β-tubulin
- Multiple plates/batches with control (DMSO) wells

Data source: Broad Bioimage Benchmark Collection
https://bbbc.broadinstitute.org/BBBC021

================================================================================
"""

import os
import sys
import csv
import gzip
import shutil
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
import json

import numpy as np
from PIL import Image
import tifffile

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Install tqdm for progress bars: pip install tqdm")


# ============================================================================
# CONSTANTS
# ============================================================================

BBBC021_BASE_URL = "https://data.broadinstitute.org/bbbc/BBBC021/"

# Compound to Mode of Action mapping
COMPOUND_TO_MOA = {
    'DMSO': 'Control',
    'Taxol': 'Microtubule_stabilizers',
    'Docetaxel': 'Microtubule_stabilizers',
    'Epothilone_B': 'Microtubule_stabilizers',
    'Colchicine': 'Microtubule_destabilizers',
    'Demecolcine': 'Microtubule_destabilizers',
    'Nocodazole': 'Microtubule_destabilizers',
    'Vincristine': 'Microtubule_destabilizers',
    'Cytochalasin_B': 'Actin_disruptors',
    'Cytochalasin_D': 'Actin_disruptors',
    'Latrunculin_B': 'Actin_disruptors',
    'AZ258': 'Aurora_kinase_inhibitors',
    'AZ841': 'Aurora_kinase_inhibitors',
    'Mevinolin': 'Cholesterol-lowering',
    'Simvastatin': 'Cholesterol-lowering',
    'Lovastatin': 'Cholesterol-lowering',
    'Chlorambucil': 'DNA_damage',
    'Cisplatin': 'DNA_damage',
    'Etoposide': 'DNA_damage',
    'Mitomycin_C': 'DNA_damage',
    'Camptothecin': 'DNA_replication',
    'Floxuridine': 'DNA_replication',
    'Methotrexate': 'DNA_replication',
    'Mitoxantrone': 'DNA_replication',
    'AZ138': 'Eg5_inhibitors',
    'PP-2': 'Epithelial',
    'Alsterpaullone': 'Kinase_inhibitors',
    'Bryostatin': 'Kinase_inhibitors',
    'PD-169316': 'Kinase_inhibitors',
    'ALLN': 'Protein_degradation',
    'Lactacystin': 'Protein_degradation',
    'MG-132': 'Protein_degradation',
    'Proteasome_inhibitor_I': 'Protein_degradation',
    'Anisomycin': 'Protein_synthesis',
    'Cyclohexamide': 'Protein_synthesis',
    'Emetine': 'Protein_synthesis',
}

# SMILES for common compounds (for Morgan fingerprints)
COMPOUND_TO_SMILES = {
    'DMSO': 'CS(C)=O',
    'Taxol': 'CC1=C2[C@@]3([C@H]([C@@H](C4=C(C(=CC=C4)[C@@H]([C@]5([C@H](C[C@@H]6[C@](C5(C)C)(CO6)OC(=O)C)OC(C7=CC=CC=C7)(NC(C8=CC=CC=C8)=O)C)O)OC(=O)C)C3)OC(=O)C)OC2=C(C(=C1OC(C)=O)C)C)O',
    'Colchicine': 'COC1=CC2=C(C(=C1)OC)C(=CC(=O)C=C2)NC(=O)C',
    'Nocodazole': 'COC(=O)NC1=NC2=CC=C(C=C2[NH]1)C3=CC=CS3',
    'Vincristine': 'CC[C@]1(CC2CC(C3=C(CCN(C2)C1)C4=CC=CC=C4N3)(C5=C(C=C6C(=C5)[C@@]78CCN9[C@H]7[C@@](C=CC9)([C@H]([C@@]([C@@H]8N6C)(C(=O)OC)O)OC(=O)C)CC)OC)C(=O)OC)O',
    'Cytochalasin_B': 'CC(C)CC1C(=O)OC2=CC3=C(C=C2C(=O)C(C1O)C)C4=CC=CC=C4N3',
    'Demecolcine': 'COC1=C(C2=C(C(=CC(=O)C=C2)NC)C=C1OC)OC',
    'Latrunculin_B': 'CC1CCCCC(=O)NC(=CC2=CSC(=N2)C(CC(C)C)O)C(CC(CC(=O)OC(C(C=C1)O)C)O)O',
    'Methotrexate': 'CN(CC1=NC2=C(N=C(N=C2N=C1)N)N)C3=CC=C(C=C3)C(=O)NC(CCC(=O)O)C(=O)O',
    'Alsterpaullone': 'O=C(NC1=CC=CC=C1)NC2=C3C(=NC=N2)N(C)C4=CC=CC=C34',
    'AZ138': 'CC1=CC=C(C=C1)C2=CC=NC3=C2C(=O)NC(=N3)N',
    'Bryostatin': 'COC(=O)C1=CC2=CC=CC=C2N1',
    'PP-2': 'CC1=CC(=C(C=C1)C2=NN=C(N2)C3=CC=CC=C3)C(C)(C)C',
}


# ============================================================================
# DOWNLOAD UTILITIES
# ============================================================================

def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> bool:
    """Download a file with progress."""
    try:
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get('content-length', 0))
            
            with open(dest_path, 'wb') as f:
                downloaded = 0
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if TQDM_AVAILABLE and total_size > 0:
                        progress = downloaded / total_size * 100
                        print(f"\r  Downloading: {progress:.1f}%", end="", flush=True)
            
            print()
            return True
    except Exception as e:
        print(f"\n  Error downloading {url}: {e}")
        return False


def download_bbbc021_metadata(output_dir: Path) -> Path:
    """Download BBBC021 metadata files."""
    metadata_url = f"{BBBC021_BASE_URL}BBBC021_v1_image.csv"
    moa_url = f"{BBBC021_BASE_URL}BBBC021_v1_moa.csv"
    compound_url = f"{BBBC021_BASE_URL}BBBC021_v1_compound.csv"
    
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading metadata files...")
    
    files = [
        (metadata_url, metadata_dir / "image.csv"),
        (moa_url, metadata_dir / "moa.csv"),
        (compound_url, metadata_dir / "compound.csv"),
    ]
    
    for url, path in files:
        if not path.exists():
            print(f"  {path.name}...")
            download_file(url, path)
        else:
            print(f"  {path.name} already exists, skipping")
    
    return metadata_dir


# ============================================================================
# PREPROCESSING
# ============================================================================

def load_and_preprocess_image(
    image_paths: Dict[str, Path],
    output_size: int = 96,
) -> Optional[np.ndarray]:
    """
    Load multi-channel image and preprocess.
    
    Args:
        image_paths: Dict mapping channel name to file path
        output_size: Target image size
    
    Returns:
        Preprocessed image as numpy array (C, H, W)
    """
    channels = []
    
    # Expected channel order: DNA (DAPI), F-actin (Phalloidin), β-tubulin
    channel_order = ['DAPI', 'Phalloidin', 'Tubulin']
    
    for channel_name in channel_order:
        if channel_name not in image_paths:
            # Try alternative names
            alt_names = {
                'DAPI': ['DNA', 'Hoechst', 'w1'],
                'Phalloidin': ['Factin', 'F-actin', 'w2'],
                'Tubulin': ['Btubulin', 'w3'],
            }
            for alt in alt_names.get(channel_name, []):
                if alt in image_paths:
                    channel_name = alt
                    break
        
        if channel_name not in image_paths:
            continue
        
        path = image_paths[channel_name]
        
        try:
            if str(path).endswith('.tif') or str(path).endswith('.tiff'):
                img = tifffile.imread(str(path))
            else:
                img = np.array(Image.open(path))
            
            # Convert to float and normalize
            img = img.astype(np.float32)
            
            # Percentile normalization (robust to outliers)
            p1, p99 = np.percentile(img, [1, 99])
            img = np.clip((img - p1) / (p99 - p1 + 1e-8), 0, 1)
            
            # Resize if needed
            if img.shape[0] != output_size or img.shape[1] != output_size:
                pil_img = Image.fromarray((img * 255).astype(np.uint8))
                pil_img = pil_img.resize((output_size, output_size), Image.BILINEAR)
                img = np.array(pil_img).astype(np.float32) / 255.0
            
            channels.append(img)
            
        except Exception as e:
            print(f"  Warning: Failed to load {path}: {e}")
            return None
    
    if len(channels) < 3:
        return None
    
    # Stack channels (C, H, W)
    return np.stack(channels[:3], axis=0)


def crop_cells(
    image: np.ndarray,
    nuclei_centers: List[Tuple[int, int]],
    crop_size: int = 96,
) -> List[np.ndarray]:
    """Crop single cells from multi-channel image."""
    crops = []
    h, w = image.shape[1:3]
    half_size = crop_size // 2
    
    for cx, cy in nuclei_centers:
        x1 = max(0, cx - half_size)
        y1 = max(0, cy - half_size)
        x2 = min(w, cx + half_size)
        y2 = min(h, cy + half_size)
        
        crop = image[:, y1:y2, x1:x2]
        
        # Pad if needed
        if crop.shape[1] < crop_size or crop.shape[2] < crop_size:
            pad_crop = np.zeros((3, crop_size, crop_size), dtype=np.float32)
            pad_crop[:, :crop.shape[1], :crop.shape[2]] = crop
            crop = pad_crop
        
        crops.append(crop)
    
    return crops


def simple_nuclei_detection(dna_channel: np.ndarray, min_size: int = 100) -> List[Tuple[int, int]]:
    """
    Simple nuclei detection based on thresholding.
    For production, use CellProfiler or similar.
    """
    # Threshold
    threshold = np.mean(dna_channel) + 2 * np.std(dna_channel)
    binary = dna_channel > threshold
    
    # Simple connected components (without scipy)
    # Find local maxima as nuclei centers
    centers = []
    
    # Grid-based sampling as fallback
    step = 96
    for y in range(step // 2, dna_channel.shape[0] - step // 2, step):
        for x in range(step // 2, dna_channel.shape[1] - step // 2, step):
            # Check if there's signal in this region
            region = dna_channel[max(0, y-48):y+48, max(0, x-48):x+48]
            if np.mean(region) > threshold * 0.5:
                centers.append((x, y))
    
    return centers


# ============================================================================
# METADATA PROCESSING
# ============================================================================

def process_metadata(metadata_dir: Path, output_dir: Path) -> Path:
    """
    Process BBBC021 metadata into a unified format.
    
    Creates metadata.csv with columns:
    - image_path: Path to processed image
    - compound: Chemical compound name
    - concentration: Drug concentration
    - moa: Mode of action
    - batch: Plate/batch identifier
    - well: Well identifier
    - is_control: Whether this is a DMSO control
    - smiles: SMILES string for compound
    """
    print("Processing metadata...")
    
    # Load image metadata
    image_csv = metadata_dir / "image.csv"
    
    if not image_csv.exists():
        print("  Metadata files not found. Creating synthetic metadata...")
        return create_synthetic_metadata(output_dir)
    
    metadata = []
    
    with open(image_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            compound = row.get('Image_Metadata_Compound', 'DMSO')
            compound = compound.replace(' ', '_').replace('-', '_')
            
            # Clean up compound name
            if compound in ['', 'none', 'None', 'DMSO']:
                compound = 'DMSO'
            
            concentration = float(row.get('Image_Metadata_Concentration', 0))
            plate = row.get('Image_Metadata_Plate_DAPI', '')
            well = row.get('Image_Metadata_Well_DAPI', '')
            
            # Get MoA
            moa = COMPOUND_TO_MOA.get(compound, 'Unknown')
            
            # Get SMILES
            smiles = COMPOUND_TO_SMILES.get(compound, '')
            
            # Create image path
            image_name = f"{plate}_{well}.png"
            
            metadata.append({
                'image_path': image_name,
                'compound': compound,
                'concentration': concentration,
                'moa': moa,
                'batch': plate,
                'well': well,
                'is_control': compound == 'DMSO',
                'smiles': smiles,
            })
    
    # Save processed metadata
    output_csv = output_dir / "metadata.csv"
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'image_path', 'compound', 'concentration', 'moa',
            'batch', 'well', 'is_control', 'smiles'
        ])
        writer.writeheader()
        writer.writerows(metadata)
    
    print(f"  Saved metadata to {output_csv}")
    print(f"  Total entries: {len(metadata)}")
    
    return output_csv


def create_synthetic_metadata(output_dir: Path, num_samples: int = 10000) -> Path:
    """Create synthetic metadata for testing without real data."""
    print("Creating synthetic metadata for testing...")
    
    compounds = list(COMPOUND_TO_MOA.keys())[:20]  # Use first 20 compounds
    batches = [f"batch_{i:02d}" for i in range(10)]
    
    metadata = []
    
    # Ensure each batch has DMSO controls
    for batch in batches:
        # Add controls
        for i in range(50):
            metadata.append({
                'image_path': f'synthetic_{batch}_ctrl_{i:03d}.png',
                'compound': 'DMSO',
                'concentration': 0.0,
                'moa': 'Control',
                'batch': batch,
                'well': f'A{i+1:02d}',
                'is_control': True,
                'smiles': COMPOUND_TO_SMILES.get('DMSO', ''),
            })
        
        # Add treated
        for compound in compounds:
            if compound == 'DMSO':
                continue
            for i in range(10):
                metadata.append({
                    'image_path': f'synthetic_{batch}_{compound}_{i:03d}.png',
                    'compound': compound,
                    'concentration': 1.0,
                    'moa': COMPOUND_TO_MOA.get(compound, 'Unknown'),
                    'batch': batch,
                    'well': f'B{(compounds.index(compound)*10 + i + 1):02d}',
                    'is_control': False,
                    'smiles': COMPOUND_TO_SMILES.get(compound, ''),
                })
    
    # Save
    output_csv = output_dir / "metadata.csv"
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'image_path', 'compound', 'concentration', 'moa',
            'batch', 'well', 'is_control', 'smiles'
        ])
        writer.writeheader()
        writer.writerows(metadata)
    
    print(f"  Created synthetic metadata with {len(metadata)} entries")
    
    return output_csv


# ============================================================================
# MAIN PREPARATION PIPELINE
# ============================================================================

def prepare_bbbc021(
    output_dir: str = "./data/bbbc021",
    download: bool = True,
    image_size: int = 96,
    create_synthetic: bool = True,
):
    """
    Full BBBC021 data preparation pipeline.
    
    Args:
        output_dir: Output directory for processed data
        download: Whether to download the data
        image_size: Target image size
        create_synthetic: Create synthetic data for testing if download fails
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("BBBC021 Data Preparation")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Image size: {image_size}x{image_size}")
    print()
    
    if download:
        # Download metadata
        print("Step 1: Downloading metadata...")
        try:
            metadata_dir = download_bbbc021_metadata(output_dir)
            print("  Metadata downloaded successfully")
        except Exception as e:
            print(f"  Failed to download metadata: {e}")
            metadata_dir = None
    else:
        metadata_dir = output_dir / "metadata"
    
    # Process metadata
    print("\nStep 2: Processing metadata...")
    if metadata_dir and (metadata_dir / "image.csv").exists():
        metadata_path = process_metadata(metadata_dir, output_dir)
    else:
        print("  Using synthetic metadata")
        metadata_path = create_synthetic_metadata(output_dir)
    
    # Create synthetic images for testing
    if create_synthetic:
        print("\nStep 3: Creating synthetic images for testing...")
        create_synthetic_images(output_dir, metadata_path, image_size)
    
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    print(f"\nTo run the ablation study:")
    print(f"  python run_bbbc021_ablation.py --data-dir {output_dir}")


def create_synthetic_images(output_dir: Path, metadata_path: Path, image_size: int = 96):
    """Create synthetic cell images for testing."""
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        reader = csv.DictReader(f)
        metadata = list(reader)
    
    print(f"  Creating {len(metadata)} synthetic images...")
    
    # Create MoA-specific patterns
    moa_patterns = {
        'Control': {'nucleus_size': 1.0, 'actin_pattern': 'normal', 'tubulin_pattern': 'normal'},
        'Microtubule_stabilizers': {'nucleus_size': 1.2, 'actin_pattern': 'normal', 'tubulin_pattern': 'bundled'},
        'Microtubule_destabilizers': {'nucleus_size': 0.8, 'actin_pattern': 'normal', 'tubulin_pattern': 'fragmented'},
        'Actin_disruptors': {'nucleus_size': 1.0, 'actin_pattern': 'disrupted', 'tubulin_pattern': 'normal'},
        'Aurora_kinase_inhibitors': {'nucleus_size': 1.3, 'actin_pattern': 'normal', 'tubulin_pattern': 'normal'},
        'DNA_damage': {'nucleus_size': 0.7, 'actin_pattern': 'normal', 'tubulin_pattern': 'normal'},
        'DNA_replication': {'nucleus_size': 1.1, 'actin_pattern': 'normal', 'tubulin_pattern': 'normal'},
        'Eg5_inhibitors': {'nucleus_size': 0.9, 'actin_pattern': 'normal', 'tubulin_pattern': 'monopolar'},
        'Epithelial': {'nucleus_size': 1.0, 'actin_pattern': 'reduced', 'tubulin_pattern': 'normal'},
        'Kinase_inhibitors': {'nucleus_size': 1.1, 'actin_pattern': 'normal', 'tubulin_pattern': 'normal'},
        'Protein_degradation': {'nucleus_size': 0.9, 'actin_pattern': 'normal', 'tubulin_pattern': 'normal'},
        'Protein_synthesis': {'nucleus_size': 0.85, 'actin_pattern': 'normal', 'tubulin_pattern': 'normal'},
        'Cholesterol-lowering': {'nucleus_size': 1.0, 'actin_pattern': 'normal', 'tubulin_pattern': 'normal'},
    }
    
    count = 0
    iterator = tqdm(metadata, desc="  Generating") if TQDM_AVAILABLE else metadata
    
    for entry in iterator:
        image_path = images_dir / entry['image_path']
        
        if image_path.exists():
            count += 1
            continue
        
        moa = entry['moa']
        pattern = moa_patterns.get(moa, moa_patterns['Control'])
        batch_seed = hash(entry['batch']) % 1000
        
        # Generate synthetic image
        image = generate_synthetic_cell_image(
            image_size=image_size,
            nucleus_size=pattern['nucleus_size'],
            actin_pattern=pattern['actin_pattern'],
            tubulin_pattern=pattern['tubulin_pattern'],
            seed=hash(entry['image_path']) % (2**31),
            batch_seed=batch_seed,
        )
        
        # Save as PNG
        image_uint8 = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(np.transpose(image_uint8, (1, 2, 0)))
        pil_image.save(image_path)
        
        count += 1
    
    print(f"  Created {count} images")


def generate_synthetic_cell_image(
    image_size: int = 96,
    nucleus_size: float = 1.0,
    actin_pattern: str = 'normal',
    tubulin_pattern: str = 'normal',
    seed: int = 42,
    batch_seed: int = 0,
) -> np.ndarray:
    """Generate a synthetic cell image with specified patterns."""
    np.random.seed(seed)
    
    # Initialize channels
    dna = np.zeros((image_size, image_size), dtype=np.float32)
    actin = np.zeros((image_size, image_size), dtype=np.float32)
    tubulin = np.zeros((image_size, image_size), dtype=np.float32)
    
    # Create coordinate grids
    y, x = np.ogrid[-image_size//2:image_size//2, -image_size//2:image_size//2]
    
    # DNA channel (nucleus)
    nucleus_radius = int(image_size // 5 * nucleus_size)
    nucleus_mask = x**2 + y**2 <= nucleus_radius**2
    dna[nucleus_mask] = 0.8 + 0.2 * np.random.rand()
    
    # Add nucleus texture
    dna += np.random.randn(image_size, image_size) * 0.1
    dna[nucleus_mask] += 0.2
    
    # Actin channel (cytoskeleton)
    if actin_pattern == 'normal':
        # Stress fibers pattern
        for _ in range(8):
            angle = np.random.uniform(0, np.pi)
            thickness = np.random.uniform(2, 4)
            offset = np.random.uniform(-20, 20)
            fiber_mask = np.abs(x * np.sin(angle) - y * np.cos(angle) - offset) < thickness
            actin[fiber_mask] += 0.4 + 0.2 * np.random.rand()
    elif actin_pattern == 'disrupted':
        # Fragmented actin
        actin += np.random.rand(image_size, image_size) * 0.5
    elif actin_pattern == 'reduced':
        # Reduced actin signal
        actin += np.random.rand(image_size, image_size) * 0.2
    
    # Tubulin channel (microtubules)
    if tubulin_pattern == 'normal':
        # Radial pattern from centrosome
        cx, cy = np.random.randint(-5, 5, 2)
        for _ in range(12):
            angle = np.random.uniform(0, 2*np.pi)
            for r in range(5, image_size//2 - 10):
                px = int(image_size//2 + cx + r * np.cos(angle))
                py = int(image_size//2 + cy + r * np.sin(angle))
                if 0 <= px < image_size and 0 <= py < image_size:
                    tubulin[py-1:py+2, px-1:px+2] = 0.5 + 0.3 * np.random.rand()
    elif tubulin_pattern == 'bundled':
        # Thick bundles (stabilized)
        for _ in range(5):
            angle = np.random.uniform(0, np.pi)
            thickness = np.random.uniform(4, 8)
            offset = np.random.uniform(-15, 15)
            bundle_mask = np.abs(x * np.sin(angle) - y * np.cos(angle) - offset) < thickness
            tubulin[bundle_mask] += 0.6
    elif tubulin_pattern == 'fragmented':
        # Fragmented tubulin (destabilized)
        tubulin += np.random.rand(image_size, image_size) * 0.4
    elif tubulin_pattern == 'monopolar':
        # Single aster (Eg5 inhibition)
        cx, cy = np.random.randint(-10, 10, 2)
        for _ in range(20):
            angle = np.random.uniform(0, 2*np.pi)
            for r in range(2, image_size//3):
                px = int(image_size//2 + cx + r * np.cos(angle))
                py = int(image_size//2 + cy + r * np.sin(angle))
                if 0 <= px < image_size and 0 <= py < image_size:
                    tubulin[py-1:py+2, px-1:px+2] = 0.6
    
    # Apply batch-specific color shift (simulates batch effects)
    np.random.seed(batch_seed)
    batch_shift = np.random.uniform(-0.1, 0.1, 3)
    batch_scale = np.random.uniform(0.9, 1.1, 3)
    
    dna = np.clip(dna * batch_scale[0] + batch_shift[0], 0, 1)
    actin = np.clip(actin * batch_scale[1] + batch_shift[1], 0, 1)
    tubulin = np.clip(tubulin * batch_scale[2] + batch_shift[2], 0, 1)
    
    # Add noise
    noise_level = 0.05
    dna += np.random.randn(image_size, image_size) * noise_level
    actin += np.random.randn(image_size, image_size) * noise_level
    tubulin += np.random.randn(image_size, image_size) * noise_level
    
    # Clip and stack
    dna = np.clip(dna, 0, 1)
    actin = np.clip(actin, 0, 1)
    tubulin = np.clip(tubulin, 0, 1)
    
    return np.stack([dna, actin, tubulin], axis=0)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare BBBC021 dataset for ablation study",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--output-dir", type=str, default="./data/bbbc021",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--image-size", type=int, default=96,
        help="Target image size"
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download BBBC021 data (requires internet connection)"
    )
    parser.add_argument(
        "--synthetic-only", action="store_true",
        help="Only create synthetic data for testing"
    )
    
    args = parser.parse_args()
    
    prepare_bbbc021(
        output_dir=args.output_dir,
        download=args.download,
        image_size=args.image_size,
        create_synthetic=args.synthetic_only or not args.download,
    )


if __name__ == "__main__":
    main()
