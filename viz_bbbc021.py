import os
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

# ==============================================================================
# 1. ROBUST DATASET CLASS
# ==============================================================================
class BBBC021VizDataset:
    def __init__(self, data_dir: str, metadata_file: str):
        self.data_dir = Path(data_dir)
        self.metadata = self._load_metadata(metadata_file)

    def _load_metadata(self, metadata_file: str) -> List[Dict]:
        """Parses CSV to find image paths."""
        # 1. Resolve Metadata Path
        if os.path.exists(metadata_file):
            metadata_path = Path(metadata_file)
        else:
            metadata_path = self.data_dir / metadata_file

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata CSV not found at: {metadata_path}")

        print(f"Reading CSV: {metadata_path}")
        
        metadata = []
        with open(metadata_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Basic Info
                compound = row.get('compound') or row.get('CPD_NAME') or 'DMSO'
                moa = row.get('moa') or row.get('ANNOT') or row.get('MOA') or 'Unknown'
                batch = str(row.get('batch', row.get('BATCH', row.get('Plate', '0')))).strip()
                conc = float(row.get('concentration', row.get('DOSE', 0)))

                # 2. Smart Path Construction
                # Check different column names for filename
                filename = row.get('image_path') or row.get('SAMPLE_KEY') or row.get('file_path') or ''
                
                # Check for nested structure columns (Week/Plate)
                week = row.get('week') or row.get('Week') or ''
                plate = row.get('plate') or row.get('Plate') or row.get('PLATE') or ''
                
                metadata.append({
                    'filename': filename,
                    'week': week,
                    'plate': plate,
                    'compound': compound,
                    'moa': moa,
                    'batch': batch,
                    'concentration': conc
                })
        return metadata

    def __len__(self):
        return len(self.metadata)

    def _find_image_file(self, meta):
        """Tries multiple path combinations to find the file."""
        candidates = []
        filename = meta['filename']
        
        # Ensure extension
        if not filename.endswith('.npy'):
            filename += '.npy'

        # Candidate 1: Direct path (data_dir/filename.npy)
        candidates.append(self.data_dir / filename)

        # Candidate 2: Nested path (data_dir/Week/Plate/filename.npy)
        if meta['week'] and meta['plate']:
            candidates.append(self.data_dir / meta['week'] / meta['plate'] / filename)
            
        # Candidate 3: Just Plate nested (data_dir/Plate/filename.npy)
        if meta['plate']:
             candidates.append(self.data_dir / meta['plate'] / filename)

        for path in candidates:
            if path.exists():
                return path
        
        return candidates[0] # Return the first one (failed) for error printing

    def __getitem__(self, idx):
        meta = self.metadata[idx]
        full_path = self._find_image_file(meta)
        
        try:
            # Load .npy file
            img_array = np.load(str(full_path))
            return img_array, meta
        except Exception as e:
            # Return path for debug printing
            return str(full_path), None

# ==============================================================================
# 2. VISUALIZATION
# ==============================================================================
def visualize_random_samples(dataset, num_samples=5, output_file="bbbc021_grid.png"):
    total_samples = len(dataset)
    if total_samples == 0:
        print("Dataset is empty!")
        return

    # Pick random indices
    indices = np.random.choice(total_samples, size=num_samples, replace=False)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(4 * num_samples, 5))
    if num_samples == 1: axes = [axes]

    print(f"\nVisualizing {num_samples} random samples...")

    for i, idx in enumerate(indices):
        data, meta = dataset[idx]
        ax = axes[i]

        if meta is None:
            # This means loading failed. 'data' contains the failed path string.
            print(f"  [Error] Could not find file: {data}")
            ax.text(0.5, 0.5, "File Not Found", ha='center', va='center')
            ax.set_title("Missing File", color='red')
            ax.axis('off')
            continue

        # Valid Image Loaded
        img_disp = data
        
        # Transpose if channels are first: (3, H, W) -> (H, W, 3)
        if img_disp.shape[0] == 3 and img_disp.shape[2] != 3:
            img_disp = np.transpose(img_disp, (1, 2, 0))

        # Normalize for Display
        # Case A: Raw uint8 [0, 255] -> Scale to [0, 1]
        if img_disp.max() > 1.0:
            img_disp = img_disp / 255.0
        # Case B: Pre-processed [-1, 1] -> Scale to [0, 1]
        elif img_disp.min() < 0:
            img_disp = (img_disp + 1.0) / 2.0
            
        img_disp = np.clip(img_disp, 0, 1)

        ax.imshow(img_disp)
        ax.axis('off')
        
        title_text = f"{meta['compound']}\n{meta['moa'][:15]}..."
        ax.set_title(title_text, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"✓ Visualization saved to: {output_file}")

# ==============================================================================
# 3. MAIN
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # TIP: Use absolute paths if relative ones are confusing
    parser.add_argument("--data-dir", type=str, required=True, help="Root folder containing .npy files")
    parser.add_argument("--metadata-file", type=str, required=True, help="Path to .csv file")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", type=str, default="bbbc021_samples.png")
    args = parser.parse_args()

    try:
        ds = BBBC021VizDataset(args.data_dir, args.metadata_file)
        visualize_random_samples(ds, args.samples, args.output)
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
