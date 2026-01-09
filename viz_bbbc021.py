import os
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from typing import Dict, List

# ==============================================================================
# 1. SIMPLIFIED DATASET CLASS (No ML dependencies, just loading)
# ==============================================================================
class BBBC021VizDataset:
    """
    Simplified version of the BBBC021 dataset class specifically for visualization.
    Removes dependencies on molecular encoders and complex splitting logic.
    """
    def __init__(self, data_dir: str, metadata_file: str):
        self.data_dir = Path(data_dir)
        self.metadata = self._load_metadata(metadata_file)
        self.image_size = 96

    def _load_metadata(self, metadata_file: str) -> List[Dict]:
        """Parses CSV to find image paths and labels."""
        # Handle path resolution
        if os.path.isabs(metadata_file) or os.path.exists(metadata_file):
            metadata_path = Path(metadata_file)
        else:
            metadata_path = self.data_dir / metadata_file

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")

        print(f"Loading metadata from: {metadata_path}")
        
        metadata = []
        with open(metadata_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Basic Info
                compound = row.get('compound') or row.get('CPD_NAME') or 'DMSO'
                moa = row.get('moa') or row.get('ANNOT') or row.get('MOA') or 'Unknown'
                batch = str(row.get('batch', row.get('BATCH', row.get('Plate', '0')))).strip()
                conc = float(row.get('concentration', row.get('DOSE', 0)))

                # Path Construction Logic (Crucial for BBBC021)
                filename = row.get('image_path') or row.get('SAMPLE_KEY') or row.get('file_path') or ''
                
                # Handle nested directory structure if present in CSV columns
                if filename and '/' not in filename and '\\' not in filename:
                    week = row.get('week') or row.get('Week') or ''
                    plate = row.get('plate') or row.get('Plate') or row.get('PLATE') or ''
                    if week and plate:
                        full_image_path = os.path.join(week, plate, filename)
                    else:
                        full_image_path = filename
                else:
                    full_image_path = filename

                metadata.append({
                    'image_path': full_image_path,
                    'compound': compound,
                    'moa': moa,
                    'batch': batch,
                    'concentration': conc
                })
        return metadata

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        meta = self.metadata[idx]
        
        # Load .npy file
        rel_path = meta['image_path']
        if not rel_path.endswith('.npy'):
            rel_path += '.npy'
            
        full_path = self.data_dir / rel_path
        
        try:
            # Load uint8 [0-255] -> Float Tensor
            img_array = np.load(str(full_path))
            # Shape is usually (H, W, C) in .npy, convert to (C, H, W) for standard consistency
            # But for plotting we will convert back later.
            return img_array, meta
        except Exception as e:
            print(f"Error loading {full_path}: {e}")
            return None, meta

# ==============================================================================
# 2. VISUALIZATION FUNCTION
# ==============================================================================
def visualize_random_samples(dataset, num_samples=5, output_file="bbbc021_grid.png"):
    """Picks random samples and plots them in a grid."""
    
    total_samples = len(dataset)
    if total_samples == 0:
        print("Dataset is empty!")
        return

    # Pick random indices
    indices = np.random.choice(total_samples, size=num_samples, replace=False)
    
    # Setup Plot
    fig, axes = plt.subplots(1, num_samples, figsize=(4 * num_samples, 5))
    if num_samples == 1: axes = [axes] # Handle single sample case

    print(f"\nVisualizing {num_samples} random samples...")

    for i, idx in enumerate(indices):
        img_array, meta = dataset[idx]
        ax = axes[i]

        if img_array is None:
            ax.text(0.5, 0.5, "Load Error", ha='center')
            continue

        # --- PREPROCESSING FOR DISPLAY ---
        # The training code converts uint8 to float [-1, 1].
        # Here we likely load the raw .npy which is usually uint8 or float.
        
        # If image is (H, W, C), matplotlib is happy.
        # If image is (C, H, W), we need to transpose.
        if img_array.shape[0] == 3 and img_array.shape[2] != 3:
            img_disp = np.transpose(img_array, (1, 2, 0))
        else:
            img_disp = img_array

        # Normalization check: if max > 1, assume 0-255 uint8.
        # If min < 0, assume -1 to 1 float.
        if img_disp.min() < 0:
            # Denormalize [-1, 1] -> [0, 1]
            img_disp = (img_disp + 1.0) / 2.0
        elif img_disp.max() > 1.0:
            # Normalize [0, 255] -> [0, 1]
            img_disp = img_disp / 255.0
            
        # Clip to ensure valid range
        img_disp = np.clip(img_disp, 0, 1)

        # Display Image
        ax.imshow(img_disp)
        ax.axis('off')

        # Add Details
        title_text = (
            f"CMP: {meta['compound']}\n"
            f"MoA: {meta['moa'][:15]}..\n"
            f"Conc: {meta['concentration']}\n"
            f"Batch: {meta['batch']}"
        )
        ax.set_title(title_text, fontsize=10, pad=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_file}")
    # plt.show() # Uncomment if running in a windowed environment

# ==============================================================================
# 3. MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize BBBC021 Dataset")
    parser.add_argument("--data-dir", type=str, default="./data/bbbc021_all", 
                        help="Path to the folder containing .npy images")
    parser.add_argument("--metadata-file", type=str, default="metadata/bbbc021_df_all.csv",
                        help="Path to metadata CSV")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to plot")
    parser.add_argument("--output", type=str, default="bbbc021_samples.png", help="Output filename")

    args = parser.parse_args()

    # 1. Init Dataset
    try:
        viz_dataset = BBBC021VizDataset(args.data_dir, args.metadata_file)
        print(f"Dataset Initialized. Found {len(viz_dataset)} samples.")
        
        # 2. Run Visualization
        visualize_random_samples(viz_dataset, num_samples=args.samples, output_file=args.output)
        
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        print("Tip: Check that --data-dir points to the folder containing the .npy files")
        print("     and --metadata-file points to the correct CSV.")