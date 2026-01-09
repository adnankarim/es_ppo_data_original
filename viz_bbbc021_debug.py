import os
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

# ==============================================================================
# 1. SMART DATASET CLASS (With Recursive Search)
# ==============================================================================
class BBBC021VizDataset:
    def __init__(self, data_dir: str, metadata_file: str):
        self.data_dir = Path(data_dir)
        self.metadata = self._load_metadata(metadata_file)
        
        # Cache for file locations to speed up random access
        self.path_cache = {}

    def _load_metadata(self, metadata_file: str) -> List[Dict]:
        """Parses CSV."""
        # Resolve Metadata Path
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
                
                # Filename
                filename = row.get('image_path') or row.get('SAMPLE_KEY') or row.get('file_path') or ''
                if not filename.endswith('.npy'):
                    filename += '.npy'
                
                metadata.append({
                    'filename': filename,
                    'compound': compound,
                    'moa': moa,
                    'batch': batch,
                })
        return metadata

    def __len__(self):
        return len(self.metadata)

    def _smart_find_file(self, filename):
        """
        Aggressively searches for the file if standard paths fail.
        """
        # 1. Check Cache
        if filename in self.path_cache:
            return self.path_cache[filename]

        # 2. Check Direct Path
        direct_path = self.data_dir / filename
        if direct_path.exists():
            self.path_cache[filename] = direct_path
            return direct_path

        # 3. Recursive Search (The "Nuclear Option")
        # Looks for the filename in ANY subfolder of data_dir
        # This fixes 99% of "path mismatch" errors.
        # print(f"  Searching for {filename} in {self.data_dir}...")
        matches = list(self.data_dir.rglob(filename))
        
        if matches:
            # Found it! Cache the parent dir to speed up future lookups
            found_path = matches[0]
            self.path_cache[filename] = found_path
            return found_path
            
        return None

    def __getitem__(self, idx):
        meta = self.metadata[idx]
        filename = meta['filename']
        
        full_path = self._smart_find_file(filename)
        
        if full_path is None:
            return None, meta, f"Missing: {filename}"
        
        try:
            img_array = np.load(str(full_path))
            return img_array, meta, str(full_path)
        except Exception as e:
            return None, meta, f"Error: {e}"

# ==============================================================================
# 2. VISUALIZATION
# ==============================================================================
def visualize_random_samples(dataset, num_samples=5, output_file="bbbc021_debug.png"):
    total_samples = len(dataset)
    if total_samples == 0:
        print("Dataset is empty!")
        return

    # Pick random indices
    indices = np.random.choice(total_samples, size=num_samples, replace=False)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(4 * num_samples, 5))
    if num_samples == 1: axes = [axes]

    print(f"\nVisualizing {num_samples} random samples...")

    success_count = 0
    for i, idx in enumerate(indices):
        img_array, meta, path_info = dataset[idx]
        ax = axes[i]

        if img_array is None:
            # Print the explicit error to console
            print(f"  [Sample {i+1} Failed] {path_info}")
            ax.text(0.5, 0.5, "File Not Found", ha='center', va='center', color='red')
            ax.set_title(meta['filename'], fontsize=8)
            ax.axis('off')
            continue

        success_count += 1
        print(f"  [Sample {i+1} OK] Found at: {path_info}")

        # Display Logic
        img_disp = img_array
        if img_disp.shape[0] == 3 and img_disp.shape[2] != 3:
            img_disp = np.transpose(img_disp, (1, 2, 0))

        if img_disp.max() > 1.0:
            img_disp = img_disp / 255.0
        elif img_disp.min() < 0:
            img_disp = (img_disp + 1.0) / 2.0
            
        img_disp = np.clip(img_disp, 0, 1)

        ax.imshow(img_disp)
        ax.axis('off')
        
        title_text = f"{meta['compound']}\n{meta['moa'][:15]}..."
        ax.set_title(title_text, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"\n✓ Visualization saved to: {output_file}")
    print(f"  (Successfully loaded {success_count}/{num_samples} images)")

    if success_count == 0:
        print("\n!!! DIAGNOSTIC TIP !!!")
        print(f"The script recursively searched inside: {dataset.data_dir}")
        print("It could not find the files. Please check:")
        print("1. Are the .npy files actually in that folder?")
        print("2. Run 'ls -R' on that folder to see the structure.")

# ==============================================================================
# 3. MAIN
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--metadata-file", type=str, required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", type=str, default="bbbc021_debug.png")
    args = parser.parse_args()

    try:
        ds = BBBC021VizDataset(args.data_dir, args.metadata_file)
        visualize_random_samples(ds, args.samples, args.output)
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
