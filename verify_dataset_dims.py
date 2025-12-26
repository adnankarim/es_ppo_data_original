import os
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import Counter

def verify_dataset(data_dir, num_samples=500):
    """
    Scans .npy files to verify dimensions, channels, and normalization.
    
    Args:
        data_dir (str): Path to dataset folder
        num_samples (int): Number of files to check (set to -1 for all files)
    """
    data_path = Path(data_dir)
    
    # 1. Gather all .npy files
    print(f"🔍 Scanning directory: {data_path} ...")
    all_files = list(data_path.rglob("*.npy"))
    
    if not all_files:
        print(f"❌ No .npy files found in {data_path}")
        return

    total_files = len(all_files)
    print(f"✅ Found {total_files} .npy files.")

    # 2. Determine sample size
    if num_samples == -1 or num_samples > total_files:
        samples = all_files
        print("   Checking ALL files (this might take a while)...")
    else:
        # Deterministic sampling for reproducibility
        np.random.seed(42)
        samples = np.random.choice(all_files, num_samples, replace=False)
        print(f"   Checking random sample of {num_samples} files...")

    # 3. Stats Containers
    shapes = Counter()
    dtypes = Counter()
    value_ranges = {
        'min': [],
        'max': [],
        'mean': []
    }
    corrupted_files = []
    
    # 4. Loop and Check
    for f in tqdm(samples, desc="Verifying"):
        try:
            # Load only metadata (mmap_mode='r') is faster if just checking shape
            # but we need min/max, so we load fully
            arr = np.load(f)
            
            shapes[arr.shape] += 1
            dtypes[str(arr.dtype)] += 1
            
            value_ranges['min'].append(arr.min())
            value_ranges['max'].append(arr.max())
            value_ranges['mean'].append(arr.mean())
            
        except Exception as e:
            corrupted_files.append((str(f), str(e)))

    # 5. Generate Report
    print("\n" + "="*60)
    print("DATASET VERIFICATION REPORT")
    print("="*60)
    
    print(f"\n📁 Total Files Checked: {len(samples)}")
    
    print(f"\n📐 SHAPES FOUND:")
    for shape, count in shapes.items():
        # Heuristic to check if it's correct
        is_correct = "✅" if (96 in shape) else "⚠️"
        print(f"   {is_correct} {shape}: {count} files ({count/len(samples):.1%})")

    print(f"\n💾 DATA TYPES:")
    for dtype, count in dtypes.items():
        is_optimal = "✅" if "uint8" in dtype else "⚠️ (Check Normalization!)"
        print(f"   {is_optimal} {dtype}: {count} files")

    # Value Range Analysis
    avg_min = np.mean(value_ranges['min'])
    avg_max = np.mean(value_ranges['max'])
    global_max = np.max(value_ranges['max'])
    
    print(f"\n📊 VALUE RANGES (Intensity):")
    print(f"   Avg Min: {avg_min:.4f}")
    print(f"   Avg Max: {avg_max:.4f}")
    print(f"   Global Peak: {global_max}")
    
    print("\n🧠 DIAGNOSIS:")
    if global_max > 1.0:
        print("   ➤ Type: 8-bit or 16-bit Integer (0-255 or 0-65535)")
        print("   ➤ Action Required: You MUST normalize in your Dataset class:")
        print("     img = img / 127.5 - 1.0  (if 8-bit)")
        print("     img = robust_norm(img)   (if 16-bit)")
    elif avg_min < 0:
        print("   ➤ Type: Normalized Float (likely -1 to 1)")
        print("   ➤ Action Required: No normalization needed in Dataset class.")
    else:
        print("   ➤ Type: Normalized Float (likely 0 to 1)")
        print("   ➤ Action Required: Scale to [-1, 1] for Diffusion:")
        print("     img = img * 2.0 - 1.0")

    if corrupted_files:
        print(f"\n❌ CORRUPTED FILES ({len(corrupted_files)}):")
        for name, err in corrupted_files[:5]:
            print(f"   {name}: {err}")
        if len(corrupted_files) > 5: print("   ...and more.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to BBBC021/IMPA dataset folder")
    parser.add_argument("--samples", type=int, default=500, help="Number of files to check (-1 for all)")
    args = parser.parse_args()
    
    verify_dataset(args.path, args.samples)