#!/usr/bin/env python3
"""
Recursively scan for all .npy files in the data folder and its subdirectories.
Provides detailed statistics and path information.
"""

import os
from pathlib import Path
from collections import defaultdict
import argparse


def scan_npy_files(data_dir, verbose=True):
    """
    Recursively scan for all .npy files in the given directory.
    
    Args:
        data_dir: Path to the data directory
        verbose: If True, print detailed information
        
    Returns:
        dict: Dictionary with statistics and file paths
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ Error: Directory does not exist: {data_path}")
        return None
    
    if not data_path.is_dir():
        print(f"❌ Error: Not a directory: {data_path}")
        return None
    
    # Collect all .npy files
    npy_files = []
    directory_counts = defaultdict(int)
    total_size = 0
    
    print(f"🔍 Scanning for .npy files in: {data_path.absolute()}")
    print("=" * 80)
    
    # Recursively walk through all subdirectories
    for root, dirs, files in os.walk(data_path):
        root_path = Path(root)
        
        for file in files:
            if file.endswith('.npy'):
                full_path = root_path / file
                relative_path = full_path.relative_to(data_path)
                
                npy_files.append({
                    'full_path': str(full_path),
                    'relative_path': str(relative_path),
                    'directory': str(root_path.relative_to(data_path)),
                    'filename': file,
                    'size': full_path.stat().st_size if full_path.exists() else 0
                })
                
                # Count files per directory
                directory_counts[str(root_path.relative_to(data_path))] += 1
                total_size += npy_files[-1]['size']
    
    # Print summary
    print(f"\n📊 SUMMARY")
    print("=" * 80)
    print(f"Total .npy files found: {len(npy_files)}")
    print(f"Total size: {total_size / (1024**3):.2f} GB" if total_size > 0 else "Total size: 0 bytes")
    print(f"Unique directories: {len(directory_counts)}")
    
    # Print directory structure
    if verbose:
        print(f"\n📁 DIRECTORY STRUCTURE (files per directory):")
        print("=" * 80)
        for directory, count in sorted(directory_counts.items()):
            print(f"  {directory}: {count} files")
    
    # Print sample paths
    if verbose and npy_files:
        print(f"\n📄 SAMPLE PATHS (first 20 files):")
        print("=" * 80)
        for i, file_info in enumerate(npy_files[:20], 1):
            print(f"  {i:3d}. {file_info['relative_path']}")
        
        if len(npy_files) > 20:
            print(f"\n  ... and {len(npy_files) - 20} more files")
    
    # Find unique directory patterns
    if verbose:
        print(f"\n🔍 DIRECTORY PATTERNS:")
        print("=" * 80)
        pattern_counts = defaultdict(int)
        for file_info in npy_files:
            dir_parts = Path(file_info['directory']).parts
            if dir_parts:
                # Get top-level pattern (e.g., "Week1", "Week2", etc.)
                pattern = dir_parts[0] if dir_parts else ""
                pattern_counts[pattern] += 1
        
        for pattern, count in sorted(pattern_counts.items()):
            print(f"  {pattern}: {count} files")
    
    # Save to file
    output_file = "npy_files_list.txt"
    with open(output_file, 'w') as f:
        f.write(f"Total .npy files: {len(npy_files)}\n")
        f.write(f"Scanned from: {data_path.absolute()}\n")
        f.write("=" * 80 + "\n\n")
        
        for file_info in npy_files:
            f.write(f"{file_info['relative_path']}\n")
    
    print(f"\n💾 Full file list saved to: {output_file}")
    
    return {
        'total_files': len(npy_files),
        'total_size': total_size,
        'directories': dict(directory_counts),
        'files': npy_files,
        'output_file': output_file
    }


def main():
    parser = argparse.ArgumentParser(
        description='Recursively scan for all .npy files in the data folder'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Path to the data directory (default: data)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Only print summary, skip detailed output'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Custom output file for file list (default: npy_files_list.txt)'
    )
    
    args = parser.parse_args()
    
    result = scan_npy_files(args.data_dir, verbose=not args.quiet)
    
    if result and args.output:
        # Copy to custom output file if specified
        import shutil
        shutil.copy(result['output_file'], args.output)
        print(f"💾 Also saved to: {args.output}")
    
    if result:
        print("\n✅ Scan completed successfully!")
        return 0
    else:
        print("\n❌ Scan failed!")
        return 1


if __name__ == '__main__':
    exit(main())
