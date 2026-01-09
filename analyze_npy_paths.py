#!/usr/bin/env python3
"""
Analyze the paths.txt file containing .npy file paths.
Provides insights into the directory structure and file naming patterns.
"""

import re
from pathlib import Path
from collections import defaultdict, Counter
import argparse


def analyze_paths_file(paths_file):
    """
    Analyze the paths.txt file to extract structure and patterns.
    
    Args:
        paths_file: Path to the paths.txt file
        
    Returns:
        dict: Analysis results
    """
    paths_path = Path(paths_file)
    
    if not paths_path.exists():
        print(f"❌ Error: File does not exist: {paths_path}")
        return None
    
    print(f"📖 Reading paths from: {paths_path}")
    print("=" * 80)
    
    # Read all paths (skip header lines)
    paths = []
    with open(paths_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip header lines
            if line and not line.startswith('Total') and not line.startswith('Scanned') and not line.startswith('='):
                if line.endswith('.npy'):
                    paths.append(line)
    
    print(f"Found {len(paths)} .npy file paths")
    print("=" * 80)
    
    # Analyze structure
    week_patterns = defaultdict(int)
    plate_patterns = defaultdict(int)
    filename_patterns = defaultdict(int)
    directory_structure = defaultdict(int)
    
    # Extract patterns from filenames (e.g., "9_2975_102.0.npy" -> week=9, plate=2975, well=102)
    filename_regex = re.compile(r'(\d+)_(\d+)_(\d+\.\d+)\.npy$')
    
    for path in paths:
        path_obj = Path(path)
        
        # Directory structure
        if len(path_obj.parts) >= 2:
            week_dir = path_obj.parts[0]  # e.g., "Week9"
            plate_dir = path_obj.parts[1]  # e.g., "39282"
            directory_structure[f"{week_dir}/{plate_dir}"] += 1
            
            # Extract week number
            week_match = re.match(r'Week(\d+)', week_dir)
            if week_match:
                week_num = week_match.group(1)
                week_patterns[week_num] += 1
                plate_patterns[f"Week{week_num}/{plate_dir}"] += 1
        
        # Filename pattern
        filename = path_obj.name
        match = filename_regex.match(filename)
        if match:
            week_from_file, plate_from_file, well = match.groups()
            filename_patterns[f"Week{week_from_file}_Plate{plate_from_file}"] += 1
    
    # Print analysis
    print("\n📊 ANALYSIS RESULTS")
    print("=" * 80)
    
    print(f"\n📁 Directory Structure:")
    print(f"  Total unique directories: {len(directory_structure)}")
    print(f"  Top 10 directories by file count:")
    for directory, count in sorted(directory_structure.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    {directory}: {count} files")
    
    print(f"\n📅 Week Distribution:")
    for week in sorted(week_patterns.keys(), key=int):
        print(f"  Week {week}: {week_patterns[week]} files")
    
    print(f"\n🔢 Total Weeks: {len(week_patterns)}")
    print(f"🔢 Total Plates: {len(plate_patterns)}")
    
    # Sample paths
    print(f"\n📄 Sample Paths (first 10):")
    for i, path in enumerate(paths[:10], 1):
        print(f"  {i:2d}. {path}")
    
    # Check for path consistency
    print(f"\n🔍 Path Consistency Check:")
    inconsistent = []
    for path in paths[:100]:  # Check first 100
        path_obj = Path(path)
        if len(path_obj.parts) >= 2:
            week_dir = path_obj.parts[0]
            plate_dir = path_obj.parts[1]
            filename = path_obj.name
            
            # Extract week from directory and filename
            week_match = re.match(r'Week(\d+)', week_dir)
            file_match = filename_regex.match(filename)
            
            if week_match and file_match:
                week_from_dir = week_match.group(1)
                week_from_file = file_match.group(1)
                
                if week_from_dir != week_from_file:
                    inconsistent.append((path, week_from_dir, week_from_file))
    
    if inconsistent:
        print(f"  ⚠️  Found {len(inconsistent)} inconsistent paths (week mismatch):")
        for path, dir_week, file_week in inconsistent[:5]:
            print(f"    {path} (dir: Week{dir_week}, file: Week{file_week})")
    else:
        print(f"  ✅ All checked paths are consistent")
    
    # Generate path mapping for data loading
    print(f"\n💡 Path Format:")
    print(f"  Relative paths from data directory: {paths[0] if paths else 'N/A'}")
    print(f"  Example full path: data/bbbc021_all/{paths[0] if paths else 'N/A'}")
    
    return {
        'total_paths': len(paths),
        'unique_directories': len(directory_structure),
        'week_distribution': dict(week_patterns),
        'directory_structure': dict(directory_structure),
        'paths': paths
    }


def create_path_mapping(paths_file, output_file='path_mapping.csv'):
    """
    Create a CSV mapping of paths for easier data loading.
    
    Args:
        paths_file: Path to the paths.txt file
        output_file: Output CSV file path
    """
    result = analyze_paths_file(paths_file)
    
    if not result:
        return
    
    import csv
    
    print(f"\n💾 Creating path mapping CSV...")
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['relative_path', 'week', 'plate', 'well', 'filename'])
        
        filename_regex = re.compile(r'(\d+)_(\d+)_(\d+\.\d+)\.npy$')
        
        for path in result['paths']:
            path_obj = Path(path)
            filename = path_obj.name
            
            match = filename_regex.match(filename)
            if match:
                week, plate, well = match.groups()
                writer.writerow([path, week, plate, well, filename])
    
    print(f"✅ Path mapping saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze paths.txt file containing .npy file paths'
    )
    parser.add_argument(
        'paths_file',
        type=str,
        default='paths.txt',
        nargs='?',
        help='Path to the paths.txt file (default: paths.txt)'
    )
    parser.add_argument(
        '--create-mapping',
        action='store_true',
        help='Create a CSV mapping file for easier data loading'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='path_mapping.csv',
        help='Output file for CSV mapping (default: path_mapping.csv)'
    )
    
    args = parser.parse_args()
    
    if args.create_mapping:
        create_path_mapping(args.paths_file, args.output)
    else:
        analyze_paths_file(args.paths_file)
    
    return 0


if __name__ == '__main__':
    exit(main())
