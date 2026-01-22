#!/usr/bin/env python3
"""Test script that exactly reproduces the dataloader's file reading behavior."""

import sys
from pathlib import Path
import soundfile as sf
import re

def test_dataloader_exact_behavior(root_dir: str, subset: str = "Train"):
    """Reproduce the exact behavior of SignalTrainLA2ADataset._find_paired_files"""
    
    root_path = Path(root_dir)
    subset_dir = root_path / subset
    
    print(f"Root dir: {root_path}")
    print(f"Subset dir: {subset_dir}")
    print(f"Subset dir exists: {subset_dir.exists()}")
    print()
    
    if not subset_dir.exists():
        print(f"❌ Subset directory not found: {subset_dir}")
        return
    
    input_pattern = re.compile(r"input_(\d+)_\.wav")
    target_pattern = re.compile(r"target_(\d+)_LA2A_([^_]+)__([^_]+)__([^_]+)\.wav")
    
    input_files: dict = {}
    target_files: dict = {}
    
    # Find all input files - EXACT code from dataloader
    print("Finding input files...")
    for file_path in subset_dir.glob("input_*.wav"):
        match = input_pattern.match(file_path.name)
        if match:
            num = int(match.group(1))
            input_files[num] = file_path
            print(f"  Found input_{num}_.wav -> {file_path}")
    
    # Find all target files - EXACT code from dataloader
    print("\nFinding target files...")
    for file_path in subset_dir.glob("target_*_LA2A_*.wav"):
        match = target_pattern.match(file_path.name)
        if match:
            num = int(match.group(1))
            s1, s2, s3 = match.groups()[1:]
            target_files[num] = (file_path, (s1, s2, s3))
            print(f"  Found target_{num}_LA2A_{s1}__{s2}__{s3}.wav -> {file_path}")
    
    print(f"\nFound {len(input_files)} input files and {len(target_files)} target files")
    
    # Process pairs - EXACT code from dataloader
    pairs = []
    for num in sorted(set(input_files.keys()) & set(target_files.keys())):
        input_path = input_files[num]
        target_path, states = target_files[num]
        
        print(f"\n{'='*60}")
        print(f"Processing pair {num}:")
        print(f"  Input: {input_path}")
        print(f"  Target: {target_path}")
        print(f"  Input type: {type(input_path)}")
        print(f"  Input as str: {str(input_path)}")
        print(f"  Input exists: {input_path.exists()}")
        
        # EXACT code from dataloader line 150
        try:
            print(f"\n  Opening with soundfile...")
            with sf.SoundFile(str(input_path)) as f:
                sample_rate = f.samplerate
                frames = len(f)
                duration = frames / sample_rate
                print(f"  ✓ Success!")
                print(f"    Sample rate: {sample_rate}")
                print(f"    Frames: {frames}")
                print(f"    Duration: {duration:.2f}s")
                
            pairs.append({
                "num": num,
                "input_path": input_path,
                "target_path": target_path,
                "states": states,
                "sample_rate": sample_rate,
                "frames": frames,
                "duration": duration,
            })
        except sf.LibsndfileError as e:
            print(f"  ❌ LibsndfileError: {e}")
            print(f"    Error code: {e.error}")
            print(f"    This is the exact error from dataloader!")
            return False
        except Exception as e:
            print(f"  ❌ Unexpected error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print(f"\n{'='*60}")
    print(f"✓ Successfully processed {len(pairs)} pairs")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_dataloader_exact.py <root_dir> [subset]")
        print("\nExample:")
        print("  python test_dataloader_exact.py /data/signaltrain/SignalTrain_LA2A_Dataset_1.1 Train")
        sys.exit(1)
    
    root_dir = sys.argv[1]
    subset = sys.argv[2] if len(sys.argv) > 2 else "Train"
    
    print(f"Soundfile version: {sf.__version__}")
    print(f"Libsndfile version: {sf.__libsndfile_version__}\n")
    
    success = test_dataloader_exact_behavior(root_dir, subset)
    sys.exit(0 if success else 1)
