#!/usr/bin/env python3
"""Test script to investigate soundfile reading issues."""

import sys
from pathlib import Path
import soundfile as sf
import os

def test_file(file_path: Path):
    """Test if a file can be read by soundfile."""
    print(f"\n{'='*60}")
    print(f"Testing: {file_path}")
    print(f"{'='*60}")
    
    # Check if file exists
    if not file_path.exists():
        print(f"❌ File does not exist")
        return False
    
    # Check file size
    file_size = file_path.stat().st_size
    print(f"File size: {file_size} bytes ({file_size / 1024:.2f} KB)")
    
    if file_size == 0:
        print(f"❌ File is empty (0 bytes)")
        return False
    
    # Check file permissions
    print(f"Readable: {os.access(file_path, os.R_OK)}")
    print(f"Writable: {os.access(file_path, os.W_OK)}")
    
    # Check first few bytes (file signature)
    try:
        with open(file_path, 'rb') as f:
            header = f.read(12)
            print(f"File header (hex): {header.hex()}")
            print(f"File header (ascii): {header[:4] if len(header) >= 4 else ''}")
            
            # Check for WAV signature
            if header[:4] == b'RIFF' and header[8:12] == b'WAVE':
                print("✓ Valid WAV file signature (RIFF...WAVE)")
            else:
                print(f"⚠ Unexpected file signature")
                print(f"  Expected: RIFF...WAVE")
                print(f"  Got: {header[:4]}...{header[8:12] if len(header) >= 12 else ''}")
    except Exception as e:
        print(f"❌ Error reading file header: {e}")
        return False
    
    # Try to read with soundfile
    try:
        with sf.SoundFile(str(file_path)) as f:
            print(f"✓ File opened successfully")
            print(f"  Format: {f.format}")
            print(f"  Subtype: {f.subtype}")
            print(f"  Sample rate: {f.samplerate} Hz")
            print(f"  Channels: {f.channels}")
            print(f"  Frames: {f.frames}")
            print(f"  Duration: {f.frames / f.samplerate:.2f} seconds")
            
            # Try reading a small sample
            sample = f.read(100, dtype='float32')
            print(f"  Sample data shape: {sample.shape}")
            print(f"  Sample data range: [{sample.min():.4f}, {sample.max():.4f}]")
            
        return True
    except sf.LibsndfileError as e:
        print(f"❌ LibsndfileError: {e}")
        print(f"  Error code: {e.error}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {type(e).__name__}: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_soundfile.py <file_path>")
        print("\nExample:")
        print("  python test_soundfile.py /path/to/dataset/Train/input_146_.wav")
        sys.exit(1)
    
    file_path = Path(sys.argv[1])
    
    print(f"Soundfile version: {sf.__version__}")
    print(f"Libsndfile version: {sf.__libsndfile_version__}")
    
    success = test_file(file_path)
    
    if success:
        print("\n✓ File can be read successfully")
        sys.exit(0)
    else:
        print("\n❌ File cannot be read")
        sys.exit(1)

if __name__ == "__main__":
    main()
