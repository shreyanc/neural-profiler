#!/usr/bin/env python3
"""Test script to reproduce the dataloader file reading issue."""

import sys
from pathlib import Path
import soundfile as sf

# Test the exact code path from dataloader.py
def test_dataloader_reading(file_path_str: str):
    """Reproduce the exact code from dataloader.py line 150."""
    print(f"Testing file: {file_path_str}")
    print(f"Type: {type(file_path_str)}")
    
    try:
        # This is the exact code from dataloader.py line 150
        with sf.SoundFile(str(file_path_str)) as f:
            sample_rate = f.samplerate
            frames = len(f)
            duration = frames / sample_rate
            print(f"✓ Success!")
            print(f"  Sample rate: {sample_rate}")
            print(f"  Frames: {frames}")
            print(f"  Duration: {duration:.2f}s")
            return True
    except sf.LibsndfileError as e:
        print(f"❌ LibsndfileError: {e}")
        print(f"  Error code: {e.error}")
        print(f"  Error message: {str(e)}")
        
        # Additional diagnostics
        file_path = Path(file_path_str)
        if file_path.exists():
            print(f"\nFile exists: True")
            print(f"File size: {file_path.stat().st_size} bytes")
            print(f"File readable: {file_path.is_file()}")
            
            # Check file header
            with open(file_path, 'rb') as f:
                header = f.read(12)
                print(f"File header: {header.hex()}")
                if len(header) >= 12:
                    print(f"  RIFF: {header[:4] == b'RIFF'}")
                    print(f"  WAVE: {header[8:12] == b'WAVE'}")
        else:
            print(f"File exists: False")
        
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_dataloader_file.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    print(f"Soundfile version: {sf.__version__}")
    print(f"Libsndfile version: {sf.__libsndfile_version__}\n")
    
    success = test_dataloader_reading(file_path)
    sys.exit(0 if success else 1)
