#!/usr/bin/env python3
"""Test script to check file integrity on Modal volume."""

import modal
import soundfile as sf
from pathlib import Path

app = modal.App("test-file-integrity")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("soundfile>=0.12.0")
)

signaltrain_volume = modal.Volume.from_name("signaltrain-dataset", create_if_missing=False)

@app.function(
    image=image,
    volumes={"/data/signaltrain": signaltrain_volume},
    timeout=60 * 5,
)
def check_file_integrity(file_path: str):
    """Check if a specific file can be read and verify its integrity."""
    import os
    
    print(f"Checking file: {file_path}")
    print(f"File exists: {os.path.exists(file_path)}")
    
    if not os.path.exists(file_path):
        return {"error": "File does not exist"}
    
    # Check file size
    file_size = os.path.getsize(file_path)
    print(f"File size: {file_size} bytes ({file_size / (1024**2):.2f} MB)")
    
    # Check file header
    with open(file_path, 'rb') as f:
        header = f.read(12)
        print(f"File header (hex): {header.hex()}")
        is_wav = (header[:4] == b'RIFF' and len(header) >= 12 and header[8:12] == b'WAVE')
        print(f"Valid WAV signature: {is_wav}")
    
    # Try to read with soundfile
    try:
        with sf.SoundFile(file_path) as f:
            print(f"✓ File opened successfully")
            print(f"  Format: {f.format}")
            print(f"  Subtype: {f.subtype}")
            print(f"  Sample rate: {f.samplerate} Hz")
            print(f"  Channels: {f.channels}")
            print(f"  Frames: {f.frames}")
            
            # Try reading a sample
            sample = f.read(100, dtype='float32')
            print(f"  Sample read successfully: shape={sample.shape}")
            
        return {
            "success": True,
            "file_size": file_size,
            "is_wav": is_wav,
            "format": f.format,
            "sample_rate": f.samplerate,
            "channels": f.channels,
            "frames": f.frames,
        }
    except sf.LibsndfileError as e:
        print(f"❌ LibsndfileError: {e}")
        return {
            "success": False,
            "error": str(e),
            "error_code": e.error,
            "file_size": file_size,
            "is_wav": is_wav,
        }
    except Exception as e:
        print(f"❌ Unexpected error: {type(e).__name__}: {e}")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "file_size": file_size,
        }

@app.local_entrypoint()
def test_file(file_path: str = "/data/signaltrain/SignalTrain_LA2A_Dataset_1.1/Train/input_146_.wav"):
    """Test a specific file on Modal volume."""
    result = check_file_integrity.remote(file_path)
    print("\n" + "="*60)
    print("Result:")
    print("="*60)
    import json
    print(json.dumps(result, indent=2))
