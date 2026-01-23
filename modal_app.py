# modal_app.py
"""Modal app to train the LSTM model on SignalTrain using Modal GPUs.

This app downloads the dataset from AWS S3 and trains the model.
"""

import os
import time
from pathlib import Path

import modal

APP_NAME = "neural-profiler-train"
WORKDIR = "/workspace"
DATASET_MOUNT = "/data/signaltrain"

# Persistent volume for the SignalTrain dataset
signaltrain_volume = modal.Volume.from_name(
    "signaltrain-dataset", create_if_missing=True
)

# Base image with runtime dependencies (GPU-enabled)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("awscli")  # AWS CLI for S3 access
    .uv_pip_install(
        # Core deps
        "numpy>=2.3.3",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "h5py>=3.8.0",
        "scipy>=1.11.0",
        "pyyaml>=6.0.0",
        "ipykernel>=7.1.0",
        "matplotlib>=3.10.8",
        "ipywidgets>=8.1.8",
        "jupyterlab-widgets>=3.0.16",
        # AWS SDK
        "boto3>=1.34.0",
        # Torch stack (GPU / CUDA 12.8)
        "torch>=2.1.2",
        "torchvision>=0.16.1",
        # Training stack
        "pytorch-lightning>=2.2.0",
        "wandb>=0.16.0",
    )
    # Include project code so imports resolve even outside the mounted path
    .add_local_dir(Path(__file__).parent, remote_path=WORKDIR)
)

app = modal.App(APP_NAME)


def get_s3_client(role_arn: str):
    """Trade a Modal OIDC token for AWS credentials and return an S3 client.
    
    Args:
        role_arn: AWS IAM role ARN to assume via OIDC
        
    Returns:
        boto3 S3 client with temporary credentials
    """
    import boto3
    
    sts_client = boto3.client("sts")
    
    # Assume role with Web Identity using Modal's OIDC token
    credential_response = sts_client.assume_role_with_web_identity(
        RoleArn=role_arn,
        RoleSessionName="ModalOIDCSession",
        WebIdentityToken=os.environ["MODAL_IDENTITY_TOKEN"]
    )
    
    # Extract credentials
    credentials = credential_response["Credentials"]
    return boto3.client(
        "s3",
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
    )


@app.function(
    image=image,
    volumes={DATASET_MOUNT: signaltrain_volume},
    timeout=60 * 60 * 4,  # 4 hours
)
def sync_dataset_from_s3(
    s3_bucket: str,
    role_arn: str,
    s3_prefix: str = "",
    target_subdir: str = "",
    force: bool = False,
):
    """Download dataset from S3 to Modal volume using OIDC authentication.
    
    Args:
        s3_bucket: S3 bucket name
        role_arn: AWS IAM role ARN to assume via OIDC (e.g., "arn:aws:iam::ACCOUNT:role/ModalOIDCRole")
        s3_prefix: S3 prefix/path within bucket (e.g., "datasets/signaltrain/" or "datasets/signaltrain/SignalTrain_LA2A_Dataset_1.1/")
        target_subdir: Optional subdirectory name under DATASET_MOUNT (e.g., "SignalTrain_LA2A_Dataset_1.1" or "" for root)
        force: If True, re-download even if dataset already exists
    """
    import boto3
    from botocore.exceptions import ClientError
    
    # Determine dataset path
    if target_subdir:
        dataset_path = f"{DATASET_MOUNT}/{target_subdir}"
    else:
        dataset_path = DATASET_MOUNT
    
    done_marker = f"{dataset_path}/.dataset_synced"
    
    if os.path.exists(done_marker) and not force:
        print(f"Dataset already synced at {dataset_path}")
        print("Use --force to re-sync")
        return
    
    if force and os.path.exists(done_marker):
        print("Force flag set. Re-syncing dataset...")
        os.remove(done_marker)
    
    # Initialize S3 client using OIDC authentication
    s3_client = get_s3_client(role_arn)
    
    # Ensure dataset directory exists
    os.makedirs(dataset_path, exist_ok=True)
    
    print("="*60)
    print(f"Syncing dataset from S3")
    print(f"Bucket: {s3_bucket}")
    print(f"Prefix: {s3_prefix}")
    print(f"Destination: {dataset_path}")
    print("="*60 + "\n")
    
    start_time = time.time()
    
    # List all objects in S3 with the given prefix
    print("Listing objects in S3...")
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix)
    
    all_objects = []
    for page in pages:
        if 'Contents' in page:
            all_objects.extend(page['Contents'])
    
    if not all_objects:
        raise ValueError(f"No objects found in s3://{s3_bucket}/{s3_prefix}")
    
    total_size = sum(obj['Size'] for obj in all_objects)
    total_size_gb = total_size / (1024**3)
    print(f"Found {len(all_objects)} objects ({total_size_gb:.2f}GB)\n")
    
    # Download files
    downloaded_size = 0
    downloaded_files = 0
    
    for i, obj in enumerate(all_objects, 1):
        s3_key = obj['Key']
        file_size = obj['Size']
        
        # Skip if it's just a directory marker
        if s3_key.endswith('/'):
            continue
        
        # Construct local file path
        # Remove the prefix from the S3 key to get relative path
        if s3_prefix and s3_key.startswith(s3_prefix):
            relative_path = s3_key[len(s3_prefix):].lstrip('/')
        else:
            relative_path = s3_key.lstrip('/')
        
        local_file_path = os.path.join(dataset_path, relative_path)
        local_dir = os.path.dirname(local_file_path)
        
        # Create directory if needed
        if local_dir:
            os.makedirs(local_dir, exist_ok=True)
        
        # Skip if file already exists and has correct size
        if os.path.exists(local_file_path) and os.path.getsize(local_file_path) == file_size:
            downloaded_size += file_size
            downloaded_files += 1
            if i % 100 == 0:
                pct = (downloaded_size / total_size * 100) if total_size > 0 else 0
                mb_dl = downloaded_size / (1024**2)
                mb_total = total_size / (1024**2)
                print(f"Progress: {i}/{len(all_objects)} files "
                      f"({mb_dl:.1f}MB/{mb_total:.1f}MB, {pct:.1f}%)")
            continue
        
        # Download file with retry and verification
        max_retries = 3
        retry_count = 0
        download_success = False
        
        while retry_count < max_retries and not download_success:
            try:
                s3_client.download_file(s3_bucket, s3_key, local_file_path)
                
                # Verify download integrity by checking file size
                if os.path.exists(local_file_path):
                    actual_size = os.path.getsize(local_file_path)
                    if actual_size == file_size:
                        download_success = True
                        downloaded_size += file_size
                        downloaded_files += 1
                    else:
                        # File size mismatch - delete and retry
                        print(f"  ⚠ Size mismatch for {s3_key}: expected {file_size}, got {actual_size}. Retrying...")
                        if os.path.exists(local_file_path):
                            os.remove(local_file_path)
                        retry_count += 1
                        if retry_count >= max_retries:
                            print(f"  ✗ Failed to download {s3_key} after {max_retries} retries")
                            continue
                else:
                    retry_count += 1
                    if retry_count >= max_retries:
                        print(f"  ✗ File not created after download: {s3_key}")
                        continue
                
                if i % 100 == 0 or i == len(all_objects):
                    pct = (downloaded_size / total_size * 100) if total_size > 0 else 0
                    mb_dl = downloaded_size / (1024**2)
                    mb_total = total_size / (1024**2)
                    print(f"Progress: {i}/{len(all_objects)} files "
                          f"({mb_dl:.1f}MB/{mb_total:.1f}MB, {pct:.1f}%)")
            
            except ClientError as e:
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"  ✗ Error downloading {s3_key} after {max_retries} retries: {e}")
                    # Continue with next file
                    continue
                else:
                    print(f"  ⚠ Error downloading {s3_key} (retry {retry_count}/{max_retries}): {e}")
                    # Clean up partial download if it exists
                    if os.path.exists(local_file_path):
                        os.remove(local_file_path)
    
    # Create done marker
    Path(done_marker).touch()
    
    # Commit to volume
    signaltrain_volume.commit()
    
    elapsed = time.time() - start_time
    print(f"\n✓ Dataset sync complete!")
    print(f"  Downloaded: {downloaded_files} files")
    print(f"  Total size: {total_size_gb:.2f}GB")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Throughput: {total_size_gb / (elapsed / 60):.2f} GB/min")
    print(f"  Dataset available at: {dataset_path}")


@app.function(
    image=image,
    volumes={DATASET_MOUNT: signaltrain_volume},
    timeout=60 * 60 * 2,  # 2 hours
)
def check_dataset_status(target_subdir: str = ""):
    """Check if dataset is synced and show status.
    
    Args:
        target_subdir: Optional subdirectory name under DATASET_MOUNT (e.g., "SignalTrain_LA2A_Dataset_1.1" or "" for root)
    """
    if target_subdir:
        dataset_path = f"{DATASET_MOUNT}/{target_subdir}"
    else:
        dataset_path = DATASET_MOUNT
    
    done_marker = f"{dataset_path}/.dataset_synced"
    
    if not os.path.exists(done_marker):
        print("Dataset not synced yet.")
        return False
    
    if not os.path.exists(dataset_path):
        print(f"Dataset marker exists but path {dataset_path} not found.")
        return False
    
    # Count files
    total_files = 0
    total_size = 0
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                total_size += os.path.getsize(file_path)
                total_files += 1
            except OSError:
                pass
    
    print(f"✓ Dataset is synced")
    print(f"  Path: {dataset_path}")
    print(f"  Files: {total_files}")
    print(f"  Size: {total_size / (1024**3):.2f}GB")
    
    # Check for expected directories
    expected_dirs = ['Train', 'Val']
    for dir_name in expected_dirs:
        dir_path = os.path.join(dataset_path, dir_name)
        if os.path.exists(dir_path):
            files_in_dir = len(list(Path(dir_path).glob('*')))
            print(f"  {dir_name}/: {files_in_dir} files")
        else:
            print(f"  ⚠ {dir_name}/: not found")
    
    return True


@app.function(
    image=image,
    volumes={DATASET_MOUNT: signaltrain_volume},
    timeout=60 * 60 * 1,  # 1 hour
)
def test_dataloader_behavior(root_dir: str, subset: str = "Train"):
    """Test the exact dataloader behavior to reproduce the error."""
    from pathlib import Path
    import soundfile as sf
    import re
    
    root_path = Path(root_dir)
    subset_dir = root_path / subset
    
    print(f"Root dir: {root_path}")
    print(f"Subset dir: {subset_dir}")
    print(f"Subset dir exists: {subset_dir.exists()}\n")
    
    if not subset_dir.exists():
        print(f"❌ Subset directory not found: {subset_dir}")
        return
    
    # EXACT code from dataloader.py _find_paired_files
    input_pattern = re.compile(r"input_(\d+)_\.wav")
    target_pattern = re.compile(r"target_(\d+)_LA2A_([^_]+)__([^_]+)__([^_]+)\.wav")
    
    input_files = {}
    target_files = {}
    
    print("Finding input files...")
    for file_path in subset_dir.glob("input_*.wav"):
        match = input_pattern.match(file_path.name)
        if match:
            num = int(match.group(1))
            input_files[num] = file_path
    
    print("Finding target files...")
    for file_path in subset_dir.glob("target_*_LA2A_*.wav"):
        match = target_pattern.match(file_path.name)
        if match:
            num = int(match.group(1))
            s1, s2, s3 = match.groups()[1:]
            target_files[num] = (file_path, (s1, s2, s3))
    
    print(f"\nFound {len(input_files)} input files and {len(target_files)} target files")
    print(f"Common numbers: {sorted(set(input_files.keys()) & set(target_files.keys()))}")
    
    print(f"\nTesting pairing logic (EXACT dataloader code)...")
    
    # EXACT code from dataloader lines 144-165
    pairs = []
    for num in sorted(set(input_files.keys()) & set(target_files.keys())):
        input_path = input_files[num]
        target_path, states = target_files[num]
        
        print(f"\n{'='*60}")
        print(f"Processing pair {num}:")
        print(f"  Input: {input_path.name}")
        print(f"  Target: {target_path.name}")
        
        try:
            # EXACT code from dataloader line 150
            with sf.SoundFile(str(input_path)) as f:
                sample_rate = f.samplerate
                frames = len(f)
                duration = frames / sample_rate
                print(f"  ✓ Input file opened successfully")
                print(f"    Sample rate: {sample_rate}, Frames: {frames}")
                
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
            print(f"  ❌ LibsndfileError on input_{num}_.wav: {e}")
            print(f"    Error code: {e.error}")
            print(f"    This is the error from dataloader!")
            return
        except Exception as e:
            print(f"  ❌ Unexpected error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return
    
    print(f"\n{'='*60}")
    print(f"✓ Successfully processed {len(pairs)} pairs!")


@app.function(
    image=image,
    volumes={DATASET_MOUNT: signaltrain_volume},
    timeout=60 * 60 * 1,  # 1 hour
)
def check_file_on_modal(file_path: str):
    """Check a specific file on Modal volume for integrity issues."""
    import soundfile as sf
    
    print(f"Checking file: {file_path}")
    print("="*60)
    
    if not os.path.exists(file_path):
        print(f"❌ File does not exist")
        return
    
    # Check file size
    file_size = os.path.getsize(file_path)
    print(f"File size: {file_size} bytes ({file_size / (1024**2):.2f} MB)")
    
    # Check file header
    with open(file_path, 'rb') as f:
        header = f.read(12)
        print(f"File header (hex): {header.hex()}")
        is_wav = (header[:4] == b'RIFF' and len(header) >= 12 and header[8:12] == b'WAVE')
        print(f"Valid WAV signature: {is_wav}")
        if not is_wav:
            print(f"  Expected: RIFF...WAVE")
            print(f"  Got: {header[:4]}...{header[8:12] if len(header) >= 12 else 'N/A'}")
    
    # Try to read with soundfile
    try:
        with sf.SoundFile(file_path) as f:
            print(f"✓ File opened successfully with soundfile")
            print(f"  Format: {f.format}")
            print(f"  Subtype: {f.subtype}")
            print(f"  Sample rate: {f.samplerate} Hz")
            print(f"  Channels: {f.channels}")
            print(f"  Frames: {f.frames}")
            print(f"  Duration: {f.frames / f.samplerate:.2f} seconds")
    except sf.LibsndfileError as e:
        print(f"❌ LibsndfileError: {e}")
        print(f"  Error code: {e.error}")
        print(f"  This suggests the file is corrupted or incomplete")
    except Exception as e:
        print(f"❌ Unexpected error: {type(e).__name__}: {e}")


@app.function(
    image=image,
    volumes={DATASET_MOUNT: signaltrain_volume},
    timeout=60 * 60 * 1,  # 1 hour
)
def list_dataset_structure():
    """List the directory structure to help diagnose dataset path issues."""
    print("="*60)
    print("Dataset Directory Structure")
    print("="*60)
    print(f"\nBase mount: {DATASET_MOUNT}")
    
    if not os.path.exists(DATASET_MOUNT):
        print(f"⚠ Base mount does not exist: {DATASET_MOUNT}")
        return
    
    print(f"\nContents of {DATASET_MOUNT}:")
    try:
        items = sorted(os.listdir(DATASET_MOUNT))
        for item in items:
            item_path = os.path.join(DATASET_MOUNT, item)
            if os.path.isdir(item_path):
                print(f"  📁 {item}/")
                # Check if Train/Val are in this directory
                for subset in ['Train', 'Val', 'Test']:
                    subset_path = os.path.join(item_path, subset)
                    if os.path.exists(subset_path):
                        file_count = len(list(Path(subset_path).glob('*')))
                        print(f"      └─ {subset}/ ({file_count} items)")
            else:
                print(f"  📄 {item}")
    except Exception as e:
        print(f"Error listing directory: {e}")
    
    # Check for common dataset paths
    print("\n" + "="*60)
    print("Checking common dataset paths:")
    print("="*60)
    
    common_paths = [
        DATASET_MOUNT,
        f"{DATASET_MOUNT}/SignalTrain_LA2A_Dataset_1.1",
        f"{DATASET_MOUNT}/signaltrain",
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            train_path = os.path.join(path, "Train")
            val_path = os.path.join(path, "Val")
            has_train = os.path.exists(train_path)
            has_val = os.path.exists(val_path)
            
            status = "✓" if (has_train and has_val) else "⚠"
            print(f"\n{status} {path}")
            if has_train:
                train_files = len(list(Path(train_path).glob('*')))
                print(f"    Train/: {train_files} files")
            else:
                print(f"    Train/: not found")
            if has_val:
                val_files = len(list(Path(val_path).glob('*')))
                print(f"    Val/: {val_files} files")
            else:
                print(f"    Val/: not found")
            
            if has_train and has_val:
                print(f"\n  → Use this path in config: root_dir: \"{path}\"")


@app.local_entrypoint()
def sync_dataset(
    s3_bucket: str,
    role_arn: str,
    s3_prefix: str = "",
    target_subdir: str = "",
    force: bool = False,
):
    """Local entrypoint to sync dataset from S3 using OIDC authentication.
    
    Usage:
        # Sync dataset from S3 (if S3 structure is: bucket/prefix/Train/, bucket/prefix/Val/):
        modal run modal_app.py::sync_dataset --s3-bucket s3-signaltrain --role-arn arn:aws:iam::ACCOUNT:role/ModalOIDCRole --s3-prefix datasets/signaltrain/
        
        # Sync with subdirectory (if S3 structure includes dataset name):
        modal run modal_app.py::sync_dataset --s3-bucket s3-signaltrain --role-arn arn:aws:iam::ACCOUNT:role/ModalOIDCRole --s3-prefix datasets/ --target-subdir SignalTrain_LA2A_Dataset_1.1
        
        # Force re-sync:
        modal run modal_app.py::sync_dataset --s3-bucket s3-signaltrain --role-arn arn:aws:iam::ACCOUNT:role/ModalOIDCRole --s3-prefix datasets/signaltrain/ --force
        
    Note: Uses Modal OIDC authentication - no AWS credentials secret needed.
    The MODAL_IDENTITY_TOKEN is automatically provided by Modal.
    """
    print("="*60)
    print("SignalTrain Dataset Sync from S3 (OIDC)")
    print("="*60 + "\n")
    
    sync_dataset_from_s3.remote(s3_bucket, role_arn, s3_prefix, target_subdir, force)


@app.local_entrypoint()
def check_status(target_subdir: str = ""):
    """Check dataset sync status.
    
    Usage:
        modal run modal_app.py::check_status
        modal run modal_app.py::check_status --target-subdir SignalTrain_LA2A_Dataset_1.1
    """
    check_dataset_status.remote(target_subdir)


@app.local_entrypoint()
def list_structure():
    """List dataset directory structure to help diagnose path issues.
    
    Usage:
        modal run modal_app.py::list_structure
    """
    list_dataset_structure.remote()


@app.local_entrypoint()
def check_file(file_path: str = "/data/signaltrain/SignalTrain_LA2A_Dataset_1.1/Train/input_146_.wav"):
    """Check a specific file on Modal volume for integrity.
    
    Usage:
        modal run modal_app.py::check_file --file-path /data/signaltrain/SignalTrain_LA2A_Dataset_1.1/Train/input_146_.wav
    """
    check_file_on_modal.remote(file_path)


@app.local_entrypoint()
def test_dataloader(root_dir: str = "/data/signaltrain/SignalTrain_LA2A_Dataset_1.1", subset: str = "Train"):
    """Test the exact dataloader behavior to reproduce the error.
    
    Usage:
        modal run modal_app.py::test_dataloader --root-dir /data/signaltrain/SignalTrain_LA2A_Dataset_1.1 --subset Train
    """
    test_dataloader_behavior.remote(root_dir, subset)


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 8,  # up to 8 hours
    volumes={DATASET_MOUNT: signaltrain_volume},
    secrets=[
        modal.Secret.from_name("wandb")  # Expects WANDB_API_KEY
    ],
)
def run_training(config_path: str = f"{WORKDIR}/config.modal.default.yaml", resume: str | None = None):
    """Run training on Modal using PyTorch Lightning with full features from train.py."""
    import sys
    from typing import Optional
    
    sys.path.insert(0, WORKDIR)
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        LearningRateMonitor,
    )
    from pytorch_lightning.loggers import WandbLogger
    
    from dataloader import (
        DatasetConfig as DataConfigForLoader,
        create_dataloaders,
        compute_audio_metrics,
    )
    from config import load_config_from_yaml, ExperimentConfig
    from train import LSTMAudioModel as BaseLSTMAudioModel
    
    print("=" * 80)
    print("TRAINING ON MODAL WITH PYTORCH LIGHTNING")
    print("=" * 80)
    print(f"Config path: {config_path}")
    print(f"Resume from: {resume if resume else 'None'}")
    print("=" * 80)
    
    # Load configuration from YAML
    config: ExperimentConfig = load_config_from_yaml(config_path)
    
    # Set random seed
    pl.seed_everything(config.seed, workers=True)
    
    # Data configuration uses the DatasetConfig defined in dataloader.py
    data_config = DataConfigForLoader(
        root_dir=config.dataset.root_dir,
        train_subset=config.dataset.train_subset,
        val_subset=config.dataset.val_subset,
        test_subset=config.dataset.test_subset,
        train_length=config.dataset.train_length,
        eval_length=config.dataset.eval_length,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
        n_params=config.dataset.n_params,
        preload=config.dataset.preload,
        half_precision=config.dataset.half_precision,
        shuffle=config.dataset.shuffle,
        pin_memory=config.dataset.pin_memory,
    )
    
    print("\nCreating dataloaders...")
    train_loader, val_loader, _ = create_dataloaders(data_config)
    print("✓ Dataloaders created")
    
    # Define LSTM model using the model from train.py wrapped in Lightning
    class LSTMAudioModel(pl.LightningModule):
        """
        PyTorch Lightning wrapper for LSTM-based audio-to-audio regression model.
        
        Uses the LSTMAudioModel from train.py as the base model.
        
        Inputs:
            - input_audio: (B, 1, T)
            - params:      (B, P) optional conditioning parameters
        Output:
            - pred_audio:  (B, 1, T)
        """
        
        def __init__(
            self,
            input_length: int,
            n_params: int = 2,
            hidden_size: int = 256,
            num_layers: int = 2,
            dropout: float = 0.1,
            lr: float = 1e-4,
            weight_decay: float = 0.0,
            use_params: bool = True,
        ):
            super().__init__()
            self.save_hyperparameters()
            
            # Use the model from train.py
            self.model = BaseLSTMAudioModel(
                input_length=input_length,
                n_params=n_params,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                use_params=use_params,
            )
        
        def forward(self, x: torch.Tensor, params: Optional[torch.Tensor] = None) -> torch.Tensor:
            """
            Args:
                x:      (B, 1, T) input audio
                params: (B, P) parameter vector (optional)
            Returns:
                (B, 1, T) predicted audio
            """
            return self.model(x, params)
        
        def _shared_step(self, batch, stage: str):
            input_audio, target_audio, params = batch  # (B,1,T), (B,1,T), (B,P)
            
            pred_audio = self(input_audio, params)
            loss = F.mse_loss(pred_audio, target_audio)
            
            # Basic metrics
            mae = F.l1_loss(pred_audio, target_audio)
            
            # Audio metrics on first element to keep it cheap
            with torch.no_grad():
                metrics = compute_audio_metrics(pred_audio[0], target_audio[0])
            
            log_dict = {
                f"{stage}/loss_mse": loss,
                f"{stage}/mae": mae,
                f"{stage}/mse": metrics["mse"],
                f"{stage}/snr": metrics["snr"],
                f"{stage}/correlation": metrics["correlation"],
            }
            
            self.log_dict(
                log_dict,
                prog_bar=(stage == "train"),
                on_step=(stage == "train"),
                on_epoch=True,
            )
            return loss
        
        def training_step(self, batch, batch_idx):
            return self._shared_step(batch, "train")
        
        def validation_step(self, batch, batch_idx):
            self._shared_step(batch, "val")
        
        def configure_optimizers(self):
            lr = self.hparams.lr
            weight_decay = self.hparams.weight_decay
            scheduler_patience = getattr(self.hparams, "scheduler_patience", 5)
            scheduler_factor = getattr(self.hparams, "scheduler_factor", 0.5)
            
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=scheduler_factor,
                patience=scheduler_patience,
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss_mse",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
    
    # Create model from config
    print("\nCreating model...")
    model = LSTMAudioModel(
        input_length=config.model.input_length,
        n_params=config.model.n_params,
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        use_params=config.model.use_params,
    )
    # Store scheduler config in model for configure_optimizers
    model.hparams.scheduler_patience = config.training.scheduler_patience
    model.hparams.scheduler_factor = config.training.scheduler_factor
    print(f"✓ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # WandB logger
    print("\nSetting up WandB logger...")
    wandb_logger = WandbLogger(
        project=config.project,
        name=config.run_name or config.experiment_name,
        entity=config.wandb_entity,
        save_dir=config.log_dir,
        log_model=False,
        tags=config.tags,
        offline=True,
    )
    print("✓ WandB logger configured")
    
    # Callbacks
    print("\nSetting up callbacks...")
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        filename="lstm-audio-{epoch:02d}-{val_loss_mse:.5f}",
        save_top_k=3,
        monitor="val/loss_mse",
        mode="min",
        save_last=True,
    )
    
    early_stopping = EarlyStopping(
        monitor="val/loss_mse",
        mode="min",
        patience=config.training.early_stopping_patience,
        verbose=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    print("✓ Callbacks configured")
    
    # Build trainer from config
    print("\nSetting up trainer...")
    trainer_kwargs = {
        "max_epochs": config.training.num_epochs,
        "logger": wandb_logger,
        "callbacks": [checkpoint_callback, early_stopping, lr_monitor],
        "accelerator": config.training.accelerator,
        "devices": config.training.devices,
        "precision": config.training.precision,
        "log_every_n_steps": config.training.log_every_n_steps,
        "deterministic": config.deterministic,
    }
    
    if config.training.gradient_clip_norm is not None:
        trainer_kwargs["gradient_clip_val"] = config.training.gradient_clip_norm
    
    trainer = pl.Trainer(**trainer_kwargs)
    print("✓ Trainer configured")
    
    # Resume from checkpoint if provided
    ckpt_path = resume if resume else None
    
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path,
    )
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print(f"Last model saved at: {checkpoint_callback.last_model_path}")


@app.local_entrypoint()
def train(
    config_path: str = f"{WORKDIR}/config.modal.default.yaml",
    resume: str | None = None,
):
    """Local entrypoint to run training on Modal.
    
    Usage:
        # Run training with default config:
        modal run modal_app.py::train
        
        # Run training with custom config:
        modal run modal_app.py::train --config-path /workspace/config.modal.default.yaml
        
        # Resume from checkpoint:
        modal run modal_app.py::train --resume /path/to/checkpoint.ckpt
        
        # Both custom config and resume:
        modal run modal_app.py::train --config-path /workspace/config.modal.default.yaml --resume /path/to/checkpoint.ckpt
    """
    run_training.remote(config_path, resume)
