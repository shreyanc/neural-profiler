# modal_app.py
"""Modal app to train the LSTM model on SignalTrain using Modal GPUs.

Inspired by https://modal.com/docs/examples/hp_sweep_gpt; this app mounts the
project code, provisions a persistent volume for the SignalTrain dataset, and
wraps the training entrypoint defined in `train.py`.
"""

import asyncio
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


async def _upload_dataset_async(local_dataset_dir: str):
    """Async helper to upload dataset."""
    src = Path(local_dataset_dir).expanduser()
    if not src.exists():
        raise FileNotFoundError(f"Dataset not found at {src}")
    
    # Upload directory via batch_upload (Modal API 1.3) - async context manager
    print(f"Uploading {src} to Modal volume 'signaltrain-dataset'...")
    async with signaltrain_volume.batch_upload(force=True) as batch:
        batch.put_directory(str(src), "/")
    print(f"Upload complete! Dataset is now available at {DATASET_MOUNT} in Modal functions.")


@app.local_entrypoint()
def upload_dataset(local_dataset_dir: str = "/home/shreyan/Documents/DATASETS/SignalTrain_LA2A_Dataset_1.1"):
    """Local entrypoint to upload dataset to Modal volume.
    
    Usage:
        modal run modal_app.py::upload_dataset --local-dataset-dir /path/to/dataset
    """
    asyncio.run(_upload_dataset_async(local_dataset_dir))


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 8,  # up to 8 hours
    volumes={DATASET_MOUNT: signaltrain_volume},
)
def run(config_path: str = f"{WORKDIR}/config.modal.default.yaml", resume: str | None = None):
    """Kick off training on Modal using the provided YAML config."""
    import os
    import subprocess

    os.chdir(WORKDIR)
    cmd = ["python", "train.py", "--config", config_path]
    if resume:
        cmd += ["--resume", resume]

    subprocess.check_call(cmd)