import logging as L
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PosixPath
from typing import Optional

import modal
from pydantic import BaseModel

MINUTES = 60  # seconds
HOURS = 60 * MINUTES

app_name = "neural-profiler-test"
app = modal.App(app_name)

gpu = "A10G"

volume = modal.Volume.from_name("neural-profiler-volume", create_if_missing=True)
volume_path = PosixPath("/vol/data")
model_filename = "lstm_profiler.pt"
best_model_filename = "best_lstm_profiler.pt"
tb_log_path = volume_path / "tb_logs"
model_save_path = volume_path / "models"

base_image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "pydantic==2.9.1"
)

torch_image = base_image.uv_pip_install(
    "torch==2.1.2",
    "tensorboard==2.17.1",
    "numpy<2",
)

torch_image = torch_image.add_local_dir(
    Path(__file__).parent, remote_path="/root/src"
)

with torch_image.imports():
    import glob
    import os
    from timeit import default_timer as timer

    # import tensorboard
    import torch
    # from src.dataset import Dataset
    # from src.logs_manager import LogsManager
    # from src.model import AttentionModel
    # from src.tokenizer import Tokenizer
