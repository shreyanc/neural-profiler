"""
Minimal configuration management for LSTM-based training on the
SignalTrain LA2A dataset, using YAML experiment configs.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path

import yaml


@dataclass
class DatasetConfig:
    """Dataset parameters for SignalTrain LA2A."""

    # Dataset paths
    root_dir: str
    train_subset: str = "Train"
    val_subset: str = "Val"
    test_subset: str = "Test"

    # Audio parameters
    train_length: int = 65536  # ~1.5 seconds at 44.1kHz
    eval_length: int = 131072  # ~3 seconds at 44.1kHz
    sample_rate: int = 44100
    n_params: int = 2  # LA2A has 2 parameters: gain and ratio

    # Data loading parameters
    preload: bool = False
    half_precision: bool = False
    shuffle: bool = True
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True

    def __post_init__(self):
        """Basic validation."""
        if not Path(self.root_dir).exists():
            raise FileNotFoundError(f"Dataset root directory not found: {self.root_dir}")

        if self.train_length <= 0 or self.eval_length <= 0:
            raise ValueError("Audio length must be positive")

        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")

        if self.num_workers < 0:
            raise ValueError("Number of workers cannot be negative")


@dataclass
class ModelConfig:
    """LSTM model parameters."""

    model_type: str = "lstm"
    input_length: int = 65536
    n_params: int = 2

    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    use_params: bool = True


@dataclass
class TrainingConfig:
    """Training parameters matching the LSTM trainer in `train.py`."""

    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    optimizer: str = "adamw"
    scheduler: str = "reduce_on_plateau"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    early_stopping_patience: int = 20

    log_every_n_steps: int = 10
    precision: str = "32"
    accelerator: str = "auto"
    devices: int = 1
    gradient_clip_norm: Optional[float] = None


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig

    # Experiment metadata
    experiment_name: str = "signaltrain_lstm_experiment"
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # W&B logging
    project: str = "neural-profiler"
    run_name: Optional[str] = None
    wandb_entity: Optional[str] = None

    # Output paths
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    # Reproducibility
    seed: int = 42
    deterministic: bool = True

    def __post_init__(self):
        # Create output directories
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        # Align model with dataset
        self.model.input_length = self.dataset.train_length
        self.model.n_params = self.dataset.n_params


def load_config_from_yaml(config_path: str) -> ExperimentConfig:
    """Load experiment configuration from a YAML file."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return create_config_from_dict(config_dict)


def create_config_from_dict(config_dict: Dict[str, Any]) -> ExperimentConfig:
    """Create configuration from a Python dictionary."""
    dataset_cfg = DatasetConfig(**config_dict.get("dataset", {}))
    model_cfg = ModelConfig(**config_dict.get("model", {}))
    training_cfg = TrainingConfig(**config_dict.get("training", {}))

    # Extra top-level keys (experiment_name, project, etc.) are passed directly
    extra_keys = {
        k: v
        for k, v in config_dict.items()
        if k not in {"dataset", "model", "training"}
    }

    return ExperimentConfig(
        dataset=dataset_cfg,
        model=model_cfg,
        training=training_cfg,
        **extra_keys,
    )


def save_config_to_yaml(config: ExperimentConfig, config_path: str) -> None:
    """Serialize an ExperimentConfig to YAML."""
    cfg = {
        "dataset": config.dataset.__dict__,
        "model": config.model.__dict__,
        "training": config.training.__dict__,
        "experiment_name": config.experiment_name,
        "description": config.description,
        "tags": config.tags,
        "project": config.project,
        "run_name": config.run_name,
        "wandb_entity": config.wandb_entity,
        "output_dir": config.output_dir,
        "checkpoint_dir": config.checkpoint_dir,
        "log_dir": config.log_dir,
        "seed": config.seed,
        "deterministic": config.deterministic,
    }

    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, indent=2)


EXAMPLE_CONFIG_YAML = """
dataset:
  root_dir: "/path/to/signaltrain/la2a/dataset"
  train_subset: "Train"
  val_subset: "Val"
  test_subset: "Test"
  train_length: 65536
  eval_length: 131072
  sample_rate: 44100
  n_params: 2
  batch_size: 8
  num_workers: 4

model:
  model_type: "lstm"
  hidden_size: 256
  num_layers: 2
  dropout: 0.1
  use_params: true

training:
  num_epochs: 100
  learning_rate: 1.0e-4
  weight_decay: 1.0e-5
  precision: "32"
  accelerator: "auto"
  devices: 1

experiment_name: "signaltrain_lstm_experiment"
project: "neural-profiler"
output_dir: "./outputs"
checkpoint_dir: "./checkpoints"
log_dir: "./logs"
seed: 42
deterministic: true
"""

