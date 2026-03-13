#!/usr/bin/env python3
"""Step-by-step training script for Modal.

This script will be built incrementally, starting with basic Modal connection
and dataset verification.
"""

import modal
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pathlib import Path

APP_NAME = "train-step-by-step"
WORKDIR = "/workspace"
DATASET_MOUNT = "/data/signaltrain"

# Persistent volume for the SignalTrain dataset
signaltrain_volume = modal.Volume.from_name(
    "signaltrain-dataset", create_if_missing=False
)

# Base image with dependencies for PyTorch Lightning and dataloader
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        # Core dependencies for dataloader
        "numpy>=2.3.3",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "scipy>=1.11.0",
        "pyyaml>=6.0.0",
        # Torch stack
        "torch>=2.1.2",
        "torchvision>=0.16.1",
        # PyTorch Lightning
        "pytorch-lightning>=2.0.0",
        # Plotting
        "matplotlib>=3.10.0",
        # Experiment tracking (optional: set MLFLOW_TRACKING_URI to enable)
        "mlflow>=2.0.0",
        # Audio-focused loss functions (STFT, ESR, DC, etc.)
        "auraloss[all]>=0.4.0",
        # Loudness metrics (for LUFS-based evaluation)
        "pyloudnorm>=0.1.0",
    )
    # Include project code so dataloader and configs can be imported
    .add_local_dir(Path(__file__).parent, remote_path=WORKDIR)
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    volumes={DATASET_MOUNT: signaltrain_volume},
    timeout=60 * 10,  # 10 minutes
)
def debug_one_step(
    root_dir: str = "/data/signaltrain/SignalTrain_LA2A_Dataset_1.1",
    train_subset: str = "Train",
    train_length: int = 65536,
    batch_size: int = 4,
    num_workers: int = 2,
    n_params: int = 2,
):
    """
    Load dataloaders and run a single forward step through TestLSTM,
    printing tensor shapes along the way.
    """
    import sys

    # Ensure project code is on PYTHONPATH inside the container
    sys.path.insert(0, WORKDIR)

    from dataloader import DatasetConfig as DataConfigForLoader, create_dataloaders
    from models import TestLSTM

    print("=" * 80)
    print("DEBUG ONE STEP: DATALOADER + TestLSTM")
    print("=" * 80)
    print(f"Dataset root: {root_dir}")
    print(f"Train subset: {train_subset}")
    print(f"Train length: {train_length}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"n_params: {n_params}")
    print("=" * 80)

    # Build data config and loaders
    data_config = DataConfigForLoader(
        root_dir=root_dir,
        train_subset=train_subset,
        val_subset="Val",
        test_subset="Test",
        train_length=train_length,
        eval_length=train_length,
        batch_size=batch_size,
        num_workers=num_workers,
        n_params=n_params,
        preload=False,
        half_precision=False,
        shuffle=True,
        pin_memory=True,
    )

    print("\nCreating dataloaders...")
    train_loader, val_loader, _ = create_dataloaders(data_config)
    print("✓ Dataloaders created")

    # Get a single batch
    print("\nFetching one batch from train_loader...")
    batch = next(iter(train_loader))
    input_audio, target_audio, params = batch

    print(f"Input audio shape:  {tuple(input_audio.shape)}  # expected (B, 1, T)")
    print(f"Target audio shape: {tuple(target_audio.shape)} # expected (B, 1, T)")
    print(f"Params shape:       {tuple(params.shape)}       # expected (B, P)")

    # Create model and run one forward step
    model = TestLSTM()
    print("\nRunning TestLSTM forward pass on one batch...")
    output_audio = model(input_audio)

    print(f"Output audio shape: {tuple(output_audio.shape)} # expected (B, 1, T)")

    return {
        "input_shape": tuple(input_audio.shape),
        "target_shape": tuple(target_audio.shape),
        "params_shape": tuple(params.shape),
        "output_shape": tuple(output_audio.shape),
    }


def create_model_from_config(model_config: dict, n_params: int):
    """
    Create a model instance based on configuration.

    model_config must contain "model_type" and the hyperparameters for that
    model (from the corresponding section in the YAML; no common keys).

    Prints the hyperparameters used so you can confirm correct values.
    """
    from models import (
        TestLSTM,
        ResidualLSTM,
        SimpleHardwareEmulationLSTM,
        HardwareEmulationLSTM,
        ReferenceLSTM,
        TCNModel,
        CausalBlockLSTM,
    )

    model_type = model_config.get("model_type", "TestLSTM").strip()

    def _print_kwargs(name: str, kwargs: dict) -> None:
        print(f"  {name} hyperparameters read from config:")
        for k, v in sorted(kwargs.items()):
            print(f"    {k}: {v}")

    if model_type == "TestLSTM":
        kwargs = {
            "hidden_size": model_config.get("hidden_size", 128),
            "num_layers": model_config.get("num_layers", 1),
            "dropout": model_config.get("dropout", 0.0),
            "learning_rate": model_config.get("learning_rate", 1e-4),
        }
        _print_kwargs("TestLSTM", kwargs)
        return TestLSTM(**kwargs)
    elif model_type == "ResidualLSTM":
        kwargs = {
            "n_params": n_params,
            "hidden_size": model_config.get("hidden_size", 128),
            "num_layers": model_config.get("num_layers", 1),
            "dropout": model_config.get("dropout", 0.0),
            "learning_rate": model_config.get("learning_rate", 1e-4),
        }
        _print_kwargs("ResidualLSTM", kwargs)
        return ResidualLSTM(**kwargs)
    elif model_type == "ReferenceLSTM":
        kwargs = {
            "n_params": n_params,
            "n_inputs": model_config.get("n_inputs", 1),
            "n_outputs": model_config.get("n_outputs", 1),
            "hidden_size": model_config.get("hidden_size", 32),
            "num_layers": model_config.get("num_layers", 1),
            "learning_rate": model_config.get("learning_rate", 1e-4),
        }
        _print_kwargs("ReferenceLSTM", kwargs)
        return ReferenceLSTM(**kwargs)
    elif model_type == "SimpleHardwareEmulationLSTM":
        kwargs = {
            "n_params": n_params,
            "hidden_size": model_config.get("hidden_size", 128),
            "num_layers": model_config.get("num_layers", 2),
            "dropout": model_config.get("dropout", 0.1),
            "learning_rate": model_config.get("learning_rate", 1e-4),
        }
        _print_kwargs("SimpleHardwareEmulationLSTM", kwargs)
        return SimpleHardwareEmulationLSTM(**kwargs)
    elif model_type == "HardwareEmulationLSTM":
        kwargs = {
            "n_params": n_params,
            "hidden_size": model_config.get("hidden_size", 256),
            "num_layers": model_config.get("num_layers", 3),
            "dropout": model_config.get("dropout", 0.1),
            "param_embed_dim": model_config.get("param_embed_dim", 64),
            "use_bidirectional": model_config.get("use_bidirectional", True),
            "use_residual": model_config.get("use_residual", True),
            "use_skip_connection": model_config.get("use_skip_connection", True),
            "learning_rate": model_config.get("learning_rate", 1e-4),
        }
        _print_kwargs("HardwareEmulationLSTM", kwargs)
        return HardwareEmulationLSTM(**kwargs)
    elif model_type == "TCNModel":
        kwargs = {
            "nparams": n_params,
            "ninputs": model_config.get("ninputs", 1),
            "noutputs": model_config.get("noutputs", 1),
            "nblocks": model_config.get("nblocks", 10),
            "kernel_size": model_config.get("kernel_size", 3),
            "dilation_growth": model_config.get("dilation_growth", 1),
            "channel_growth": model_config.get("channel_growth", 1),
            "channel_width": model_config.get("channel_width", 32),
            "stack_size": model_config.get("stack_size", 10),
            "grouped": model_config.get("grouped", False),
            "causal": model_config.get("causal", False),
            "skip_connections": model_config.get("skip_connections", False),
            "learning_rate": model_config.get("learning_rate", 1e-4),
        }
        _print_kwargs("TCNModel", kwargs)
        return TCNModel(**kwargs)
    elif model_type == "CausalBlockLSTM":
        kwargs = {
            "n_params": n_params,
            "hidden_size": model_config.get("hidden_size", 128),
            "num_layers": model_config.get("num_layers", 1),
            "dropout": model_config.get("dropout", 0.0),
            "block_size": model_config.get("block_size", 22050),
            "learning_rate": model_config.get("learning_rate", 1e-4),
        }
        _print_kwargs("CausalBlockLSTM", kwargs)
        return CausalBlockLSTM(**kwargs)
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. Supported: "
            f"'TestLSTM', 'ResidualLSTM', 'SimpleHardwareEmulationLSTM', "
            f"'HardwareEmulationLSTM', 'ReferenceLSTM', 'TCNModel', 'CausalBlockLSTM'"
        )


def _make_pre_filter(
    pre_emphasis: str,
    sample_rate: float,
    pre_emphasis_coeff: float = 0.95,
):
    """
    Build a pre_filter callable for use with esr_loss/dc_loss.

    Args:
        pre_emphasis: "none" | "high_pass" | "a_weighted"
        sample_rate: Sampling rate in Hz (used for a_weighted).
        pre_emphasis_coeff: Coefficient for high_pass (used with models.pre_emphasis_filter; default 0.95).

    Returns:
        Callable (tensor -> tensor) or None for "none".
    """
    from models import a_weighted_pre_emphasis_filter, pre_emphasis_filter

    if pre_emphasis is None or pre_emphasis.strip().lower() == "none":
        return None
    pre_emphasis = pre_emphasis.strip().lower()
    if pre_emphasis == "high_pass":
        coeff = float(pre_emphasis_coeff)
        return lambda x: pre_emphasis_filter(x, coeff=coeff)
    if pre_emphasis == "a_weighted":
        return lambda x: a_weighted_pre_emphasis_filter(x, sample_rate)
    raise ValueError(
        f"pre_emphasis must be 'none', 'high_pass', or 'a_weighted'; got {pre_emphasis!r}"
    )


def _model_forward_accepts_params(model):
    """Return True if model.forward has a second parameter (e.g. params)."""
    import inspect
    sig = inspect.signature(model.forward)
    params = list(sig.parameters.keys())
    return len(params) >= 2


_VALID_LOSS_TYPES = (
    "esr",
    "dc",
    "esr_dc",
    "mse",
    "mae",
    "huber",
    "stft",
    "mrstft",
    "l1_stft",
)


class LossConfigWrapper(pl.LightningModule):
    """
    Wraps a Lightning module and overrides training_step/validation_step to use
    configurable loss (esr, dc, esr_dc, mse, mae, huber, stft, mrstft, l1_stft) and optional pre_filter.
    When pre_filter is set, it is applied inside the wrapper to the signals before the selected loss.
    Logs a separate validation metric every validation step regardless of training loss.
    """

    def __init__(
        self,
        model: pl.LightningModule,
        loss_type: str,
        pre_filter=None,
        model_takes_params: bool = False,
        sample_rate: float = 44100,
        lr_scheduler_patience: int | None = None,
        lr_scheduler_factor: float = 0.1,
    ):
        super().__init__()
        import auraloss

        self.model = model
        self.loss_type = loss_type.strip().lower()
        self.pre_filter = pre_filter
        self._model_takes_params = model_takes_params
        self.sample_rate = sample_rate
        self._lr_scheduler_patience = lr_scheduler_patience
        self._lr_scheduler_factor = lr_scheduler_factor
        if self.loss_type not in _VALID_LOSS_TYPES:
            raise ValueError(
                f"loss_type must be one of {_VALID_LOSS_TYPES}; got {self.loss_type!r}"
            )
        # Auraloss time-domain losses (no pre_filter; applied in wrapper when configured)
        self._esr_loss_fn = auraloss.time.ESRLoss()
        self._dc_loss_fn = auraloss.time.DCLoss()
        # Auraloss frequency-domain losses
        self._stft_loss_fn = auraloss.freq.STFTLoss(
            fft_size=1024, hop_size=256, win_length=1024
        )
        self._mrstft_loss_fn = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[1024, 2048, 8192],
            hop_sizes=[256, 512, 2048],
            win_lengths=[1024, 2048, 8192],
            sample_rate=sample_rate,
        )

    def forward(self, x, params=None):
        if self._model_takes_params and params is not None:
            return self.model(x, params)
        return self.model(x)

    def _apply_pre_filter(self, pred_audio, target_audio):
        """Apply pre_filter when configured; return (pred, target) for the loss."""
        if self.pre_filter is not None:
            return self.pre_filter(pred_audio), self.pre_filter(target_audio)
        return pred_audio, target_audio

    def _compute_loss_for_type(self, pred_audio, target_audio, loss_type: str):
        """Compute loss for a given loss type (used for training loss and validation metric).
        Pre_filter is applied inside the wrapper to the signals before the selected loss.
        """
        p, t = self._apply_pre_filter(pred_audio, target_audio)

        if loss_type == "esr":
            return self._esr_loss_fn(p, t)
        if loss_type == "dc":
            return self._dc_loss_fn(p, t)
        if loss_type == "esr_dc":
            return self._esr_loss_fn(p, t) + self._dc_loss_fn(p, t)
        if loss_type == "mse":
            return F.mse_loss(p, t)
        if loss_type == "mae":
            return F.l1_loss(p, t)
        if loss_type == "huber":
            return F.smooth_l1_loss(p, t)
        if loss_type == "stft":
            return self._stft_loss_fn(p, t)
        if loss_type == "mrstft":
            return self._mrstft_loss_fn(p, t)
        if loss_type == "l1_stft":
            return F.l1_loss(p, t) + self._stft_loss_fn(p, t)
        raise RuntimeError(f"Unexpected loss_type: {loss_type!r}")

    def _compute_loss(self, pred_audio, target_audio):
        return self._compute_loss_for_type(pred_audio, target_audio, self.loss_type)

    def training_step(self, batch, batch_idx):
        input_audio, target_audio, params = batch
        pred_audio = self.forward(input_audio, params)
        loss = self._compute_loss(pred_audio, target_audio)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_audio, target_audio, params = batch
        pred_audio = self.forward(input_audio, params)
        loss = self._compute_loss(pred_audio, target_audio)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        result = self.model.configure_optimizers()
        # Optional ReduceLROnPlateau (paper: factor 0.1 after 10 epochs without val improvement)
        if self._lr_scheduler_patience is not None:
            if isinstance(result, torch.optim.Optimizer):
                optimizer = result
            elif isinstance(result, dict):
                optimizer = result["optimizer"]
            elif isinstance(result, (list, tuple)):
                optimizer = result[0]
            else:
                optimizer = result
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self._lr_scheduler_factor,
                patience=self._lr_scheduler_patience,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return result


@app.function(
    image=image,
    gpu="A10G",  # Use A10G GPU for training
    volumes={DATASET_MOUNT: signaltrain_volume},
    secrets=[modal.Secret.from_name("mlflow")],  # MLFLOW_* and DATABRICKS_* env vars
    timeout=60 * 60 * 2,  # 2 hours (increased for multiple epochs)
)
def train_model(
    config_path: str = None,
    # Legacy parameters (used if config_path is None)
    root_dir: str = "/data/signaltrain/SignalTrain_LA2A_Dataset_1.1",
    train_subset: str = "Train",
    train_length: int = 65536,
    batch_size: int = 4,
    num_workers: int = 2,
    n_params: int = 2,
    hidden_size: int = 128,
    learning_rate: float = 1e-4,
    num_epochs: int = 1,
    mlflow_registered_model_name: str = None,
):
    """
    Train model for multiple epochs using config file or individual parameters.
    
    Args:
        config_path: Path to YAML config file (if None, uses individual parameters)
        root_dir: Dataset root directory (used if config_path is None)
        train_subset: Training subset name (used if config_path is None)
        train_length: Audio length in samples (used if config_path is None)
        batch_size: Batch size (used if config_path is None)
        num_workers: Number of data loader workers (used if config_path is None)
        n_params: Number of hardware parameters (used if config_path is None)
        hidden_size: Model hidden size (used if config_path is None)
        learning_rate: Learning rate (used if config_path is None)
        num_epochs: Number of training epochs (used if config_path is None)
    """
    import os
    import sys
    import yaml
    from pathlib import Path

    # Ensure project code is on PYTHONPATH inside the container
    sys.path.insert(0, WORKDIR)

    from dataloader import DatasetConfig as DataConfigForLoader, create_dataloaders
    from run_utils import init_run_dir
    
    # Check GPU availability
    print("\n" + "=" * 80)
    print("GPU CHECK")
    print("=" * 80)
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✓ GPU available: {gpu_name}")
        print(f"  GPU count: {gpu_count}")
        print(f"  Current device: {torch.cuda.current_device()}")
        print(f"  Device name: {torch.cuda.get_device_name(0)}")
        
        # Disable cuDNN for RNN operations to avoid CUDNN_STATUS_NOT_SUPPORTED
        # This is necessary for very long sequences (e.g., 65536 samples)
        # cuDNN has limitations on sequence length, so we use PyTorch's native implementation
        torch.backends.cudnn.enabled = True
        # print(f"  cuDNN disabled for RNN operations (required for long sequences)")
        # Use Tensor Cores on supported GPUs (e.g. A10): trade precision for performance
        torch.set_float32_matmul_precision("medium")
    else:
        print("✗ GPU not available - training will run on CPU")
    print("=" * 80)
    
    # Load config from YAML if provided
    if config_path:
        config_path_full = Path(config_path)
        if not config_path_full.is_absolute():
            # If relative, try multiple locations
            # First try workspace directory
            workspace_config = Path(WORKDIR) / config_path
            # Also try current directory (for local testing)
            current_config = Path(config_path)
            
            if workspace_config.exists():
                config_path_full = workspace_config
            elif current_config.exists():
                config_path_full = current_config
            else:
                # Default to workspace
                config_path_full = workspace_config
        
        print(f"Loading config from: {config_path_full}")
        if not config_path_full.exists():
            raise FileNotFoundError(f"Config file not found: {config_path_full}")
        
        with open(config_path_full, "r") as f:
            config_dict = yaml.safe_load(f)
        
        # Extract config sections
        dataset_config = config_dict.get("dataset", {}) or {}
        model_config = config_dict.get("model", {}) or {}
        training_config = config_dict.get("training", {}) or {}
        experiment_config = {
            k: v for k, v in config_dict.items() if k not in ["dataset", "model", "training"]
        }

        # Use dataset / training values
        root_dir = dataset_config.get("root_dir", root_dir)
        train_subset = dataset_config.get("train_subset", train_subset)
        train_length = dataset_config.get("train_length", train_length)
        eval_length = dataset_config.get("eval_length", train_length)
        batch_size = dataset_config.get("batch_size", batch_size)
        num_workers = dataset_config.get("num_workers", num_workers)
        n_params = dataset_config.get("n_params", n_params)
        num_epochs = training_config.get("num_epochs", num_epochs)
        gradient_accumulation_steps = training_config.get("gradient_accumulation_steps", 1)
        precision = training_config.get("precision", "16-mixed")
        loss_type = training_config.get("loss", "mse")
        pre_emphasis = training_config.get("pre_emphasis", "none")
        pre_emphasis_coeff = training_config.get("pre_emphasis_coeff", 0.95)
        lr_scheduler_patience = training_config.get("lr_scheduler_patience")
        lr_scheduler_factor = training_config.get("lr_scheduler_factor", 0.1)
        sample_rate = dataset_config.get("sample_rate", 44100)

        # Determine model_type and use only that model's section (no common hyperparams).
        model_type = (model_config.get("model_type") or "TestLSTM").strip()

        _MODEL_TYPE_TO_SECTION = {
            "TestLSTM": "test_lstm",
            "ResidualLSTM": "residual_lstm",
            "ReferenceLSTM": "reference_lstm",
            "SimpleHardwareEmulationLSTM": "simple_hardware_emulation_lstm",
            "HardwareEmulationLSTM": "hardware_emulation_lstm",
            "TCNModel": "tcn",
            "CausalBlockLSTM": "causal_block_lstm",
        }
        section_key = _MODEL_TYPE_TO_SECTION.get(model_type)
        if section_key is None:
            raise ValueError(
                f"Unknown model_type: {model_type!r}. Supported: "
                f"{list(_MODEL_TYPE_TO_SECTION.keys())}"
            )
        section = model_config.get(section_key) or {}
        # model_config for create_model_from_config = only this model's hyperparams
        model_config = {"model_type": model_type, **dict(section)}

        # Learning rate: training config overrides model section
        learning_rate = training_config.get(
            "learning_rate", model_config.get("learning_rate", learning_rate)
        )
        # Ensure model gets the training LR (create_model_from_config reads from model_config)
        model_config["learning_rate"] = learning_rate

        mlflow_registered_model_name = experiment_config.get("mlflow_registered_model_name")
        
        print(f"✓ Config loaded: model_type={model_type}, num_epochs={num_epochs}, accum={gradient_accumulation_steps}, precision={precision}, loss={loss_type}, pre_emphasis={pre_emphasis}")
    else:
        # Use individual parameters (no config file)
        eval_length = train_length
        model_type = "TestLSTM"
        model_config = {
            "model_type": "TestLSTM",
            "hidden_size": hidden_size,
            "learning_rate": learning_rate,
        }
        precision = "16-mixed"
        gradient_accumulation_steps = 1
        loss_type = "mse"
        pre_emphasis = "none"
        pre_emphasis_coeff = 0.95
        lr_scheduler_patience = None
        lr_scheduler_factor = 0.1
        sample_rate = 44100
        # mlflow_registered_model_name from function arg (already set above)
        print("Using individual parameters (no config file)")

    # Run output dir: outputs/YYYY-MM-DD_HH-MM-SS_<model>_<key_hyperparams>/ (datetime first for ordering)
    run_info = init_run_dir(
        base_dir=Path(WORKDIR),
        model_type=model_type,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        train_length=train_length,
        model_config=model_config,
    )
    print(f"Output run dir: {run_info.output_run_dir}")

    # Optional: MLflow experiment tracking. Set MLFLOW_TRACKING_URI to enable;
    # for Databricks use MLFLOW_TRACKING_URI=databricks + DATABRICKS_HOST + DATABRICKS_TOKEN
    # (e.g. via Modal secret). See train_epoch() docstring for full setup.
    use_mlflow = bool(os.environ.get("MLFLOW_TRACKING_URI"))
    if use_mlflow:
        import mlflow
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(
            os.environ.get("MLFLOW_EXPERIMENT_NAME", "neural-profiler")
        )
        mlflow.start_run(run_name=run_info.run_dir_name)
        # Log run parameters
        mlflow.log_params({
            "model_type": model_type,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "train_length": train_length,
            "root_dir": root_dir,
            "train_subset": train_subset,
            "n_params": n_params,
            "num_workers": num_workers,
        })
        for k, v in (model_config or {}).items():
            if v is not None and k not in ("learning_rate",):
                mlflow.log_param(f"model_{k}", v)
        mlflow.log_param("loss", loss_type)
        mlflow.log_param("pre_emphasis", pre_emphasis)
        mlflow.log_param("precision", precision)
        print("✓ MLflow tracking enabled")

    print("=" * 80)
    print(f"TRAINING: {model_type} ({num_epochs} epoch{'s' if num_epochs > 1 else ''})")
    print("=" * 80)
    print(f"Dataset root: {root_dir}")
    print(f"Train subset: {train_subset}")
    print(f"Train length: {train_length}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"n_params: {n_params}")
    print(f"Learning rate: {learning_rate}")
    print(f"Num epochs: {num_epochs}")
    print("=" * 80)

    # Build data config and loaders
    data_config = DataConfigForLoader(
        root_dir=root_dir,
        train_subset=train_subset,
        val_subset="Val",
        test_subset="Test",
        train_length=train_length,
        eval_length=eval_length,
        batch_size=batch_size,
        num_workers=num_workers,
        n_params=n_params,
        preload=False,
        half_precision=False,
        shuffle=True,
        pin_memory=True,
    )

    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(data_config)
    print("✓ Dataloaders created")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    if test_loader is not None:
        print(f"  Test batches: {len(test_loader)}")

    # Create model from config
    print(f"\nCreating {model_type} model...")
    print(f"  n_params (from dataset config): {n_params}")
    print(f"  model_config passed to create_model_from_config: {model_config}")
    model = create_model_from_config(model_config, n_params)
    print(f"✓ Model created: {model_type}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,} total")

    # Wrap with configurable loss and validation metric (so val_metric is always reported)
    loss_type = str(loss_type).strip().lower() if loss_type else "mse"
    # Build pre_filter when pre_emphasis is configured; wrapper applies it to signals before any loss.
    pre_filter = None
    if pre_emphasis and str(pre_emphasis).strip().lower() not in ("", "none"):
        pre_filter = _make_pre_filter(pre_emphasis, sample_rate, pre_emphasis_coeff)
    model_takes_params = model_type in (
        "HardwareEmulationLSTM",
        "SimpleHardwareEmulationLSTM",
        "ResidualLSTM",
        "ReferenceLSTM",
        "TCNModel",
        "CausalBlockLSTM",
    )
    model = LossConfigWrapper(
        model=model,
        loss_type=loss_type,
        pre_filter=pre_filter,
        model_takes_params=model_takes_params,
        sample_rate=sample_rate,
        lr_scheduler_patience=lr_scheduler_patience,
        lr_scheduler_factor=lr_scheduler_factor,
    )
    sched_info = f", lr_scheduler=ReduceLROnPlateau(patience={lr_scheduler_patience}, factor={lr_scheduler_factor})" if lr_scheduler_patience is not None else ""
    print(
        f"✓ Loss wrapper: loss={loss_type}, pre_emphasis={pre_emphasis}{sched_info}"
    )

    # Callback for epoch-level summary (no per-batch printing; progress bar shows loss)
    class PrintBatchLossCallback(pl.Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            """Print epoch start."""
            current_epoch = trainer.current_epoch + 1
            print(f"\n--- Epoch {current_epoch}/{num_epochs} ---")

        def on_train_epoch_end(self, trainer, pl_module):
            """Print epoch end summary."""
            current_epoch = trainer.current_epoch + 1
            epoch_train_loss = trainer.callback_metrics.get('train_loss_epoch', None)
            val_loss = trainer.callback_metrics.get('val_loss', None)
            if epoch_train_loss is not None:
                print(f"  Epoch {current_epoch}/{num_epochs} train loss: {epoch_train_loss.item():.6f}")
            if val_loss is not None:
                print(f"  Epoch {current_epoch}/{num_epochs} val loss: {val_loss.item():.6f}")

    class MlflowLoggingCallback(pl.Callback):
        """Log train/val loss to MLflow at end of each epoch (if an MLflow run is active)."""
        def on_train_epoch_end(self, trainer, pl_module):
            try:
                import mlflow
                if mlflow.active_run() is None:
                    return
                step = trainer.current_epoch + 1
                train_loss = trainer.callback_metrics.get("train_loss_epoch")
                val_loss = trainer.callback_metrics.get("val_loss")
                if train_loss is not None:
                    mlflow.log_metric("train_loss", train_loss.item(), step=step)
                if val_loss is not None:
                    mlflow.log_metric("val_loss", val_loss.item(), step=step)
            except Exception:
                pass

    # Save checkpoint with lowest validation loss
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=str(run_info.output_run_dir),
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    callbacks_list = [PrintBatchLossCallback(), checkpoint_callback]
    if use_mlflow:
        callbacks_list.append(MlflowLoggingCallback())

    # Create trainer
    # Configure accelerator based on GPU availability
    accelerator = "gpu" if gpu_available else "cpu"
    devices = 1 if gpu_available else "auto"
    
    print(f"\nConfiguring PyTorch Lightning Trainer:")
    print(f"  Accelerator: {accelerator}")
    print(f"  Devices: {devices}")
    print(f"  Precision: {precision}")
    
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator=accelerator,
        devices=devices,
        accumulate_grad_batches=gradient_accumulation_steps,
        enable_model_summary=False,
        logger=None,
        log_every_n_steps=10,
        callbacks=callbacks_list,
        precision=precision,
    )

    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    
    # Train for specified number of epochs
    trainer.fit(model, train_loader, val_loader)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    
    # Get final metrics
    final_train_loss = trainer.callback_metrics.get('train_loss_epoch', None)
    # Prefer the best validation loss observed during training (from the checkpoint callback)
    best_val_metric = getattr(checkpoint_callback, "best_model_score", None)
    if best_val_metric is not None:
        final_val_loss = best_val_metric
    else:
        # Fallback to the last epoch's val_loss if no best score is available
        final_val_loss = trainer.callback_metrics.get('val_loss', None)

    if final_train_loss is not None:
        print(f"Final train loss: {final_train_loss.item():.6f}")
    if final_val_loss is not None:
        print(f"Final val loss (best model): {final_val_loss.item():.6f}")

    # Load best model (lowest val loss) for evaluation and plotting
    best_path = getattr(checkpoint_callback, "best_model_path", None)
    if best_path and Path(best_path).exists():
        ckpt = torch.load(best_path, map_location=model.device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        best_val = getattr(checkpoint_callback, "best_model_score", None)
        best_val_str = f"{best_val.item():.6f}" if hasattr(best_val, "item") else f"{best_val:.6f}"
        print(f"\nLoaded best model from {best_path} (val_loss={best_val_str})")
    else:
        print("\nUsing final epoch model for validation plot (no best checkpoint found).")

    # ------------------------------------------------------------------
    # Test-set evaluation (MAE, STFT, dB LUFS) using best model
    # Implemented following micro-tcn test metrics:
    # https://github.com/csteinmetz1/micro-tcn/blob/3e1067bcaf07e4ecea88ae16e55437024d1d7eb6/test.py#L149
    # For paper-comparable STFT/LUFS: use eval_length >= 65536 (~1.5 s); micro-tcn uses
    # 8388608. Short segments inflate LUFS (ITU-R BS.1770 is unreliable < few seconds).
    # ------------------------------------------------------------------
    test_metrics = None
    if test_loader is not None:
        import numpy as np
        import auraloss
        import pyloudnorm as pyln

        from models import causal_crop, center_crop

        print("\n" + "=" * 80)
        print("TEST EVALUATION (best model)")
        print("=" * 80)

        l1_fn = torch.nn.L1Loss()
        stft_fn = auraloss.freq.STFTLoss()
        meter = pyln.Meter(sample_rate)

        mae_scores = []
        stft_scores = []
        lufs_diff_scores = []

        # Align metrics with micro-tcn: crop input/target to output length (causal vs center)
        inner = getattr(model, "model", model)
        is_causal = getattr(
            getattr(inner, "hparams", None), "causal", False
        )

        model.eval()
        # Decide whether to pass params based on wrapped inner model type
        _models_taking_params = (
            "HardwareEmulationLSTM",
            "SimpleHardwareEmulationLSTM",
            "ResidualLSTM",
            "ReferenceLSTM",
            "TCNModel",
            "CausalBlockLSTM",
        )

        with torch.no_grad():
            for bidx, (input_audio, target_audio, params) in enumerate(test_loader):
                input_audio = input_audio.to(model.device)
                target_audio = target_audio.to(model.device)
                params = params.to(model.device)

                if model_takes_params or model_type in _models_taking_params:
                    output_audio = model(input_audio, params)
                else:
                    output_audio = model(input_audio, None)

                out_len = output_audio.shape[-1]
                if target_audio.shape[-1] != out_len:
                    if is_causal:
                        input_audio = causal_crop(input_audio, out_len)
                        target_audio = causal_crop(target_audio, out_len)
                    else:
                        input_audio = center_crop(input_audio, out_len)
                        target_audio = center_crop(target_audio, out_len)

                # Per-item metrics, mirroring micro-tcn style
                for i_tensor, o_tensor, t_tensor in zip(
                    torch.split(input_audio, 1, dim=0),
                    torch.split(output_audio, 1, dim=0),
                    torch.split(target_audio, 1, dim=0),
                ):
                    mae = l1_fn(o_tensor, t_tensor).item()
                    stft_val = stft_fn(o_tensor, t_tensor).item()

                    t_np = t_tensor.squeeze().detach().cpu().numpy()
                    o_np = o_tensor.squeeze().detach().cpu().numpy()
                    target_lufs = meter.integrated_loudness(t_np)
                    output_lufs = meter.integrated_loudness(o_np)
                    lufs_diff = float(abs(output_lufs - target_lufs))

                    mae_scores.append(mae)
                    stft_scores.append(stft_val)
                    lufs_diff_scores.append(lufs_diff)

        if mae_scores:
            mean_mae = float(np.mean(mae_scores))
            mean_stft = float(np.mean(stft_scores))
            mean_lufs_diff = float(np.mean(lufs_diff_scores))

            print("-" * 64)
            print("TEST METRICS (mean over test set)")
            print(f"MAE:        {mean_mae:0.4e}")
            print(f"STFT:       {mean_stft:0.4f}")
            print(f"dB LUFS Δ:  {mean_lufs_diff:0.4f}")

            test_metrics = {
                "test_mae": mean_mae,
                "test_stft": mean_stft,
                "test_lufs_db": mean_lufs_diff,
            }
        else:
            print("No test samples found for evaluation.")

    # Plot fixed test-set triplets for the specified file and segments,
    # and save corresponding audio (input, target, predicted) as artifacts.
    print(
        "\nPlotting fixed test-set triplets for input_259_.wav / "
        "target_259_LA2A_2c__1__80.wav at requested segments..."
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import soundfile as sf

    fig = None
    audio_artifact_paths = []

    if test_loader is None:
        print("No test_loader available; skipping fixed test-set triplet plotting.")
    else:
        # Locate the test dataset pair for file index 259
        test_dataset = test_loader.dataset
        pair_259 = None
        for pair in getattr(test_dataset, "pairs", []):
            if pair.get("num") == 259:
                pair_259 = pair
                break

        if pair_259 is None:
            print("Could not find test pair with num=259; skipping fixed triplet plotting.")
        else:
            from dataloader import SignalTrainLA2ADataset

            if not isinstance(test_dataset, SignalTrainLA2ADataset):
                print(
                    "Test dataset is not SignalTrainLA2ADataset; "
                    "skipping fixed triplet plotting."
                )
            else:
                model.eval()

                # Segments in seconds (start, end) as provided by user
                # First four are used for both plotting and audio; the last two
                # are additional audio-only segments.
                segments_plot_sec = [
                    (2 * 60 + 14, 2 * 60 + 17),
                    (2 * 60 + 56, 2 * 60 + 59),
                    (7 * 60 + 0, 7 * 60 + 3),
                    (8 * 60 + 37, 8 * 60 + 40),
                ]
                segments_audio_extra_sec = [
                    (2 * 60 + 17, 2 * 60 + 27),
                    (2 * 60 + 56, 3 * 60 + 6),
                ]
                segments_sec = segments_plot_sec + segments_audio_extra_sec

                sr = int(pair_259["sample_rate"])
                segments_data = []

                # Models that take params: pass them for inference; others get None
                _models_taking_params = (
                    "HardwareEmulationLSTM",
                    "SimpleHardwareEmulationLSTM",
                    "ResidualLSTM",
                    "ReferenceLSTM",
                    "TCNModel",
                    "CausalBlockLSTM",
                )

                # Directory for saving audio triplets
                audio_dir = run_info.output_run_dir / "audio_triplets_259"
                audio_dir.mkdir(parents=True, exist_ok=True)

                with torch.no_grad():
                    for idx, (start_sec, end_sec) in enumerate(segments_sec):
                        start_sample = int(start_sec * sr)
                        num_samples = int((end_sec - start_sec) * sr)

                        # Load exact input/target segments from disk
                        input_seg = test_dataset._read_audio_segment(
                            pair_259["input_path"],
                            start_sample=start_sample,
                            num_samples=num_samples,
                        )
                        target_seg = test_dataset._read_audio_segment(
                            pair_259["target_path"],
                            start_sample=start_sample,
                            num_samples=num_samples,
                        )

                        # Ensure float32 NumPy arrays
                        input_seg = np.asarray(input_seg, dtype=np.float32).flatten()
                        target_seg = np.asarray(target_seg, dtype=np.float32).flatten()

                        # Build parameter vector from filename states
                        params_np = test_dataset._encode_states_to_params(
                            pair_259["states"]
                        ).astype(np.float32)

                        # Convert to tensors and run model
                        input_tensor = (
                            torch.from_numpy(input_seg)
                            .float()
                            .unsqueeze(0)
                            .unsqueeze(0)
                            .to(model.device)
                        )
                        params_tensor = (
                            torch.from_numpy(params_np)
                            .float()
                            .unsqueeze(0)
                            .to(model.device)
                        )

                        if model_takes_params or model_type in _models_taking_params:
                            pred_tensor = model(input_tensor, params_tensor)
                        else:
                            pred_tensor = model(input_tensor, None)

                        pred_seg = (
                            pred_tensor.squeeze(0).squeeze(0).detach().cpu().numpy()
                        ).astype(np.float32)

                        # Make sure all segments are the same length
                        min_len = min(
                            len(input_seg), len(target_seg), len(pred_seg)
                        )
                        input_seg = input_seg[:min_len]
                        target_seg = target_seg[:min_len]
                        pred_seg = pred_seg[:min_len]

                        # Save audio for this segment
                        start_m, start_s = divmod(start_sec, 60)
                        end_m, end_s = divmod(end_sec, 60)
                        seg_label = (
                            f"s{idx+1}_{int(start_m):02d}-{int(start_s):02d}_"
                            f"{int(end_m):02d}-{int(end_s):02d}"
                        )

                        in_path = audio_dir / f"input_259_{seg_label}.wav"
                        tgt_path = audio_dir / f"target_259_{seg_label}.wav"
                        pred_path = audio_dir / f"predicted_259_{seg_label}.wav"

                        sf.write(str(in_path), input_seg, sr)
                        sf.write(str(tgt_path), target_seg, sr)
                        sf.write(str(pred_path), pred_seg, sr)

                        audio_artifact_paths.extend(
                            [str(in_path), str(tgt_path), str(pred_path)]
                        )

                        time_axis = np.arange(min_len) / float(sr)
                        # Only keep the first four segments for plotting;
                        # the extra two are audio-only.
                        if idx < len(segments_plot_sec):
                            segments_data.append(
                                {
                                    "time": time_axis,
                                    "input": input_seg,
                                    "target": target_seg,
                                    "pred": pred_seg,
                                    "label": seg_label,
                                }
                            )

                if segments_data:
                    # Compute global y range so all plots share the same y_lim
                    all_vals = np.concatenate(
                        [
                            seg["input"].ravel()
                            for seg in segments_data
                        ]
                        + [
                            seg["target"].ravel()
                            for seg in segments_data
                        ]
                        + [
                            seg["pred"].ravel()
                            for seg in segments_data
                        ]
                    )
                    y_min, y_max = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
                    if y_max <= y_min:
                        y_min, y_max = y_min - 1e-6, y_max + 1e-6

                    num_segments = len(segments_data)
                    fig, axes = plt.subplots(
                        num_segments * 3,
                        1,
                        sharex=True,
                        figsize=(14, 1.8 * num_segments * 3),
                    )
                    axes = np.atleast_1d(axes)

                    for i, seg in enumerate(segments_data):
                        inp = seg["input"]
                        tgt = seg["target"]
                        pred = seg["pred"]
                        t_axis = seg["time"]

                        ax_in = axes[i * 3]
                        ax_tgt = axes[i * 3 + 1]
                        ax_pred = axes[i * 3 + 2]

                        ax_in.plot(t_axis, inp, color="C0", linewidth=0.5)
                        ax_in.set_ylabel("Input")
                        ax_in.set_title(f"Triplet {i + 1} ({seg['label']})")

                        ax_tgt.plot(t_axis, tgt, color="C1", linewidth=0.5)
                        ax_tgt.set_ylabel("Target")

                        ax_pred.plot(t_axis, pred, color="C2", linewidth=0.5)
                        ax_pred.set_ylabel("Predicted")

                    for ax in axes:
                        ax.set_ylim(y_min, y_max)
                    axes[-1].set_xlabel("Time (s)")
                    fig.suptitle(
                        "Test-set triplets for file 259: input, target, predicted",
                        fontsize=12,
                    )
                    plt.tight_layout()

                    plot_path = run_info.output_run_dir / "test_triplets_259.png"
                    plt.savefig(plot_path, dpi=320, bbox_inches="tight")
                    print(f"✓ Saved fixed test-set triplets to {plot_path}")

    if use_mlflow:
        import mlflow
        try:
            if final_train_loss is not None:
                mlflow.log_metric("final_train_loss", final_train_loss.item())
            if final_val_loss is not None:
                mlflow.log_metric("final_val_loss", final_val_loss.item())
            if test_metrics is not None:
                mlflow.log_metric("test_mae", test_metrics["test_mae"])
                mlflow.log_metric("test_stft", test_metrics["test_stft"])
                mlflow.log_metric("test_lufs_db", test_metrics["test_lufs_db"])
            if fig is not None:
                mlflow.log_figure(fig, "test_triplets_259.png")
            # Log audio triplets as artifacts if we created them
            for p in audio_artifact_paths:
                mlflow.log_artifact(p, artifact_path="audio_triplets_259")
            # Log model as artifact and optionally register to model registry
            mlflow.pytorch.log_model(
                model,
                artifact_path="model",
                registered_model_name=mlflow_registered_model_name,
            )
            if mlflow_registered_model_name:
                print(f"✓ Model logged and registered to MLflow as '{mlflow_registered_model_name}'")
            else:
                print("✓ Model logged to MLflow run artifacts")
        except Exception as e:
            # Artifact upload can fail if the Databricks PAT lacks the "files" scope.
            # Params and metrics are already logged; run will still be visible.
            print(f"⚠ MLflow artifact upload skipped: {e}")
        finally:
            mlflow.end_run()

    plt.close()
    # pl_logger.finalize()

    return {
        "final_train_loss": final_train_loss.item() if final_train_loss is not None else None,
        "final_val_loss": final_val_loss.item() if final_val_loss is not None else None,
        "test_mae": test_metrics["test_mae"] if test_metrics is not None else None,
        "test_stft": test_metrics["test_stft"] if test_metrics is not None else None,
        "test_lufs_db": test_metrics["test_lufs_db"] if test_metrics is not None else None,
        "train_batches": len(train_loader),
        "val_batches": len(val_loader),
        "test_batches": len(test_loader) if test_loader is not None else 0,
        "num_epochs": num_epochs,
    }


@app.function(
    image=image,
    volumes={DATASET_MOUNT: signaltrain_volume},
    timeout=60 * 5,  # 5 minutes
)
def check_dataset_exists(
    root_dir: str = "/data/signaltrain/SignalTrain_LA2A_Dataset_1.1",
):
    """Check if the SignalTrain dataset exists at the specified path."""
    import os
    from pathlib import Path
    
    print("=" * 80)
    print("CHECKING SIGNALTRAIN DATASET")
    print("=" * 80)
    print(f"Dataset path: {root_dir}")
    print()
    
    # Check if the root directory exists
    if os.path.exists(root_dir):
        print(f"✓ Dataset root directory exists: {root_dir}")
        
        # Check for expected subdirectories
        expected_dirs = ["Train", "Val", "Test"]
        for subdir in expected_dirs:
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.exists(subdir_path):
                # Count files in the subdirectory
                files = list(Path(subdir_path).rglob("*"))
                file_count = len([f for f in files if f.is_file()])
                print(f"  ✓ {subdir}/ exists ({file_count} files)")
            else:
                print(f"  ✗ {subdir}/ does not exist")
        
        return True
    else:
        print(f"✗ Dataset root directory does not exist: {root_dir}")
        print(f"  Checked mount point: {DATASET_MOUNT}")
        
        # List what's actually in the mount point
        if os.path.exists(DATASET_MOUNT):
            print(f"\nContents of {DATASET_MOUNT}:")
            try:
                items = os.listdir(DATASET_MOUNT)
                if items:
                    for item in items:
                        item_path = os.path.join(DATASET_MOUNT, item)
                        item_type = "directory" if os.path.isdir(item_path) else "file"
                        print(f"  - {item} ({item_type})")
                else:
                    print("  (empty)")
            except Exception as e:
                print(f"  Error listing contents: {e}")
        else:
            print(f"  Mount point {DATASET_MOUNT} does not exist")
        
        return False


@app.local_entrypoint()
def main(root_dir: str = "/data/signaltrain/SignalTrain_LA2A_Dataset_1.1"):
    """Local entrypoint to check if dataset exists."""
    print("Connecting to Modal and checking dataset...")
    result = check_dataset_exists.remote(root_dir=root_dir)
    
    if result:
        print("\n✓ Dataset exists and is accessible!")
    else:
        print("\n✗ Dataset not found. Please sync the dataset first.")
    
    return result


@app.local_entrypoint()
def train_epoch(
    config_path: str = None,
    # Legacy parameters (used if config_path is None)
    root_dir: str = "/data/signaltrain/SignalTrain_LA2A_Dataset_1.1",
    train_subset: str = "Train",
    train_length: int = 65536,
    batch_size: int = 4,
    num_workers: int = 2,
    n_params: int = 2,
    hidden_size: int = 128,
    learning_rate: float = 1e-4,
    num_epochs: int = 1,
):
    """
    Local entrypoint to train for multiple epochs.

    Usage:
        # With config file:
        modal run train_step_by_step.py::train_epoch --config-path config.train_step_by_step.yaml

        # With individual parameters:
        modal run train_step_by_step.py::train_epoch --num-epochs 10 --batch-size 8

    MLflow experiment tracking:
        Create a Modal secret named "mlflow" with your tracking server env vars.
        The secret is attached to the training function, so just run as above.

        Databricks:
          modal secret create mlflow \\
            MLFLOW_TRACKING_URI=databricks \\
            MLFLOW_EXPERIMENT_NAME="/Users/you@org/your-experiment" \\
            DATABRICKS_HOST=https://your-workspace.cloud.databricks.com \\
            DATABRICKS_TOKEN=dapi...
        Then: modal run train_step_by_step.py::train_epoch ...
        For artifact upload (e.g. val_triplets.png), create the PAT with the
        "files" scope (Settings → Developer → Access tokens → Generate new token).

        If you don't use MLflow, create an empty secret so the run succeeds:
          modal secret create mlflow MLFLOW_TRACKING_URI=
    """
    result = train_model.remote(
        config_path=config_path,
        root_dir=root_dir,
        train_subset=train_subset,
        train_length=train_length,
        batch_size=batch_size,
        num_workers=num_workers,
        n_params=n_params,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
    )
    
    return result


if __name__ == "__main__":
    main()
