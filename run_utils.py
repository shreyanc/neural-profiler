"""Run folder naming and initialization for training scripts.

Shared utilities so different train scripts (e.g. train_step_by_step.py)
can create consistent, filesystem-safe run directories with datetime prefixes.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class RunDirInfo:
    """Result of initializing a run directory."""

    run_name: str
    """Human-readable run name (e.g. for logging)."""

    run_dir_name: str
    """Filesystem-safe directory name including datetime prefix."""

    output_run_dir: Path
    """Absolute path to the created run directory."""


def _run_name_parts(
    model_type: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    train_length: int,
    model_config: Optional[dict] = None,
) -> list:
    """Build the list of run name components (model, ep, bs, lr, len, h?, L?)."""
    if model_type == "HardwareEmulationLSTM":
        model_label = "HWEmuLSTM"
    elif model_type == "SimpleHardwareEmulationLSTM":
        model_label = "SimpleHWEmuLSTM"
    else:
        model_label = model_type
    lr_str = (
        f"{learning_rate:.0e}"
        .replace(".", "")
        .replace("e-0", "e-")
        .replace("e+0", "e")
    )
    parts = [
        model_label,
        f"ep{num_epochs}",
        f"bs{batch_size}",
        f"lr{lr_str}",
        f"len{train_length}",
    ]
    if model_config:
        if model_config.get("hidden_size") is not None:
            parts.append(f"h{model_config['hidden_size']}")
        if model_config.get("num_layers") is not None:
            parts.append(f"L{model_config['num_layers']}")
    return parts


def make_run_dir_name(
    model_type: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    train_length: int,
    model_config: Optional[dict] = None,
    run_start: Optional[datetime] = None,
) -> str:
    """Build a filesystem-safe run directory name with datetime prefix.

    Format: YYYY-MM-DD_HH-MM-SS_<model>_ep<N>_bs<N>_lr<lr>_len<N>_[h<N>]_[L<N>]

    Args:
        model_type: e.g. "TestLSTM", "HardwareEmulationLSTM".
        num_epochs: Number of training epochs.
        batch_size: Batch size.
        learning_rate: Learning rate (used for label only).
        train_length: Training segment length in samples.
        model_config: Optional dict with "hidden_size", "num_layers" for extra tags.
        run_start: Start time for the datetime prefix (default: now UTC).

    Returns:
        A filesystem-safe directory name string.
    """
    if run_start is None:
        run_start = datetime.now(timezone.utc)
    datetime_prefix = run_start.strftime("%Y-%m-%d_%H-%M-%S")
    run_name_parts = _run_name_parts(
        model_type, num_epochs, batch_size, learning_rate, train_length, model_config
    )
    run_name = "_".join(str(p) for p in run_name_parts)
    run_name_safe = "".join(
        c if c.isalnum() or c in "._-" else "_" for c in run_name
    )
    return f"{datetime_prefix}_{run_name_safe}"


def init_run_dir(
    base_dir: Path,
    model_type: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    train_length: int,
    model_config: Optional[dict] = None,
    run_start: Optional[datetime] = None,
) -> RunDirInfo:
    """Create the run output directory and return run info.

    The directory created is base_dir / "outputs" / <run_dir_name>.

    Args:
        base_dir: Base path (e.g. workspace root) under which "outputs" lives.
        model_type: e.g. "TestLSTM", "HardwareEmulationLSTM".
        num_epochs: Number of training epochs.
        batch_size: Batch size.
        learning_rate: Learning rate.
        train_length: Training segment length in samples.
        model_config: Optional model config for extra dir name parts.
        run_start: Optional start time for datetime prefix.

    Returns:
        RunDirInfo with run_name, run_dir_name, and output_run_dir (created).
    """
    if run_start is None:
        run_start = datetime.now(timezone.utc)
    run_dir_name = make_run_dir_name(
        model_type=model_type,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        train_length=train_length,
        model_config=model_config,
        run_start=run_start,
    )
    run_name_parts = _run_name_parts(
        model_type, num_epochs, batch_size, learning_rate, train_length, model_config
    )
    run_name = "_".join(str(p) for p in run_name_parts)
    output_run_dir = base_dir / "outputs" / run_dir_name
    output_run_dir.mkdir(parents=True, exist_ok=True)
    return RunDirInfo(
        run_name=run_name,
        run_dir_name=run_dir_name,
        output_run_dir=output_run_dir,
    )
