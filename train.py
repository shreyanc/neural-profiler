"""
Training script for mapping input audio signals to target signals
using an LSTM-based architecture with PyTorch Lightning and Weights & Biases.

The script expects the SignalTrain LA2A dataset structure used by `dataloader.py`
and trains a sequence-to-sequence regression model that predicts the processed
audio given the clean input (optionally conditioned on LA2A parameters).
"""

import argparse
import os
from typing import Optional

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


class LSTMAudioModel(pl.LightningModule):
    """
    LSTM-based model for audio-to-audio regression.

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

        self.input_length = input_length
        self.n_params = n_params
        self.use_params = use_params

        input_size = 1 + (n_params if use_params and n_params > 0 else 0)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.output_proj = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, params: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:      (B, 1, T) input audio
            params: (B, P) parameter vector (optional)
        Returns:
            (B, 1, T) predicted audio
        """
        b, c, t = x.shape
        assert c == 1, "Expected mono audio with shape (B, 1, T)"

        # Reshape to (B, T, 1) - transpose is safe and commonly contiguous
        seq = x.transpose(1, 2).contiguous()  # (B, T, 1)

        if self.use_params and params is not None and self.n_params > 0:
            # Create parameter tensor that repeats along time dimension
            # Use repeat instead of expand to ensure contiguity
            p = params.unsqueeze(1).repeat(1, t, 1)  # (B, T, P)
            seq = torch.cat([seq, p], dim=-1)  # (B, T, 1+P)
            # Make contiguous after concatenation
            seq = seq.contiguous()

        # Ensure contiguous before LSTM (defensive programming)
        if not seq.is_contiguous():
            seq = seq.contiguous()
        
        lstm_out, _ = self.lstm(seq)  # (B, T, H)
        pred = self.output_proj(lstm_out)  # (B, T, 1)
        pred = pred.transpose(1, 2).contiguous()  # (B, 1, T)
        return pred

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
            # verbose=True,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LSTM model for SignalTrain LA2A dataset using YAML config",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from (optional)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration from YAML
    config: ExperimentConfig = load_config_from_yaml(args.config)
    # config: ExperimentConfig = load_config_from_yaml("./config.example.yaml")

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

    train_loader, val_loader, _ = create_dataloaders(data_config)

    # Create model from config
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

    # WandB logger
    wandb_logger = WandbLogger(
        project=config.project,
        name=config.run_name or config.experiment_name,
        entity=config.wandb_entity,
        save_dir=config.log_dir,  # save_dir is valid in pytorch-lightning 2.2+
        log_model=False,
        tags=config.tags,
        offline=True,
    )

    # Callbacks
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
        verbose=True,  # verbose is valid in pytorch-lightning 2.2+
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Build trainer from config
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

    # Resume from checkpoint if provided
    # ckpt_path = args.resume if args.resume else None
    trainer.fit(
        model,
        train_dataloaders=train_loader,  # Changed from train_dataloaders (plural) to train_dataloader (singular) in PL 2.0+
        val_dataloaders=val_loader,  # Changed from val_dataloaders (plural) to val_dataloader (singular) in PL 2.0+
        # ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    main()

