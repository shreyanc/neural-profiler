#!/usr/bin/env python3
"""Training script for LSTM-based audio-to-audio model on SignalTrain LA2A dataset.

This script provides the model architecture and training functions that can be used
standalone or imported by modal_app.py for training on Modal.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloader import DatasetConfig as DataConfigForLoader, create_dataloaders
from config import ExperimentConfig, load_config_from_yaml


class LSTMAudioModel(nn.Module):
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
        use_params: bool = True,
    ):
        super().__init__()
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
        
        # Reshape to (B, T, 1)
        seq = x.transpose(1, 2).contiguous()  # (B, T, 1)
        
        if self.use_params and params is not None and self.n_params > 0:
            # Create parameter tensor that repeats along time dimension
            p = params.unsqueeze(1).repeat(1, t, 1)  # (B, T, P)
            seq = torch.cat([seq, p], dim=-1)  # (B, T, 1+P)
            seq = seq.contiguous()
        
        # Ensure contiguous before LSTM
        if not seq.is_contiguous():
            seq = seq.contiguous()
        
        lstm_out, _ = self.lstm(seq)  # (B, T, H)
        pred = self.output_proj(lstm_out)  # (B, T, 1)
        pred = pred.transpose(1, 2).contiguous()  # (B, 1, T)
        return pred


def train_model(
    config: ExperimentConfig,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train the LSTM model using the provided configuration.
    
    Args:
        config: Experiment configuration
        device: PyTorch device (if None, uses cuda if available, else cpu)
        verbose: Whether to print training progress
        
    Returns:
        Dictionary with training results including final losses and metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if verbose:
        print("=" * 80)
        print("TRAINING LSTM MODEL")
        print("=" * 80)
        print(f"Device: {device}")
        print(f"Dataset root: {config.dataset.root_dir}")
        print(f"Train subset: {config.dataset.train_subset}")
        print(f"Val subset: {config.dataset.val_subset}")
        print(f"Train length: {config.dataset.train_length}")
        print(f"Eval length: {config.dataset.eval_length}")
        print(f"Batch size: {config.dataset.batch_size}")
        print(f"Num workers: {config.dataset.num_workers}")
        print(f"Num epochs: {config.training.num_epochs}")
        print(f"Learning rate: {config.training.learning_rate}")
        print(f"LSTM hidden size: {config.model.hidden_size}")
        print(f"LSTM num layers: {config.model.num_layers}")
        print(f"LSTM dropout: {config.model.dropout}")
        print(f"Use params: {config.model.use_params}")
        print("=" * 80)
    
    # Set random seed for reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Create data configuration
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
    
    # Create dataloaders
    if verbose:
        print("\nCreating dataloaders...")
    train_loader, val_loader, _ = create_dataloaders(data_config)
    if verbose:
        print(f"✓ Dataloaders created")
    
    # Create model
    if verbose:
        print("\nCreating LSTM model...")
    model = LSTMAudioModel(
        input_length=config.model.input_length,
        n_params=config.model.n_params,
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        use_params=config.model.use_params,
    )
    model = model.to(device)
    if verbose:
        print(f"✓ Model created: {model}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,} total")
    
    # Create optimizer
    if config.training.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
    elif config.training.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.training.optimizer}")
    
    # Create scheduler
    if config.training.scheduler.lower() == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.training.scheduler_factor,
            patience=config.training.scheduler_patience,
            verbose=verbose,
        )
    else:
        scheduler = None
    
    # Training loop
    if verbose:
        print("\n" + "=" * 80)
        print("TRAINING")
        print("=" * 80)
    
    best_val_loss = float("inf")
    patience_counter = 0
    
    for epoch in range(config.training.num_epochs):
        if verbose:
            print(f"\nEpoch {epoch + 1}/{config.training.num_epochs}")
        
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (input_audio, target_audio, params) in enumerate(train_loader):
            input_audio = input_audio.to(device)
            target_audio = target_audio.to(device)
            params = params.to(device)
            
            # Forward pass
            pred_audio = model(input_audio, params)
            
            # Compute loss
            loss = F.mse_loss(pred_audio, target_audio)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping if specified
            if config.training.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.training.gradient_clip_norm
                )
            
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            if verbose and (batch_idx + 1) % config.training.log_every_n_steps == 0:
                avg_loss = train_loss / train_batches
                print(f"  Batch {batch_idx + 1}: loss = {loss.item():.6f}, avg_loss = {avg_loss:.6f}")
        
        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0.0
        if verbose:
            print(f"  Train loss: {avg_train_loss:.6f} ({train_batches} batches)")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_idx, (input_audio, target_audio, params) in enumerate(val_loader):
                input_audio = input_audio.to(device)
                target_audio = target_audio.to(device)
                params = params.to(device)
                
                pred_audio = model(input_audio, params)
                loss = F.mse_loss(pred_audio, target_audio)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
        if verbose:
            print(f"  Val loss: {avg_val_loss:.6f} ({val_batches} batches)")
        
        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step(avg_val_loss)
            if verbose:
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"  Learning rate: {current_lr:.2e}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model checkpoint
            checkpoint_path = Path(config.checkpoint_dir) / "best_model.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "train_loss": avg_train_loss,
                    "config": config,
                },
                checkpoint_path,
            )
            if verbose:
                print(f"  ✓ Saved best model checkpoint (val_loss: {avg_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= config.training.early_stopping_patience:
                if verbose:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    print(f"Best val loss: {best_val_loss:.6f}")
                break
    
    if verbose:
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"Final train loss: {avg_train_loss:.6f}")
        print(f"Final val loss: {avg_val_loss:.6f}")
        print(f"Best val loss: {best_val_loss:.6f}")
    
    return {
        "final_train_loss": avg_train_loss,
        "final_val_loss": avg_val_loss,
        "best_val_loss": best_val_loss,
        "train_batches": train_batches,
        "val_batches": val_batches,
        "epochs_completed": epoch + 1,
    }


def main():
    """Main entry point for standalone training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train LSTM model on SignalTrain LA2A dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="config.example.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). If not specified, auto-detects.",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config_from_yaml(args.config)
    
    # Set device
    if args.device is None:
        device = None  # Will auto-detect in train_model
    else:
        device = torch.device(args.device)
    
    # Train model
    results = train_model(config, device=device, verbose=True)
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    import json
    print(json.dumps(results, indent=2))
    
    return results


if __name__ == "__main__":
    main()
