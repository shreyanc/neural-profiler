#!/usr/bin/env python3
"""Minimal training script with identity model for Modal.

This script performs minimal training using the dataloaders with an identity model
(just passes input through). No logging, callbacks, or complex ML features.
"""

import sys
from pathlib import Path

import modal

APP_NAME = "train-minimal"
WORKDIR = "/workspace"
DATASET_MOUNT = "/data/signaltrain"

# Persistent volume for the SignalTrain dataset
signaltrain_volume = modal.Volume.from_name(
    "signaltrain-dataset", create_if_missing=False
)

# Base image with dependencies
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
        # Torch stack
        "torch>=2.1.2",
        "torchvision>=0.16.1",
    )
    # Include project code so imports resolve
    .add_local_dir(Path(__file__).parent, remote_path=WORKDIR)
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    volumes={DATASET_MOUNT: signaltrain_volume},
    timeout=60 * 60 * 2,  # 2 hours
)
def train_minimal(
    root_dir: str = "/data/signaltrain/SignalTrain_LA2A_Dataset_1.1",
    train_subset: str = "Train",
    val_subset: str = "Val",
    train_length: int = 65536,
    eval_length: int = 131072,
    batch_size: int = 8,
    num_workers: int = 4,
    n_params: int = 2,
    num_epochs: int = 1,
    learning_rate: float = 1e-4,
    hidden_size: int = 256,
    num_layers: int = 2,
    dropout: float = 0.1,
    use_params: bool = True,
):
    """Minimal training loop with identity model."""
    import sys
    import torch
    import torch.nn.functional as F
    
    sys.path.insert(0, WORKDIR)
    
    from dataloader import DatasetConfig as DataConfigForLoader, create_dataloaders
    
    print("=" * 80)
    print("MINIMAL TRAINING WITH IDENTITY MODEL")
    print("=" * 80)
    print(f"Dataset root: {root_dir}")
    print(f"Train subset: {train_subset}")
    print(f"Val subset: {val_subset}")
    print(f"Train length: {train_length}")
    print(f"Eval length: {eval_length}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"Num epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"LSTM hidden size: {hidden_size}")
    print(f"LSTM num layers: {num_layers}")
    print(f"LSTM dropout: {dropout}")
    print(f"Use params: {use_params}")
    print("=" * 80)
    
    # Create data configuration
    data_config = DataConfigForLoader(
        root_dir=root_dir,
        train_subset=train_subset,
        val_subset=val_subset,
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
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, _ = create_dataloaders(data_config)
    print(f"✓ Dataloaders created")
    
    # Create LSTM model
    print("\nCreating LSTM model...")
    class LSTMAudioModel(torch.nn.Module):
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
            
            self.lstm = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            
            self.output_proj = torch.nn.Linear(hidden_size, 1)
        
        def forward(self, x: torch.Tensor, params: torch.Tensor = None) -> torch.Tensor:
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
    
    model = LSTMAudioModel(
        input_length=train_length,
        n_params=n_params,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        use_params=use_params,
    )
    print(f"✓ Model created: {model}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,} total")
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (input_audio, target_audio, params) in enumerate(train_loader):
            # Forward pass
            pred_audio = model(input_audio, params)
            
            # Compute loss
            loss = F.mse_loss(pred_audio, target_audio)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            if (batch_idx + 1) % 10 == 0:
                avg_loss = train_loss / train_batches
                print(f"  Batch {batch_idx + 1}: loss = {loss.item():.6f}, avg_loss = {avg_loss:.6f}")
        
        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0.0
        print(f"  Train loss: {avg_train_loss:.6f} ({train_batches} batches)")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_idx, (input_audio, target_audio, params) in enumerate(val_loader):
                pred_audio = model(input_audio, params)
                loss = F.mse_loss(pred_audio, target_audio)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
        print(f"  Val loss: {avg_val_loss:.6f} ({val_batches} batches)")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Final train loss: {avg_train_loss:.6f}")
    print(f"Final val loss: {avg_val_loss:.6f}")
    
    return {
        "final_train_loss": avg_train_loss,
        "final_val_loss": avg_val_loss,
        "train_batches": train_batches,
        "val_batches": val_batches,
    }


@app.local_entrypoint()
def main(
    root_dir: str = "/data/signaltrain/SignalTrain_LA2A_Dataset_1.1",
    train_subset: str = "Train",
    val_subset: str = "Val",
    train_length: int = 65536,
    eval_length: int = 131072,
    batch_size: int = 8,
    num_workers: int = 4,
    n_params: int = 2,
    num_epochs: int = 1,
    learning_rate: float = 1e-4,
    hidden_size: int = 256,
    num_layers: int = 2,
    dropout: float = 0.1,
    use_params: bool = True,
):
    """Local entrypoint to run minimal training on Modal."""
    result = train_minimal.remote(
        root_dir=root_dir,
        train_subset=train_subset,
        val_subset=val_subset,
        train_length=train_length,
        eval_length=eval_length,
        batch_size=batch_size,
        num_workers=num_workers,
        n_params=n_params,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        use_params=use_params,
    )
    
    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)
    import json
    print(json.dumps(result, indent=2))
    
    return result


if __name__ == "__main__":
    main()
