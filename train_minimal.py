import modal
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from modal_common import (
    DATASET_MOUNT,
    WORKDIR,
    image_signaltrain_training,
    make_signaltrain_volume,
)
from train_minimal_core import (
    ResidualLSTMmodeler,
    SignalTrainLA2ADatasetSingle,
    LSTMmodeler,
    create_loss_function,
)

APP_NAME = "train-minimal"

signaltrain_volume = make_signaltrain_volume(create_if_missing=False)
image = image_signaltrain_training()

app = modal.App(APP_NAME)

# Parameter names accepted by train_minimal_model (for sweep merge / coercion).
_TRAIN_MINIMAL_INT_PARAMS = frozenset(
    {
        "clip_idx",
        "segment_length",
        "sample_rate",
        "batch_size",
        "val_batch_size",
        "test_batch_size",
        "hidden_size",
        "num_layers",
        "block_size",
        "num_epochs",
        "nparams",
        "nblocks",
        "kernel_size",
        "dilation_growth",
        "channel_growth",
        "channel_width",
        "stack_size",
    }
)
_TRAIN_MINIMAL_FLOAT_PARAMS = frozenset({"learning_rate"})
_TRAIN_MINIMAL_BOOL_PARAMS = frozenset({"grouped", "causal", "skip_connections"})
_TRAIN_MINIMAL_MODEL_PARAM_NAMES = (
    _TRAIN_MINIMAL_INT_PARAMS
    | _TRAIN_MINIMAL_FLOAT_PARAMS
    | _TRAIN_MINIMAL_BOOL_PARAMS
    | frozenset(
        {
            "root_dir",
            "subset_dir_name",
            "subset",
            "model_name",
            "train_loss_type",
            "val_loss_type",
        }
    )
)


def _coerce_train_minimal_param(name: str, value):
    """Coerce a wandb.config value to the type expected by train_minimal_model."""
    if name in _TRAIN_MINIMAL_BOOL_PARAMS:
        if isinstance(value, bool):
            return value
        return str(value).lower() in ("true", "1", "yes")
    if name in _TRAIN_MINIMAL_INT_PARAMS:
        return int(value)
    if name in _TRAIN_MINIMAL_FLOAT_PARAMS:
        return float(value)
    return value


def _merge_sweep_config_into_train_kwargs(
    base: dict, wandb_cfg: dict
) -> dict:
    """Overlay sweep-sampled keys from wandb.config onto CLI defaults."""
    out = dict(base)
    for key, raw in wandb_cfg.items():
        if key.startswith("_"):
            continue
        if key not in _TRAIN_MINIMAL_MODEL_PARAM_NAMES:
            continue
        out[key] = _coerce_train_minimal_param(key, raw)
    return out


@app.function(
    image=image,
    gpu="A10G",  # Use A10G GPU for training
    volumes={DATASET_MOUNT: signaltrain_volume},
    timeout=60 * 60 * 2,  # 2 hours
    secrets=[
        modal.Secret.from_name("wandb")  # Expects WANDB_API_KEY
    ],
)
def train_minimal_model(
    root_dir: str = "/data/signaltrain/SignalTrain_LA2A_Dataset_1.1",
    subset_dir_name: str = "Train",
    subset: str = "train",
    clip_idx: int = 263,
    segment_length: int = 65536,
    sample_rate: int = 44100,
    batch_size: int = 2,
    val_batch_size: int = 1,
    test_batch_size: int = 1,
    model_name: str = "residual_lstm",
    hidden_size: int = 128,
    num_layers: int = 1,
    block_size: int = 2048,
    learning_rate: float = 0.001,
    num_epochs: int = 1,
    train_loss_type: str = "MSE",
    val_loss_type: str = "MSE",
    # TCNModel-specific parameters
    nparams: int = 0,
    nblocks: int = 10,
    kernel_size: int = 3,
    dilation_growth: int = 1,
    channel_growth: int = 1,
    channel_width: int = 32,
    stack_size: int = 10,
    grouped: bool = False,
    causal: bool = False,
    skip_connections: bool = False,
    wandb_run_id: str | None = None,
    wandb_project: str = "neural-profiler",
):
    """
    Train the minimal LSTM model on Modal.
    
    This function wraps the core training logic from train_minimal.py
    to enable running on Modal infrastructure.
    """
    import sys
    import wandb
    
    # Ensure project code is on PYTHONPATH inside the container
    sys.path.insert(0, WORKDIR)
    
    # Import TCNModel from models.py
    from models import TCNModel
    
    print("=" * 80)
    print("MINIMAL TRAINING ON MODAL")
    print("=" * 80)
    print(f"Dataset root: {root_dir}")
    print(f"Subset dir: {subset_dir_name}")
    print(f"Subset: {subset}")
    print(f"Clip index: {clip_idx}")
    print(f"Segment length: {segment_length}")
    print(f"Sample rate: {sample_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Hidden size: {hidden_size}")
    print(f"Block size: {block_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Num epochs: {num_epochs}")
    print(f"Train loss type: {train_loss_type}")
    print(f"Val loss type: {val_loss_type}")
    print("=" * 80)
    
    # Check GPU availability
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        # Use Tensor Cores on supported GPUs (e.g. A10): trade precision for performance
        torch.set_float32_matmul_precision("medium")
    else:
        print("⚠ GPU not available - training will run on CPU")
    
    # Create datasets - core implementation logic unchanged
    print("\nCreating datasets...")
    dataset = SignalTrainLA2ADatasetSingle(
        root_dir, subset_dir_name, subset, clip_idx, segment_length, sample_rate
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    val_dataset = SignalTrainLA2ADatasetSingle(
        root_dir, subset_dir_name, "val", clip_idx, segment_length, sample_rate
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=val_batch_size, shuffle=False
    )
    
    test_dataset = SignalTrainLA2ADatasetSingle(
        root_dir, subset_dir_name, "test", clip_idx, segment_length, sample_rate
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False
    )
    print(f"✓ Datasets created")
    print(f"  Train batches: {len(dataloader)}")
    print(f"  Val batches: {len(val_dataloader)}")
    print(f"  Test batches: {len(test_dataloader)}")
    
    # Create model and optimizer - core implementation logic unchanged
    print("\nCreating model...")

    if model_name == "residual_lstm":
        model = ResidualLSTMmodeler(input_size=1, hidden_size=hidden_size, output_size=1, block_size=block_size)
    elif model_name == "lstm":
        model = LSTMmodeler(input_size=1, hidden_size=hidden_size, output_size=1, block_size=block_size)
    elif model_name == "tcn":
        model = TCNModel(
            nparams=nparams,
            ninputs=1,
            noutputs=1,
            nblocks=nblocks,
            kernel_size=kernel_size,
            dilation_growth=dilation_growth,
            channel_growth=channel_growth,
            channel_width=channel_width,
            stack_size=stack_size,
            grouped=grouped,
            causal=causal,
            skip_connections=skip_connections,
            learning_rate=learning_rate,
        )
    else:
        raise ValueError(f"Invalid model name: {model_name}. Supported: 'lstm', 'residual_lstm', 'tcn'")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Move model to GPU if available
    if gpu_available:
        model = model.cuda()
        print("✓ Model moved to GPU")
    
    print(f"✓ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create loss functions
    print("\nCreating loss functions...")
    train_loss_fn = create_loss_function(train_loss_type, sample_rate)
    val_loss_fn = create_loss_function(val_loss_type, sample_rate)
    
    # Move loss functions to GPU if available (for auraloss losses)
    if gpu_available:
        if hasattr(train_loss_fn, 'cuda'):
            train_loss_fn = train_loss_fn.cuda()
        if hasattr(val_loss_fn, 'cuda'):
            val_loss_fn = val_loss_fn.cuda()
    
    print(f"✓ Training loss: {train_loss_type}")
    print(f"✓ Validation loss: {val_loss_type}")
    
    # Initialize WandB (same run id when resuming from local sweep entrypoint)
    print("\nInitializing WandB...")
    wandb_config_payload = {
        "root_dir": root_dir,
        "subset_dir_name": subset_dir_name,
        "subset": subset,
        "clip_idx": clip_idx,
        "segment_length": segment_length,
        "sample_rate": sample_rate,
        "batch_size": batch_size,
        "val_batch_size": val_batch_size,
        "test_batch_size": test_batch_size,
        "hidden_size": hidden_size,
        "block_size": block_size,
        "num_layers": num_layers,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "train_loss_type": train_loss_type,
        "val_loss_type": val_loss_type,
        "model_name": model_name,
        "nparams": nparams,
        "nblocks": nblocks,
        "kernel_size": kernel_size,
        "dilation_growth": dilation_growth,
        "channel_growth": channel_growth,
        "channel_width": channel_width,
        "stack_size": stack_size,
        "grouped": grouped,
        "causal": causal,
        "skip_connections": skip_connections,
        "model_params": sum(p.numel() for p in model.parameters()),
    }
    run_name = (
        f"{model_name}-minimal-{hidden_size}h-layers{num_layers}-bs{block_size}-"
        f"{train_loss_type}-{val_loss_type}-epochs{num_epochs}"
    )
    init_kw: dict = {
        "project": wandb_project,
        "name": run_name,
        "config": wandb_config_payload,
        "mode": "online",
    }
    if wandb_run_id:
        init_kw["id"] = wandb_run_id
        init_kw["resume"] = "allow"
    run = wandb.init(**init_kw)
    print("✓ WandB initialized")
    print(f"  WandB Run URL: {run.url}")
    print(f"  WandB Run ID: {run.id}")
    print(f"  WandB Project: {run.project}")
    print(f"  WandB Entity: {run.entity}")
    
    # Training loop - core implementation logic unchanged
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        for idx, batch in enumerate(dataloader):
            input_audio, target_audio = batch
            
            # Move to GPU if available
            if gpu_available:
                input_audio = input_audio.cuda()
                target_audio = target_audio.cuda()
            
            optimizer.zero_grad()
            pred_audio = model(input_audio.unsqueeze(1))
            target_audio_expanded = target_audio.unsqueeze(1)
            # print(input_audio.unsqueeze(1).shape, pred_audio.shape, target_audio_expanded.shape)
            
            # Use configurable training loss
            loss = train_loss_fn(pred_audio, target_audio_expanded)
            # print(f"Batch {idx}, Loss: {loss.item():.6f}")
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            # Log batch-level metrics to wandb every 10 batches
            if (idx + 1) % 10 == 0:
                wandb.log({
                    "train/batch_loss": loss.item(),
                    "train/avg_loss": train_loss / train_batches,
                    "train/epoch": epoch + 1,
                    "train/batch": idx + 1,
                })
        
        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0.0
        print(f"Train Loss: {avg_train_loss:.6f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for idx, batch in enumerate(val_dataloader):
                input_audio, target_audio = batch
                
                # Move to GPU if available
                if gpu_available:
                    input_audio = input_audio.cuda()
                    target_audio = target_audio.cuda()
                
                pred_audio = model(input_audio.unsqueeze(1))
                target_audio_expanded = target_audio.unsqueeze(1)
                
                # Use configurable validation loss
                batch_val_loss = val_loss_fn(pred_audio, target_audio_expanded).item()
                val_loss += batch_val_loss
                val_batches += 1
                # print(f"Batch {idx}, Val Loss: {batch_val_loss:.6f}")
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
        # print(f"Val Loss: {avg_val_loss:.6f}")
        
        # Test phase
        test_loss = 0.0
        test_batches = 0
        with torch.no_grad():
            for idx, batch in enumerate(test_dataloader):
                input_audio, target_audio = batch
                
                # Move to GPU if available
                if gpu_available:
                    input_audio = input_audio.cuda()
                    target_audio = target_audio.cuda()
                
                pred_audio = model(input_audio.unsqueeze(1))
                target_audio_expanded = target_audio.unsqueeze(1)
                
                # Use validation loss function for test set as well
                batch_test_loss = val_loss_fn(pred_audio, target_audio_expanded).item()
                test_loss += batch_test_loss
                test_batches += 1
                # print(f"Batch {idx}, Test Loss: {batch_test_loss:.6f}")
        avg_test_loss = test_loss / test_batches if test_batches > 0 else 0.0
        # print(f"Test Loss: {avg_test_loss:.6f}")
        
        # Log epoch-level metrics to wandb
        wandb.log({
            "train/epoch_loss": avg_train_loss,
            "val/epoch_loss": avg_val_loss,
            "test/epoch_loss": avg_test_loss,
            "epoch": epoch + 1,
        })
        print(f"  ✓ Logged to WandB: {run.url}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Final train loss: {avg_train_loss:.6f}")
    print(f"Final val loss: {avg_val_loss:.6f}")
    print(f"Final test loss: {avg_test_loss:.6f}")
    print(f"\nWandB Run URL: {run.url}")
    
    # Compute test metrics using auraloss
    print("\n" + "=" * 80)
    print("COMPUTING TEST METRICS")
    print("=" * 80)
    
    import auraloss
    import pyloudnorm as pyln
    
    l1_fn = torch.nn.L1Loss()
    stft_fn = auraloss.freq.STFTLoss()
    meter = pyln.Meter(sample_rate)
    
    mae_scores = []
    stft_scores = []
    lufs_diff_scores = []
    
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(test_dataloader):
            input_audio, target_audio = batch
            
            # Move to GPU if available
            if gpu_available:
                input_audio = input_audio.cuda()
                target_audio = target_audio.cuda()
            
            pred_audio = model(input_audio.unsqueeze(1))
            
            # Compute metrics for each item in the batch
            for i in range(pred_audio.shape[0]):
                pred_item = pred_audio[i:i+1]  # (1, 1, T)
                target_item = target_audio[i:i+1].unsqueeze(1)  # (1, 1, T)
                
                # MAE (Mean Absolute Error)
                mae = l1_fn(pred_item, target_item).item()
                mae_scores.append(mae)
                
                # STFT loss
                stft_val = stft_fn(pred_item, target_item).item()
                stft_scores.append(stft_val)
                
                # LUFS difference
                target_np = target_item.squeeze().detach().cpu().numpy()
                pred_np = pred_item.squeeze().detach().cpu().numpy()
                
                try:
                    target_lufs = meter.integrated_loudness(target_np)
                    output_lufs = meter.integrated_loudness(pred_np)
                    lufs_diff = float(abs(output_lufs - target_lufs))
                    lufs_diff_scores.append(lufs_diff)
                except Exception as e:
                    # Skip LUFS computation if audio is too quiet or invalid
                    print(f"  ⚠ Skipping LUFS for batch {idx}, item {i}: {e}")
    
    if mae_scores:
        mean_mae = float(np.mean(mae_scores))
        mean_stft = float(np.mean(stft_scores))
        mean_lufs_diff = float(np.mean(lufs_diff_scores)) if lufs_diff_scores else None
        
        print("-" * 64)
        print("TEST METRICS (mean over test set)")
        print(f"MAE:        {mean_mae:0.4e}")
        print(f"STFT:       {mean_stft:0.4f}")
        if mean_lufs_diff is not None:
            print(f"dB LUFS Δ:  {mean_lufs_diff:0.4f}")
        else:
            print(f"dB LUFS Δ:  N/A (could not compute)")
        
        # Log to wandb
        test_metrics = {
            "test/mae": mean_mae,
            "test/stft": mean_stft,
        }
        if mean_lufs_diff is not None:
            test_metrics["test/lufs_db"] = mean_lufs_diff
        
        wandb.log(test_metrics)
        print(f"✓ Logged test metrics to WandB")
    else:
        print("⚠ No test samples found for evaluation.")
    
    # Plot the last three segments from test set
    print("\n" + "=" * 80)
    print("GENERATING WAVEFORM PLOTS")
    print("=" * 80)
    
    model.eval()
    test_dataset_size = len(test_dataset)
    num_segments_to_plot = min(3, test_dataset_size)
    
    if num_segments_to_plot > 0:
        # Get the last 3 segments from the test dataset
        segments_to_plot = []
        for i in range(test_dataset_size - num_segments_to_plot, test_dataset_size):
            input_audio, target_audio = test_dataset[i]
            
            # Move to GPU if available
            if gpu_available:
                input_audio = input_audio.cuda()
                target_audio = target_audio.cuda()
            
            # Run inference
            with torch.no_grad():
                pred_audio = model(input_audio.unsqueeze(0).unsqueeze(1))
                pred_audio = pred_audio.squeeze(0).squeeze(0)
            
            # Move back to CPU and convert to numpy
            input_audio_np = input_audio.cpu().numpy()
            target_audio_np = target_audio.cpu().numpy()
            pred_audio_np = pred_audio.cpu().numpy()
            
            segments_to_plot.append({
                'input': input_audio_np,
                'target': target_audio_np,
                'pred': pred_audio_np,
                'segment_idx': i
            })
        
        # Create time axis (in seconds)
        time_axis = np.arange(len(segments_to_plot[0]['input'])) / sample_rate
        
        # Create figure with subplots: 3 rows (one per segment) x 3 columns (input, target, pred)
        fig, axes = plt.subplots(num_segments_to_plot, 3, figsize=(15, 4 * num_segments_to_plot), dpi=300)
        if num_segments_to_plot == 1:
            axes = axes.reshape(1, -1)
        
        # Compute global y range for consistent scaling
        all_values = np.concatenate([
            seg['input'] for seg in segments_to_plot
        ] + [
            seg['target'] for seg in segments_to_plot
        ] + [
            seg['pred'] for seg in segments_to_plot
        ])
        y_min, y_max = float(np.nanmin(all_values)), float(np.nanmax(all_values))
        if y_max <= y_min:
            y_min, y_max = y_min - 1e-6, y_max + 1e-6
        
        # Plot each segment
        for seg_idx, seg_data in enumerate(segments_to_plot):
            # Plot input
            axes[seg_idx, 0].plot(time_axis, seg_data['input'], 'b-', linewidth=0.5, alpha=0.7)
            axes[seg_idx, 0].set_title(f'Segment {seg_data["segment_idx"]} - Input', fontsize=10)
            axes[seg_idx, 0].set_ylabel('Amplitude', fontsize=9)
            axes[seg_idx, 0].set_ylim(y_min, y_max)
            axes[seg_idx, 0].grid(True, alpha=0.3)
            
            # Plot target
            axes[seg_idx, 1].plot(time_axis, seg_data['target'], 'g-', linewidth=0.5, alpha=0.7)
            axes[seg_idx, 1].set_title(f'Segment {seg_data["segment_idx"]} - Target', fontsize=10)
            axes[seg_idx, 1].set_ylabel('Amplitude', fontsize=9)
            axes[seg_idx, 1].set_ylim(y_min, y_max)
            axes[seg_idx, 1].grid(True, alpha=0.3)
            
            # Plot prediction
            axes[seg_idx, 2].plot(time_axis, seg_data['pred'], 'r-', linewidth=0.5, alpha=0.7)
            axes[seg_idx, 2].set_title(f'Segment {seg_data["segment_idx"]} - Prediction', fontsize=10)
            axes[seg_idx, 2].set_ylabel('Amplitude', fontsize=9)
            axes[seg_idx, 2].set_ylim(y_min, y_max)
            axes[seg_idx, 2].grid(True, alpha=0.3)
        
        # Set x-axis label only on bottom row
        for col in range(3):
            axes[-1, col].set_xlabel('Time (s)', fontsize=9)
        
        plt.tight_layout()
        
        # Log to wandb (figure already has dpi=300 set)
        wandb.log({
            "test/waveform_plots": wandb.Image(fig)
        })
        print(f"✓ Logged waveform plots to WandB (last {num_segments_to_plot} segments)")
        
        plt.close(fig)
    else:
        print("⚠ No test segments available for plotting")
    
    # Finish wandb run
    wandb.finish()
    print("✓ WandB run finished")
    
    return {
        "final_train_loss": avg_train_loss,
        "final_val_loss": avg_val_loss,
        "final_test_loss": avg_test_loss,
        "num_epochs": num_epochs,
        "train_batches": len(dataloader),
        "val_batches": len(val_dataloader),
        "test_batches": len(test_dataloader),
    }


@app.local_entrypoint()
def train(
    root_dir: str = "/data/signaltrain/SignalTrain_LA2A_Dataset_1.1",
    subset_dir_name: str = "Train",
    subset: str = "train",
    clip_idx: int = 263,
    segment_length: int = 65536,
    sample_rate: int = 44100,
    batch_size: int = 1,
    val_batch_size: int = 1,
    test_batch_size: int = 1,
    model_name: str = "residual_lstm",
    hidden_size: int = 64,
    num_layers: int = 2,
    block_size: int = 2048,
    learning_rate: float = 0.0001,
    num_epochs: int = 1,
    train_loss_type: str = "ESR+DC",
    val_loss_type: str = "MSE",
    # TCNModel-specific parameters
    nparams: int = 0,
    nblocks: int = 10,
    kernel_size: int = 3,
    dilation_growth: int = 1,
    channel_growth: int = 1,
    channel_width: int = 32,
    stack_size: int = 10,
    grouped: bool = False,
    causal: bool = False,
    skip_connections: bool = False,
):
    """
    Local entrypoint to run minimal training on Modal.
    
    Usage:
        # Run with default parameters (MSE for both train and val):
        modal run train_minimal.py::train
        
        # Run with custom parameters:
        modal run train_minimal.py::train --num-epochs 10 --batch-size 4 --clip-idx 100
        
        # Run with custom loss functions:
        modal run train_minimal.py::train --train-loss-type STFT --val-loss-type MAE
        
    Supported loss types: MAE, MSE, STFT, L1+STFT, ESR, DC, ESR+DC
    """
    result = train_minimal_model.remote(
        root_dir=root_dir,
        subset_dir_name=subset_dir_name,
        subset=subset,
        clip_idx=clip_idx,
        segment_length=segment_length,
        sample_rate=sample_rate,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        test_batch_size=test_batch_size,
        model_name=model_name,
        hidden_size=hidden_size,
        num_layers=num_layers,
        block_size=block_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        train_loss_type=train_loss_type,
        val_loss_type=val_loss_type,
        nparams=nparams,
        nblocks=nblocks,
        kernel_size=kernel_size,
        dilation_growth=dilation_growth,
        channel_growth=channel_growth,
        channel_width=channel_width,
        stack_size=stack_size,
        grouped=grouped,
        causal=causal,
        skip_connections=skip_connections,
    )
    
    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)
    import json
    print(json.dumps(result, indent=2))
    
    return result


@app.local_entrypoint()
def train_sweep(
    root_dir: str = "/data/signaltrain/SignalTrain_LA2A_Dataset_1.1",
    subset_dir_name: str = "Train",
    subset: str = "train",
    clip_idx: int = 263,
    segment_length: int = 65536,
    sample_rate: int = 44100,
    batch_size: int = 1,
    val_batch_size: int = 1,
    test_batch_size: int = 1,
    model_name: str = "residual_lstm",
    hidden_size: int = 64,
    num_layers: int = 2,
    block_size: int = 2048,
    learning_rate: float = 0.0001,
    num_epochs: int = 1,
    train_loss_type: str = "ESR+DC",
    val_loss_type: str = "MSE",
    nparams: int = 0,
    nblocks: int = 10,
    kernel_size: int = 3,
    dilation_growth: int = 1,
    channel_growth: int = 1,
    channel_width: int = 32,
    stack_size: int = 10,
    grouped: bool = False,
    causal: bool = False,
    skip_connections: bool = False,
    wandb_project: str = "neural-profiler",
):
    """
    Local entrypoint for Weights & Biases sweeps (see sweep_train_minimal.yaml).

    Intended command (created by ``wandb sweep sweep_train_minimal.yaml``):

        wandb agent <entity>/neural-profiler/<sweep_id>

    Each trial runs this entrypoint on your machine: it calls ``wandb.init()`` so the
    sampled hyperparameters appear in ``wandb.config``, then launches
    ``train_minimal_model`` on Modal with the same run id so all logs attach to
    that sweep trial.

    Fixed (non-swept) options can still be overridden via Modal CLI, e.g.:

        modal run train_minimal.py::train_sweep --num-epochs 5 --clip-idx 100
    """
    import wandb

    base_kwargs = {
        "root_dir": root_dir,
        "subset_dir_name": subset_dir_name,
        "subset": subset,
        "clip_idx": clip_idx,
        "segment_length": segment_length,
        "sample_rate": sample_rate,
        "batch_size": batch_size,
        "val_batch_size": val_batch_size,
        "test_batch_size": test_batch_size,
        "model_name": model_name,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "block_size": block_size,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "train_loss_type": train_loss_type,
        "val_loss_type": val_loss_type,
        "nparams": nparams,
        "nblocks": nblocks,
        "kernel_size": kernel_size,
        "dilation_growth": dilation_growth,
        "channel_growth": channel_growth,
        "channel_width": channel_width,
        "stack_size": stack_size,
        "grouped": grouped,
        "causal": causal,
        "skip_connections": skip_connections,
    }

    run = wandb.init(project=wandb_project, config=base_kwargs)
    try:
        merged = _merge_sweep_config_into_train_kwargs(
            base_kwargs, dict(wandb.config)
        )
        train_minimal_model.remote(
            wandb_run_id=run.id,
            wandb_project=wandb_project,
            **merged,
        )
    finally:
        wandb.finish()


# Allow running locally for testing (when not using Modal)
if __name__ == "__main__":
    # Dataset path - update to your SignalTrain LA2A root directory
    ROOT_DIR = "/home/shreyan/Documents/DATASETS/SignalTrain_LA2A_Dataset_1.1"
    SUBSET_DIR_NAME = "Train"
    SUBSET = "train"
    CLIP_IDX = 263  # Index of the clip to load
    SEGMENT_LENGTH = 65536  # ~1.5 seconds at 44.1kHz
    SAMPLE_RATE = 44100

    dataset = SignalTrainLA2ADatasetSingle(ROOT_DIR, SUBSET_DIR_NAME, SUBSET, CLIP_IDX, SEGMENT_LENGTH, SAMPLE_RATE)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)

    val_dataset = SignalTrainLA2ADatasetSingle(ROOT_DIR, SUBSET_DIR_NAME, "val", CLIP_IDX, SEGMENT_LENGTH, SAMPLE_RATE)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_dataset = SignalTrainLA2ADatasetSingle(ROOT_DIR, SUBSET_DIR_NAME, "test", CLIP_IDX, SEGMENT_LENGTH, SAMPLE_RATE)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = ResidualLSTMmodeler(input_size=1, hidden_size=128, output_size=1, block_size=2048)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1):
        for idx, batch in enumerate(dataloader):
            input_audio, target_audio = batch
            optimizer.zero_grad()
            pred_audio = model(input_audio.unsqueeze(1))
            # print(input_audio.unsqueeze(1).shape, pred_audio.shape, target_audio.unsqueeze(1).shape)
            loss = torch.nn.functional.mse_loss(pred_audio, target_audio.unsqueeze(1))
            print(f"Batch {idx}, Loss: {loss}")
            loss.backward()
            optimizer.step()

        val_loss = 0.0
        for idx, batch in enumerate(val_dataloader):
            input_audio, target_audio = batch
            pred_audio = model(input_audio.unsqueeze(1))
            val_loss += torch.nn.functional.mse_loss(pred_audio, target_audio.unsqueeze(1))
            print(f"Batch {idx}, Val Loss: {val_loss}")
        print(f"Val Loss: {val_loss/(idx+1)}")

        test_loss = 0.0
        for idx, batch in enumerate(test_dataloader):
            input_audio, target_audio = batch
            pred_audio = model(input_audio.unsqueeze(1))
            test_loss += torch.nn.functional.mse_loss(pred_audio, target_audio.unsqueeze(1))
            print(f"Batch {idx}, Test Loss: {test_loss}")
        print(f"Test Loss: {test_loss/(idx+1)}")