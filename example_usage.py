"""
Example usage of the SignalTrain LA2A Dataset Dataloader

This script demonstrates how to use the dataloader for training neural audio models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from pathlib import Path

from dataloader import (
    SignalTrainLA2ADataset,
    DatasetConfig,
    create_dataloaders,
    validate_dataset_structure,
    get_dataset_info,
    compute_audio_metrics,
    normalize_audio
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleAudioModel(nn.Module):
    """Simple example model for audio processing."""
    
    def __init__(self, input_length: int = 65536, n_params: int = 2):
        super().__init__()
        self.input_length = input_length
        self.n_params = n_params
        
        # Simple 1D CNN architecture
        self.conv1 = nn.Conv1d(1, 64, kernel_size=15, padding=7)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=15, padding=7)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=15, padding=7)
        self.conv4 = nn.Conv1d(64, 1, kernel_size=15, padding=7)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        # Parameter conditioning
        self.param_embedding = nn.Linear(n_params, 64)
        
    def forward(self, x, params):
        # Parameter conditioning
        param_features = self.param_embedding(params)
        
        # Audio processing
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.relu(self.conv3(x))
        x = self.dropout(x)
        x = self.conv4(x)
        
        return x


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (input_audio, target_audio, params) in enumerate(train_loader):
        # Move data to device
        input_audio = input_audio.to(device)
        target_audio = target_audio.to(device)
        params = params.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(input_audio, params)
        loss = criterion(output, target_audio)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 10 == 0:
            logger.info(f"Batch {batch_idx}, Loss: {loss.item():.6f}")
    
    return total_loss / num_batches


def validate_epoch(model, val_loader, criterion, device):
    """Validate the model for one epoch."""
    model.eval()
    total_loss = 0.0
    total_metrics = {'mse': 0.0, 'mae': 0.0, 'snr': 0.0, 'correlation': 0.0}
    num_batches = 0
    
    with torch.no_grad():
        for input_audio, target_audio, params in val_loader:
            # Move data to device
            input_audio = input_audio.to(device)
            target_audio = target_audio.to(device)
            params = params.to(device)
            
            # Forward pass
            output = model(input_audio, params)
            loss = criterion(output, target_audio)
            
            total_loss += loss.item()
            
            # Compute metrics on first batch
            if num_batches == 0:
                metrics = compute_audio_metrics(output[0], target_audio[0])
                for key in total_metrics:
                    total_metrics[key] += metrics[key]
            
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_metrics = {key: val / min(num_batches, 1) for key, val in total_metrics.items()}
    
    return avg_loss, avg_metrics


def main():
    """Main training loop example."""
    
    # Configuration
    config = DatasetConfig(
        root_dir="/path/to/signaltrain/la2a/dataset",  # Update this path
        train_length=65536,
        eval_length=131072,
        batch_size=8,
        num_workers=4,
        preload=False,  # Set to True if you have enough RAM
        half_precision=False
    )
    
    # Check if dataset exists
    if not Path(config.root_dir).exists():
        logger.error(f"Dataset directory not found: {config.root_dir}")
        logger.info("Please update the root_dir in the config to point to your dataset")
        return
    
    # Validate dataset structure
    validation_results = validate_dataset_structure(config.root_dir)
    if not validation_results['valid']:
        logger.error("Dataset validation failed:")
        for error in validation_results['errors']:
            logger.error(f"  - {error}")
        return
    
    logger.info("Dataset validation passed!")
    logger.info(f"Dataset stats: {validation_results['stats']}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    # Get dataset info
    train_info = get_dataset_info(train_loader.dataset)
    val_info = get_dataset_info(val_loader.dataset)
    
    logger.info(f"Training dataset: {train_info}")
    logger.info(f"Validation dataset: {val_info}")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model
    model = SimpleAudioModel(
        input_length=config.train_length,
        n_params=config.n_params
    ).to(device)
    
    # Set up training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    num_epochs = 10
    best_val_loss = float('inf')
    
    logger.info("Starting training...")
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log results
        logger.info(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        logger.info(f"Val Metrics: {val_metrics}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            logger.info("Saved best model")
    
    logger.info("Training completed!")
    
    # Test on a few samples
    logger.info("Testing model on sample data...")
    model.eval()
    with torch.no_grad():
        for i, (input_audio, target_audio, params) in enumerate(val_loader):
            if i >= 3:  # Test only first 3 batches
                break
                
            input_audio = input_audio.to(device)
            target_audio = target_audio.to(device)
            params = params.to(device)
            
            output = model(input_audio, params)
            metrics = compute_audio_metrics(output[0], target_audio[0])
            
            logger.info(f"Sample {i+1} metrics: {metrics}")


def demo_data_loading():
    """Demonstrate basic data loading functionality."""
    
    config = DatasetConfig(
        root_dir="/path/to/signaltrain/la2a/dataset",  # Update this path
        train_length=32768,  # Shorter for demo
        batch_size=2,
        num_workers=0  # No multiprocessing for demo
    )
    
    if not Path(config.root_dir).exists():
        logger.info("Dataset not found. Creating a demo with synthetic data...")
        demo_with_synthetic_data()
        return
    
    # Create dataset
    dataset = SignalTrainLA2ADataset(
        root_dir=config.root_dir,
        subset="train",
        length=config.train_length,
        augment=True
    )
    
    logger.info(f"Dataset created with {len(dataset)} samples")
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # Load a batch
    for batch_idx, (input_audio, target_audio, params) in enumerate(dataloader):
        logger.info(f"Batch {batch_idx}:")
        logger.info(f"  Input shape: {input_audio.shape}")
        logger.info(f"  Target shape: {target_audio.shape}")
        logger.info(f"  Params shape: {params.shape}")
        logger.info(f"  Params range: {params.min():.3f} to {params.max():.3f}")
        
        # Compute metrics
        metrics = compute_audio_metrics(input_audio[0], target_audio[0])
        logger.info(f"  Sample metrics: {metrics}")
        
        if batch_idx >= 2:  # Show only first few batches
            break


def demo_with_synthetic_data():
    """Create a demo with synthetic data when real dataset is not available."""
    logger.info("Creating synthetic dataset for demonstration...")
    
    # Create synthetic data
    sample_rate = 44100
    duration = 2.0  # seconds
    length = int(sample_rate * duration)
    
    # Generate synthetic audio
    t = torch.linspace(0, duration, length)
    input_audio = torch.sin(2 * torch.pi * 440 * t) * 0.5  # 440 Hz sine wave
    
    # Simulate LA2A processing (simple compression)
    threshold = 0.3
    ratio = 4.0
    compressed = torch.where(
        torch.abs(input_audio) > threshold,
        torch.sign(input_audio) * (threshold + (torch.abs(input_audio) - threshold) / ratio),
        input_audio
    )
    
    # Add some harmonic distortion
    target_audio = compressed + 0.1 * torch.sin(2 * torch.pi * 880 * t)
    
    # Create synthetic parameters
    params = torch.tensor([0.5, 2.0])  # gain, ratio
    
    logger.info("Synthetic data created:")
    logger.info(f"  Input audio shape: {input_audio.shape}")
    logger.info(f"  Target audio shape: {target_audio.shape}")
    logger.info(f"  Parameters: {params}")
    
    # Compute metrics
    metrics = compute_audio_metrics(input_audio.unsqueeze(0), target_audio.unsqueeze(0))
    logger.info(f"  Metrics: {metrics}")
    
    # Test normalization
    normalized_input = normalize_audio(input_audio.unsqueeze(0), method='peak')
    normalized_target = normalize_audio(target_audio.unsqueeze(0), method='peak')
    
    logger.info(f"  Normalized input range: {normalized_input.min():.3f} to {normalized_input.max():.3f}")
    logger.info(f"  Normalized target range: {normalized_target.min():.3f} to {normalized_target.max():.3f}")


if __name__ == "__main__":
    # Run the demo
    demo_data_loading()
    
    # Uncomment to run full training example
    # main()
