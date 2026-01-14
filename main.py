"""
Neural Profiler - Main Entry Point

This module demonstrates the usage of the SignalTrain LA2A Dataset dataloader
and provides a simple training example.
"""

import torch
import logging
from pathlib import Path

from dataloader import (
    SignalTrainLA2ADataset,
    DatasetConfig,
    create_dataloaders,
    validate_dataset_structure,
    get_dataset_info,
    compute_audio_metrics
)
from config import get_default_config, get_small_config
from example_usage import SimpleAudioModel, train_epoch, validate_epoch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the neural profiler."""
    logger.info("Neural Profiler - SignalTrain LA2A Dataset Demo")
    
    # Configuration
    dataset_root = "/path/to/signaltrain/la2a/dataset"  # Update this path
    
    # Check if dataset exists
    if not Path(dataset_root).exists():
        logger.warning(f"Dataset not found at {dataset_root}")
        logger.info("Please update the dataset_root path in main.py")
        logger.info("You can also run the example_usage.py script for a demo with synthetic data")
        return
    
    # Get configuration
    config = get_small_config(dataset_root)  # Use small config for demo
    
    logger.info(f"Using configuration: {config.experiment_name}")
    logger.info(f"Description: {config.description}")
    
    # Validate dataset
    validation_results = validate_dataset_structure(config.dataset.root_dir)
    if not validation_results['valid']:
        logger.error("Dataset validation failed:")
        for error in validation_results['errors']:
            logger.error(f"  - {error}")
        return
    
    logger.info("Dataset validation passed!")
    logger.info(f"Dataset stats: {validation_results['stats']}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(config.dataset)
    
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
        input_length=config.dataset.train_length,
        n_params=config.dataset.n_params
    ).to(device)
    
    # Set up training
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        patience=config.training.scheduler_patience, 
        factor=config.training.scheduler_factor
    )
    
    # Quick training demo
    logger.info("Running quick training demo...")
    
    for epoch in range(3):  # Just a few epochs for demo
        logger.info(f"Epoch {epoch+1}/3")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log results
        logger.info(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        logger.info(f"Val Metrics: {val_metrics}")
    
    logger.info("Demo completed!")
    logger.info("For full training, use the example_usage.py script or create your own training loop")


def demo_data_loading():
    """Demonstrate basic data loading functionality."""
    logger.info("Data Loading Demo")
    
    # Configuration
    config = DatasetConfig(
        root_dir="/path/to/signaltrain/la2a/dataset",  # Update this path
        train_length=32768,  # Shorter for demo
        batch_size=2,
        num_workers=0  # No multiprocessing for demo
    )
    
    if not Path(config.root_dir).exists():
        logger.info("Dataset not found. Please update the path in main.py")
        return
    
    # Create dataset
    dataset = SignalTrainLA2ADataset(
        root_dir=config.root_dir,
        subset="train",
        length=config.train_length,
        augment=True
    )
    
    logger.info(f"Dataset created with {len(dataset)} samples")
    
    # Load a sample
    input_audio, target_audio, params = dataset[0]
    logger.info(f"Sample loaded:")
    logger.info(f"  Input shape: {input_audio.shape}")
    logger.info(f"  Target shape: {target_audio.shape}")
    logger.info(f"  Params shape: {params.shape}")
    logger.info(f"  Params: {params}")
    
    # Compute metrics
    metrics = compute_audio_metrics(input_audio, target_audio)
    logger.info(f"  Sample metrics: {metrics}")


if __name__ == "__main__":
    # Run the main demo
    main()
    
    # Uncomment to run data loading demo
    # demo_data_loading()
