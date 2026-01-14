# SignalTrain LA2A Dataset Dataloader

A comprehensive PyTorch dataloader and helper functions for the SignalTrain LA2A Dataset, designed for neural audio processing and hardware emulation tasks.

## Overview

The SignalTrain LA2A Dataset contains:
- **Input audio signals**: Clean, unprocessed audio
- **Target audio signals**: Audio processed through LA2A hardware compressor
- **Parameter values**: Gain and ratio settings used for processing

This dataloader provides:
- Efficient data loading with support for HDF5 and WAV formats
- Data augmentation for improved model generalization
- Audio preprocessing and normalization utilities
- Comprehensive configuration management
- Built-in validation and error handling

## Installation

The dataloader requires the following dependencies (already included in `pyproject.toml`):

```bash
pip install numpy librosa soundfile h5py scipy pyyaml torch torchvision
```

## Quick Start

### Basic Usage

```python
from dataloader import SignalTrainLA2ADataset, DatasetConfig, create_dataloaders

# Create configuration
config = DatasetConfig(
    root_dir="/path/to/signaltrain/la2a/dataset",
    train_length=65536,  # ~1.5 seconds at 44.1kHz
    eval_length=131072,  # ~3 seconds at 44.1kHz
    batch_size=8,
    num_workers=4
)

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(config)

# Use in training loop
for input_audio, target_audio, params in train_loader:
    # input_audio: [batch_size, 1, length] - clean audio
    # target_audio: [batch_size, 1, length] - processed audio
    # params: [batch_size, 2] - [gain, ratio] parameters
    pass
```

### Using Configuration Files

```python
from config import get_default_config, load_config_from_yaml

# Load from YAML file
config = load_config_from_yaml("config.yaml")

# Or use predefined configurations
config = get_default_config("/path/to/dataset")
config = get_small_config("/path/to/dataset")  # For quick testing
config = get_large_config("/path/to/dataset")  # For full training
```

## Dataset Structure

The dataloader expects the following directory structure:

```
signaltrain_la2a_dataset/
├── train/
│   ├── sample1.h5
│   ├── sample2.h5
│   └── ...
├── val/
│   ├── sample1.h5
│   ├── sample2.h5
│   └── ...
└── test/  # Optional
    ├── sample1.h5
    ├── sample2.h5
    └── ...
```

### HDF5 Format

Each HDF5 file should contain:
- `input`: Input audio signal (float32, 1D array)
- `target`: Target audio signal (float32, 1D array)
- `params`: Parameter values (float32, 1D array of length 2)

### WAV Format (Alternative)

For WAV files, the dataloader expects:
- `{filename}_input.wav`: Input audio
- `{filename}_target.wav`: Target audio
- `{filename}_params.npy`: Parameter values

## Features

### Data Augmentation

The dataloader includes several augmentation techniques:

```python
dataset = SignalTrainLA2ADataset(
    root_dir="/path/to/dataset",
    subset="train",
    augment=True,  # Enable augmentation
    length=65536
)
```

Available augmentations:
- **Gain variation**: Random gain changes (0.8x to 1.2x)
- **Phase inversion**: Random polarity inversion
- **Time stretching**: Slight time stretching (0.95x to 1.05x)

### Audio Preprocessing

```python
from dataloader import normalize_audio, compute_audio_metrics

# Normalize audio
normalized_audio = normalize_audio(audio, method='peak')  # or 'rms', 'lufs'

# Compute quality metrics
metrics = compute_audio_metrics(input_audio, target_audio)
# Returns: {'mse': float, 'mae': float, 'snr': float, 'correlation': float}
```

### Dataset Validation

```python
from dataloader import validate_dataset_structure

# Validate dataset structure
results = validate_dataset_structure("/path/to/dataset")
if results['valid']:
    print("Dataset is valid!")
    print(f"Stats: {results['stats']}")
else:
    print("Dataset validation failed:")
    for error in results['errors']:
        print(f"  - {error}")
```

## Configuration Options

### DatasetConfig

```python
@dataclass
class DatasetConfig:
    root_dir: str                    # Path to dataset
    train_subset: str = "train"      # Training subset name
    val_subset: str = "val"          # Validation subset name
    test_subset: str = "test"        # Test subset name
    train_length: int = 65536        # Training audio length (samples)
    eval_length: int = 131072        # Evaluation audio length (samples)
    sample_rate: int = 44100         # Audio sample rate
    n_params: int = 2                # Number of parameters
    preload: bool = False            # Load all data into memory
    half_precision: bool = False     # Use float16 precision
    shuffle: bool = True             # Shuffle training data
    batch_size: int = 8              # Batch size
    num_workers: int = 4             # Number of worker processes
    pin_memory: bool = True          # Pin memory for GPU transfer
```

### ModelConfig

```python
@dataclass
class ModelConfig:
    model_type: str = "simple_cnn"   # Model architecture type
    input_length: int = 65536        # Input audio length
    n_params: int = 2                # Number of parameters
    # ... additional model-specific parameters
```

### TrainingConfig

```python
@dataclass
class TrainingConfig:
    num_epochs: int = 100            # Number of training epochs
    learning_rate: float = 1e-4      # Learning rate
    optimizer: str = "adam"          # Optimizer type
    scheduler: str = "reduce_on_plateau"  # Learning rate scheduler
    loss_function: str = "mse"       # Loss function
    # ... additional training parameters
```

## Examples

### Complete Training Example

See `example_usage.py` for a complete training example with:
- Model definition
- Training loop
- Validation
- Metrics computation

### Synthetic Data Demo

If you don't have the actual dataset, you can run the demo with synthetic data:

```python
python example_usage.py
```

This will create synthetic audio data and demonstrate the dataloader functionality.

### Configuration Examples

```python
# Small config for quick testing
config = get_small_config("/path/to/dataset")

# Large config for full training
config = get_large_config("/path/to/dataset")

# Custom config
config = DatasetConfig(
    root_dir="/path/to/dataset",
    train_length=32768,
    batch_size=4,
    num_workers=2,
    preload=True
)
```

## Performance Tips

1. **Preloading**: Set `preload=True` if you have sufficient RAM to load the entire dataset
2. **Workers**: Increase `num_workers` for faster data loading (typically 4-8)
3. **Pin Memory**: Keep `pin_memory=True` when using GPU
4. **Half Precision**: Use `half_precision=True` to reduce memory usage
5. **Batch Size**: Adjust based on your GPU memory

## Error Handling

The dataloader includes comprehensive error handling:

- **File not found**: Clear error messages for missing files
- **Invalid data**: Validation of audio length and parameter count
- **Memory issues**: Graceful handling of memory constraints
- **Format errors**: Support for both HDF5 and WAV formats

## API Reference

### SignalTrainLA2ADataset

```python
class SignalTrainLA2ADataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        subset: str = "train",
        length: int = 65536,
        preload: bool = False,
        half: bool = False,
        augment: bool = False,
        sample_rate: int = 44100,
        n_params: int = 2
    ):
        pass
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (input_audio, target_audio, parameters)"""
        pass
```

### Helper Functions

```python
def create_dataloaders(config: DatasetConfig) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Create train, validation, and optional test dataloaders"""

def validate_dataset_structure(root_dir: str) -> Dict[str, Any]:
    """Validate dataset directory structure and files"""

def get_dataset_info(dataset: SignalTrainLA2ADataset) -> Dict[str, Any]:
    """Get information about a dataset instance"""

def normalize_audio(audio: torch.Tensor, method: str = 'peak') -> torch.Tensor:
    """Normalize audio tensor using specified method"""

def compute_audio_metrics(input_audio: torch.Tensor, target_audio: torch.Tensor) -> Dict[str, float]:
    """Compute audio quality metrics between input and target"""
```

## Troubleshooting

### Common Issues

1. **"Dataset directory not found"**
   - Check that the `root_dir` path is correct
   - Ensure the directory contains `train/` and `val/` subdirectories

2. **"No data files found"**
   - Verify that HDF5 or WAV files exist in the subset directories
   - Check file permissions

3. **Memory errors**
   - Reduce `batch_size` or `num_workers`
   - Set `preload=False`
   - Use `half_precision=True`

4. **Slow data loading**
   - Increase `num_workers`
   - Set `preload=True` if you have sufficient RAM
   - Use SSD storage for the dataset

### Getting Help

For issues or questions:
1. Check the error messages and logs
2. Validate your dataset structure using `validate_dataset_structure()`
3. Test with the synthetic data demo first
4. Review the example usage scripts

## License

This dataloader is part of the Neural Profiler project. See the main project README for license information.
