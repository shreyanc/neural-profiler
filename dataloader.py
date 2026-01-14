"""
SignalTrain LA2A Dataset Dataloader and Helper Functions

This module provides a comprehensive dataloader for the SignalTrain LA2A Dataset,
including data preprocessing, augmentation, and validation utilities.
"""

import os
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Union, List, Dict, Any
import soundfile as sf
import librosa
from pathlib import Path
import logging
from dataclasses import dataclass
import random
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration class for SignalTrain LA2A Dataset parameters."""
    root_dir: str
    train_subset: str = "Train"
    val_subset: str = "Val"
    test_subset: str = "Test"
    train_length: int = 65536  # ~1.5 seconds at 44.1kHz
    eval_length: int = 131072  # ~3 seconds at 44.1kHz
    sample_rate: int = 44100
    n_params: int = 2  # LA2A has 2 parameters: gain and ratio
    preload: bool = False
    half_precision: bool = False
    shuffle: bool = True
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True


class SignalTrainLA2ADataset(Dataset):
    """
    PyTorch Dataset for SignalTrain LA2A Dataset.
    
    The LA2A dataset contains:
    - Input audio signals (clean)
    - Target audio signals (processed through LA2A hardware)
    - Parameter values (gain, ratio) used for processing
    
    Args:
        root_dir: Path to the dataset root directory
        subset: Dataset subset ('train', 'val', 'test')
        length: Length of audio segments in samples
        preload: Whether to load all data into memory
        half: Whether to use half precision (float16)
        augment: Whether to apply data augmentation
    """
    
    def __init__(
        self,
        root_dir: str,
        subset: str = "Train",
        length: int = 65536,
        preload: bool = False,
        half: bool = False,
        augment: bool = False,
        sample_rate: int = 44100,
        n_params: int = 2
    ):
        self.root_dir = Path(root_dir)
        self.subset = subset
        self.length = length
        self.preload = preload
        self.half = half
        self.augment = augment
        self.sample_rate = sample_rate
        self.n_params = n_params

        # Validate dataset directory
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

        # Validate subset directory
        self.subset_dir = self.root_dir / self.subset
        if not self.subset_dir.exists():
            raise FileNotFoundError(f"Subset directory not found: {self.subset_dir}")

        # Build index of paired input/target files following the notebook naming convention:
        # - Input:  input_<num>_.wav
        # - Target: target_<num>_LA2A_<s1>__<s2>__<s3>.wav
        self.pairs = self._find_paired_files()
        if not self.pairs:
            raise FileNotFoundError(
                f"No paired input/target files found in {self.subset_dir}. "
                "Expected files like 'input_<num>_.wav' and "
                "'target_<num>_LA2A_<s1>__<s2>__<s3>.wav'."
            )

        logger.info(
            f"Found {len(self.pairs)} paired input/target files in {self.subset} subset "
            f"under {self.subset_dir}"
        )

        if self.preload:
            logger.warning(
                "Preloading is not supported for streaming large WAV files; "
                "data will be loaded on demand instead."
            )

    def _find_paired_files(self) -> List[Dict[str, Any]]:
        """
        Find all paired input and target audio files in the subset directory.

        This mirrors the pairing logic used in audio_visualization.ipynb and
        treats that notebook as the source of truth for file naming.
        """
        input_pattern = re.compile(r"input_(\d+)_\.wav")
        target_pattern = re.compile(r"target_(\d+)_LA2A_([^_]+)__([^_]+)__([^_]+)\.wav")

        input_files: Dict[int, Path] = {}
        target_files: Dict[int, Tuple[Path, Tuple[str, str, str]]] = {}

        # Find all input files
        for file_path in self.subset_dir.glob("input_*.wav"):
            match = input_pattern.match(file_path.name)
            if match:
                num = int(match.group(1))
                input_files[num] = file_path

        # Find all target files
        for file_path in self.subset_dir.glob("target_*_LA2A_*.wav"):
            match = target_pattern.match(file_path.name)
            if match:
                num = int(match.group(1))
                s1, s2, s3 = match.groups()[1:]
                target_files[num] = (file_path, (s1, s2, s3))

        pairs: List[Dict[str, Any]] = []
        for num in sorted(set(input_files.keys()) & set(target_files.keys())):
            input_path = input_files[num]
            target_path, states = target_files[num]

            # Read metadata from the input file (assume input/target share SR & length)
            with sf.SoundFile(str(input_path)) as f:
                sample_rate = f.samplerate
                frames = len(f)
                duration = frames / sample_rate

            pairs.append(
                {
                    "num": num,
                    "input_path": input_path,
                    "target_path": target_path,
                    "states": states,  # (s1, s2, s3) from filename
                    "sample_rate": sample_rate,
                    "frames": frames,
                    "duration": duration,
                }
            )

        if not pairs:
            logger.warning(
                f"No valid input/target pairs found in {self.subset_dir}. "
                "Check that files follow the expected naming convention."
            )

        return pairs

    @staticmethod
    def _read_audio_segment(
        file_path: Path,
        start_sample: int,
        num_samples: int,
    ) -> np.ndarray:
        """
        Read a segment of audio from a WAV file given a start sample and length.

        This matches the on-demand loading style used in audio_visualization.ipynb.
        """
        with sf.SoundFile(str(file_path)) as f:
            total_frames = len(f)

            # Clamp start_sample and num_samples to valid range
            start_sample = max(0, min(start_sample, total_frames))
            num_samples = max(0, min(num_samples, total_frames - start_sample))

            f.seek(start_sample)
            audio_data = f.read(num_samples, dtype="float32")

        # If stereo, take first channel
        if audio_data.ndim > 1:
            audio_data = audio_data[:, 0]

        return audio_data

    def _encode_states_to_params(self, states: Tuple[str, str, str]) -> np.ndarray:
        """
        Convert LA2A state strings from the filename into a numeric parameter vector.

        We keep this flexible so that self.n_params controls the final
        dimensionality:
        - Extract numeric values from each state token when possible
        - Truncate or pad with zeros to match self.n_params
        """

        def _extract_numeric(token: str) -> float:
            # Try to extract the first number (e.g. "3c" -> 3.0, "0" -> 0.0)
            match = re.search(r"[-+]?\d*\.?\d+", token)
            if match:
                try:
                    return float(match.group())
                except ValueError:
                    return 0.0
            return 0.0

        full_params = np.array([_extract_numeric(s) for s in states], dtype=np.float32)

        if self.n_params <= 0:
            return np.zeros(0, dtype=np.float32)

        if self.n_params <= len(full_params):
            return full_params[: self.n_params]

        # Pad with zeros if more parameters are requested than available
        padded = np.zeros(self.n_params, dtype=np.float32)
        padded[: len(full_params)] = full_params
        return padded
    
    def _apply_augmentation(self, input_audio: np.ndarray, target_audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation to audio samples."""
        # Random gain variation
        if random.random() < 0.3:
            gain_factor = random.uniform(0.8, 1.2)
            input_audio *= gain_factor
            target_audio *= gain_factor
        
        # Random phase inversion
        if random.random() < 0.1:
            input_audio *= -1
            target_audio *= -1
        
        # Random time stretching (slight)
        if random.random() < 0.2:
            stretch_factor = random.uniform(0.95, 1.05)
            input_audio = librosa.effects.time_stretch(input_audio, rate=stretch_factor)
            target_audio = librosa.effects.time_stretch(target_audio, rate=stretch_factor)
        
        return input_audio, target_audio
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Returns:
            Tuple of (input_audio, target_audio, parameters)
        """
        pair = self.pairs[idx]

        total_frames = pair["frames"]
        segment_length = self.length

        # Choose a random segment start (in samples). For very short files,
        # we read as much as possible and pad later.
        if total_frames <= segment_length:
            start_sample = 0
            num_samples = total_frames
        else:
            max_start = total_frames - segment_length
            start_sample = random.randint(0, max_start)
            num_samples = segment_length

        # Load audio segments from disk on demand (streaming, not full file)
        input_audio = self._read_audio_segment(
            pair["input_path"],
            start_sample=start_sample,
            num_samples=num_samples,
        )
        target_audio = self._read_audio_segment(
            pair["target_path"],
            start_sample=start_sample,
            num_samples=num_samples,
        )

        # Ensure mono NumPy arrays
        input_audio = np.asarray(input_audio, dtype=np.float32).flatten()
        target_audio = np.asarray(target_audio, dtype=np.float32).flatten()

        # Pad or trim to exactly self.length samples
        def _fix_length(x: np.ndarray, length: int) -> np.ndarray:
            if x.shape[0] == length:
                return x
            if x.shape[0] < length:
                return np.pad(x, (0, length - x.shape[0]), mode="constant")
            # If slightly longer due to rounding, trim
            return x[:length]

        input_audio = _fix_length(input_audio, self.length)
        target_audio = _fix_length(target_audio, self.length)

        # Derive parameter vector from filename states
        params = self._encode_states_to_params(pair["states"])
        
        # Apply augmentation if enabled
        if self.augment and self.subset == 'train':
            input_audio, target_audio = self._apply_augmentation(input_audio, target_audio)
        
        # Convert to tensors
        input_tensor = torch.from_numpy(input_audio.astype(np.float32))
        target_tensor = torch.from_numpy(target_audio.astype(np.float32))
        params_tensor = torch.from_numpy(params.astype(np.float32))
        
        # Add channel dimension
        input_tensor = input_tensor.unsqueeze(0)
        target_tensor = target_tensor.unsqueeze(0)
        
        # Convert to half precision if requested
        if self.half:
            input_tensor = input_tensor.half()
            target_tensor = target_tensor.half()
            params_tensor = params_tensor.half()
        
        return input_tensor, target_tensor, params_tensor


def create_dataloaders(config: DatasetConfig) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create DataLoader instances for training, validation, and optionally test sets.
    
    Args:
        config: Dataset configuration
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = SignalTrainLA2ADataset(
        root_dir=config.root_dir,
        subset=config.train_subset,
        length=config.train_length,
        preload=config.preload,
        half=config.half_precision,
        augment=True,  # Enable augmentation for training
        sample_rate=config.sample_rate,
        n_params=config.n_params
    )
    
    val_dataset = SignalTrainLA2ADataset(
        root_dir=config.root_dir,
        subset=config.val_subset,
        length=config.eval_length,
        preload=config.preload,
        half=config.half_precision,
        augment=False,  # No augmentation for validation
        sample_rate=config.sample_rate,
        n_params=config.n_params
    )
    
    # Create test dataset if test subset exists
    test_dataset = None
    test_loader = None
    test_subset_dir = Path(config.root_dir) / config.test_subset
    if test_subset_dir.exists():
        test_dataset = SignalTrainLA2ADataset(
            root_dir=config.root_dir,
            subset=config.test_subset,
            length=config.eval_length,
            preload=config.preload,
            half=config.half_precision,
            augment=False,
            sample_rate=config.sample_rate,
            n_params=config.n_params
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True  # Drop last incomplete batch for consistent training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False
    )
    
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=False
        )
    
    logger.info(f"Created dataloaders: train={len(train_dataset)}, val={len(val_dataset)}")
    if test_dataset:
        logger.info(f"Test dataset size: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def validate_dataset_structure(root_dir: str) -> Dict[str, Any]:
    """
    Validate the structure of the SignalTrain LA2A Dataset.
    
    Args:
        root_dir: Path to the dataset root directory
        
    Returns:
        Dictionary containing validation results
    """
    root_path = Path(root_dir)
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    # Check if root directory exists
    if not root_path.exists():
        results['valid'] = False
        results['errors'].append(f"Root directory does not exist: {root_dir}")
        return results
    
    # Check for required subsets
    required_subsets = ['Train', 'Val']
    for subset in required_subsets:
        subset_dir = root_path / subset
        if not subset_dir.exists():
            results['valid'] = False
            results['errors'].append(f"Missing subset directory: {subset}")
        else:
            # Count files in subset
            h5_files = list(subset_dir.glob("*.h5"))
            wav_files = list(subset_dir.glob("*.wav"))
            results['stats'][subset] = {
                'h5_files': len(h5_files),
                'wav_files': len(wav_files),
                'total_files': len(h5_files) + len(wav_files)
            }
    
    # Check for test subset (optional)
    test_dir = root_path / 'Test'
    if test_dir.exists():
        h5_files = list(test_dir.glob("*.h5"))
        wav_files = list(test_dir.glob("*.wav"))
        results['stats']['test'] = {
            'h5_files': len(h5_files),
            'wav_files': len(wav_files),
            'total_files': len(h5_files) + len(wav_files)
        }
    
    return results


def get_dataset_info(dataset: SignalTrainLA2ADataset) -> Dict[str, Any]:
    """
    Get information about a dataset instance.
    
    Args:
        dataset: Dataset instance
        
    Returns:
        Dictionary containing dataset information
    """
    info = {
        'subset': dataset.subset,
        'length': dataset.length,
        'sample_rate': dataset.sample_rate,
        'n_params': dataset.n_params,
        'preload': dataset.preload,
        'half_precision': dataset.half,
        'augment': dataset.augment,
        'num_files': len(dataset.data_files),
        'total_samples': len(dataset)
    }
    
    return info


def normalize_audio(audio: torch.Tensor, method: str = 'peak') -> torch.Tensor:
    """
    Normalize audio tensor.
    
    Args:
        audio: Audio tensor
        method: Normalization method ('peak', 'rms', 'lufs')
        
    Returns:
        Normalized audio tensor
    """
    if method == 'peak':
        peak = torch.max(torch.abs(audio))
        if peak > 0:
            return audio / peak
    elif method == 'rms':
        rms = torch.sqrt(torch.mean(audio ** 2))
        if rms > 0:
            return audio / rms
    elif method == 'lufs':
        # Simple LUFS-like normalization (approximation)
        rms = torch.sqrt(torch.mean(audio ** 2))
        if rms > 0:
            target_lufs = -23.0  # Broadcast standard
            current_lufs = 20 * torch.log10(rms)
            gain = target_lufs - current_lufs
            return audio * (10 ** (gain / 20))
    
    return audio


def compute_audio_metrics(input_audio: torch.Tensor, target_audio: torch.Tensor) -> Dict[str, float]:
    """
    Compute audio quality metrics between input and target audio.
    
    Args:
        input_audio: Input audio tensor
        target_audio: Target audio tensor
        
    Returns:
        Dictionary containing computed metrics
    """
    # Ensure tensors are on the same device
    if input_audio.device != target_audio.device:
        target_audio = target_audio.to(input_audio.device)
    
    # Compute MSE
    mse = F.mse_loss(input_audio, target_audio).item()
    
    # Compute MAE
    mae = F.l1_loss(input_audio, target_audio).item()
    
    # Compute SNR (Signal-to-Noise Ratio)
    signal_power = torch.mean(input_audio ** 2)
    noise_power = torch.mean((input_audio - target_audio) ** 2)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-8)).item()
    
    # Compute correlation coefficient
    correlation = torch.corrcoef(torch.stack([input_audio.flatten(), target_audio.flatten()]))[0, 1].item()
    
    return {
        'mse': mse,
        'mae': mae,
        'snr': snr,
        'correlation': correlation
    }


# Example usage and testing functions
def test_dataloader(config: DatasetConfig):
    """Test the dataloader with a sample configuration."""
    try:
        # Validate dataset structure
        validation_results = validate_dataset_structure(config.root_dir)
        if not validation_results['valid']:
            logger.error("Dataset validation failed:")
            for error in validation_results['errors']:
                logger.error(f"  - {error}")
            return False
        
        logger.info("Dataset validation passed!")
        logger.info(f"Dataset stats: {validation_results['stats']}")
        
        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(config)
        
        # Test loading a batch
        logger.info("Testing data loading...")
        for batch_idx, (input_audio, target_audio, params) in enumerate(train_loader):
            logger.info(f"Batch {batch_idx}: input shape={input_audio.shape}, "
                       f"target shape={target_audio.shape}, params shape={params.shape}")
            
            # Test audio metrics
            metrics = compute_audio_metrics(input_audio[0], target_audio[0])
            logger.info(f"Sample metrics: {metrics}")
            
            if batch_idx >= 2:  # Test only first few batches
                break
        
        logger.info("Dataloader test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Dataloader test failed: {e}")
        return False


if __name__ == "__main__":
    # Example configuration
    config = DatasetConfig(
        root_dir="/home/shreyan/Documents/DATASETS/SignalTrain_LA2A_Dataset_1.1",
        train_length=65536,
        eval_length=131072,
        batch_size=4,
        num_workers=2
    )
    
    # Test the dataloader
    test_dataloader(config)
