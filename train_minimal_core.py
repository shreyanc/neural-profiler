"""Dataset, small LSTM models, and loss factory used by train_minimal (Modal).

Kept separate from train_minimal.py so the Modal entrypoint stays focused on
infrastructure (image, volumes, W&B sweep CLI).
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


class SignalTrainLA2ADatasetSingle(torch.utils.data.Dataset):
    """Dataset that loads a single clip and splits it into segments."""

    def __init__(
        self,
        root_dir: str,
        subset_dir_name: str = "Train",
        subset: str = "train",  # can be train, val, test
        clip_idx: int = 0,
        length: int = 65536,
        sample_rate: int = 44100,
    ):
        self.root_dir = Path(root_dir)
        self.subset_dir_name = subset_dir_name
        self.subset = subset
        self.clip_idx = clip_idx
        self.length = length
        self.sample_rate = sample_rate

        self.subset_dir = self.root_dir / self.subset_dir_name
        if not self.subset_dir.exists():
            raise FileNotFoundError(f"Subset directory not found: {self.subset_dir}")

        input_audio_path = self.subset_dir / f"input_{self.clip_idx}_.wav"

        # Find target file using glob pattern (fixes the original implementation)
        target_pattern = str(self.subset_dir / f"target_{self.clip_idx}_*.wav")
        target_matches = glob.glob(target_pattern)

        if not target_matches:
            raise FileNotFoundError(f"No target file found matching pattern: {target_pattern}")
        if len(target_matches) > 1:
            raise ValueError(f"Multiple target files found: {target_matches}")

        target_audio_path = Path(target_matches[0])

        if not input_audio_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_audio_path}")

        # Load audio files
        input_audio = sf.read(str(input_audio_path), dtype="float32", always_2d=True)[0][:, 0]
        target_audio = sf.read(str(target_audio_path), dtype="float32", always_2d=True)[0][:, 0]

        # Ensure lengths of input and target are the same
        if len(input_audio) != len(target_audio):
            raise ValueError(
                f"Input and target audio lengths do not match: "
                f"{len(input_audio)} != {len(target_audio)}"
            )

        # Split into segments
        input_audio_segments = []
        target_audio_segments = []

        for i in range(0, len(input_audio), self.length):
            if i + self.length > len(input_audio):
                end_sample = len(input_audio)
            else:
                end_sample = i + self.length

            # Pad last segment if needed
            input_seg = input_audio[i:end_sample].copy()
            target_seg = target_audio[i:end_sample].copy()

            if len(input_seg) < self.length:
                # Pad with zeros
                input_seg = np.pad(input_seg, (0, self.length - len(input_seg)), mode="constant")
                target_seg = np.pad(target_seg, (0, self.length - len(target_seg)), mode="constant")

            input_audio_segments.append(torch.from_numpy(input_seg))
            target_audio_segments.append(torch.from_numpy(target_seg))

        self.input_audio_segments = input_audio_segments
        self.target_audio_segments = target_audio_segments
        self.input_audio_path = input_audio_path
        self.target_audio_path = target_audio_path

        # Implement train/val/test split based on the 'subset' argument.
        num_segments = len(self.input_audio_segments)
        test_size = int(np.ceil(0.10 * num_segments))
        val_size = int(np.ceil(0.10 * num_segments))

        if self.subset.lower() == "test":
            start_idx = num_segments - test_size
            end_idx = num_segments
        elif self.subset.lower() == "val":
            start_idx = num_segments - test_size - val_size
            end_idx = num_segments - test_size
        else:  # "train" or fallback
            start_idx = 0
            end_idx = num_segments - test_size - val_size

        self.input_audio_segments = self.input_audio_segments[start_idx:end_idx]
        self.target_audio_segments = self.target_audio_segments[start_idx:end_idx]

    def __len__(self):
        return len(self.input_audio_segments)

    def __getitem__(self, idx):
        return self.input_audio_segments[idx], self.target_audio_segments[idx]


class LSTMmodeler(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, block_size, num_layers=1):
        super().__init__()
        # input size is 1 for mono audio and 2 for stereo audio
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.block_size = block_size
        self.hidden_size = hidden_size
        self.name = "lstm"
        self.num_layers = num_layers

    def process_in_blocks(self, seq: torch.Tensor, hidden_state=None):
        outputs = []
        for i in range(0, seq.shape[1], self.block_size):
            chunk = seq[:, i : i + self.block_size, :]
            out, hidden_state = self.lstm(chunk, hidden_state)
            outputs.append(out)
        return torch.cat(outputs, dim=1), hidden_state

    def forward(self, x):
        B, C, T = x.shape
        assert C == 1, f"Expected mono audio (C=1), got C={C}"

        # (B, 1, T) -> (B, T, 1)
        # x_seq should be of shape (batch_size, sequence_length, input_size)
        x_seq = x.transpose(1, 2).contiguous()

        x, _ = self.process_in_blocks(x_seq, hidden_state=None)
        x = self.fc(x)
        return x.transpose(1, 2).contiguous()


class ResidualLSTMmodeler(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, block_size, num_layers=1):
        super().__init__()
        # input size is 1 for mono audio and 2 for stereo audio
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.num_layers = num_layers
        self.block_size = block_size
        self.hidden_size = hidden_size
        self.name = "residual_lstm"

    def process_in_blocks(self, seq: torch.Tensor, hidden_state=None):
        outputs = []
        for i in range(0, seq.shape[1], self.block_size):
            chunk = seq[:, i : i + self.block_size, :]
            out, hidden_state = self.lstm(chunk, hidden_state)
            outputs.append(out)
        return torch.cat(outputs, dim=1), hidden_state

    def forward(self, x):
        B, C, T = x.shape
        assert C == 1, f"Expected mono audio (C=1), got C={C}"

        # (B, 1, T) -> (B, T, 1)
        # x_seq should be of shape (batch_size, sequence_length, input_size)
        x_seq = x.transpose(1, 2).contiguous()

        x, _ = self.process_in_blocks(x_seq, hidden_state=None)
        delta = self.fc(x)
        output = x_seq + delta
        return output.transpose(1, 2).contiguous()


def create_loss_function(loss_type: str, sample_rate: int = 44100):
    """
    Create a loss function based on the specified loss type.

    Supported loss types:
    - MAE: Mean Absolute Error (L1 loss)
    - MSE: Mean Squared Error (L2 loss)
    - STFT: Short-Time Fourier Transform loss
    - L1+STFT: Combined L1 and STFT loss
    - ESR: Error-to-Signal Ratio loss
    - DC: DC error loss
    - ESR+DC: Combined ESR and DC loss

    Args:
        loss_type: String specifying the loss type (case-insensitive)
        sample_rate: Sample rate for audio (used for STFT-based losses)

    Returns:
        A callable loss function that takes (pred, target) tensors
    """
    import auraloss

    loss_type = loss_type.strip().upper()

    if loss_type == "MAE":
        return torch.nn.L1Loss()

    elif loss_type == "MSE":
        return torch.nn.MSELoss()

    elif loss_type == "STFT":
        return auraloss.freq.STFTLoss(
            fft_size=1024,
            hop_size=256,
            win_length=1024,
            sample_rate=sample_rate,
        )

    elif loss_type == "L1+STFT":
        l1_loss = torch.nn.L1Loss()
        stft_loss = auraloss.freq.STFTLoss(
            fft_size=1024,
            hop_size=256,
            win_length=1024,
            sample_rate=sample_rate,
        )

        def combined_loss(pred, target):
            return l1_loss(pred, target) + stft_loss(pred, target)

        return combined_loss

    elif loss_type == "ESR":
        return auraloss.time.ESRLoss()

    elif loss_type == "DC":
        return auraloss.time.DCLoss()

    elif loss_type == "ESR+DC":
        esr_loss = auraloss.time.ESRLoss()
        dc_loss = auraloss.time.DCLoss()

        def combined_loss(pred, target):
            return esr_loss(pred, target) + dc_loss(pred, target)

        return combined_loss

    else:
        raise ValueError(
            f"Unsupported loss type: {loss_type}. "
            f"Supported types: MAE, MSE, STFT, L1+STFT, ESR, DC, ESR+DC"
        )
