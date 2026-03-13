"""
PyTorch Lightning models for audio hardware emulation.

This module contains LSTM-based models for learning audio transformations
that mimic hardware audio processors (e.g., LA2A compressor).
"""

import math
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from scipy.signal import bilinear_zpk, freqs, zpk2sos, zpk2tf


def pre_emphasis_filter(x: torch.Tensor, coeff: float = 0.95) -> torch.Tensor:
    """
    Pre-emphasis filter: concatenate signal with (x - coeff * x) along the channel dim.

    Equivalent to TF: ``tf.concat([x, x - coeff * x], 1)``.
    Expects input shape (..., C, T); output shape (..., 2*C, T).

    Args:
        x: Input tensor (e.g. audio (B, 1, T)).
        coeff: Pre-emphasis coefficient (default 0.95).

    Returns:
        Concatenated tensor [x, x - coeff * x] along channel dimension.
    """
    return torch.cat([x, x - coeff * x], dim=1)


def _a_weighting_sos_design(fs: float) -> np.ndarray:
    """
    Design digital A-weighting filter (ANSI S1.4-1983 / IEC 61672-1).
    Returns second-order sections (sos) as numpy array of shape (num_sections, 6).
    """
    pi_ = math.pi
    # C-weighting base: 2 zeros at 0, 4 poles (2 at ~20.6 Hz, 2 at ~12.2 kHz)
    z = np.array([0.0, 0.0])
    p = np.array([
        -2 * pi_ * 20.598997057568145,
        -2 * pi_ * 20.598997057568145,
        -2 * pi_ * 12194.21714799801,
        -2 * pi_ * 12194.21714799801,
    ])
    k = 1.0
    # A-weighting: add 2 poles at ~107.7 Hz and ~737.9 Hz, 2 zeros at 0
    z = np.concatenate([z, [0.0, 0.0]])
    p = np.concatenate([p, [-2 * pi_ * 107.65264864304628, -2 * pi_ * 737.8622307362899]])
    # Normalize to 0 dB at 1 kHz (analog)
    b, a = zpk2tf(z, p, k)
    _, h_1k = freqs(b, a, [2 * pi_ * 1000])
    k = k / np.abs(h_1k[0])
    # Bilinear transform: analog → digital
    z_d, p_d, k_d = bilinear_zpk(z, p, k, fs)
    return zpk2sos(z_d, p_d, k_d)


def _sos_filt_torch(x: torch.Tensor, sos: torch.Tensor) -> torch.Tensor:
    """
    Apply cascade of second-order sections along the last dimension.
    x: (..., T), sos: (num_sections, 6) with [b0, b1, b2, a0, a1, a2] per row.
    Returns tensor same shape as x.
    Vectorized over batch; recurrence over time is a single loop with slice ops
    so it can be compiled/fused (e.g. torch.compile).
    """
    *leading, T = x.shape
    x_flat = x.reshape(-1, T)
    device, dtype = x.device, x.dtype
    # Keep coefficients on the same device as x for fast scalar access
    sos = sos.to(device=device, dtype=dtype)
    for i in range(sos.shape[0]):
        b0, b1, b2, a0, a1, a2 = sos[i, 0].item(), sos[i, 1].item(), sos[i, 2].item(), sos[i, 3].item(), sos[i, 4].item(), sos[i, 5].item()
        y = torch.zeros_like(x_flat)
        # Precompute padded inputs for the FIR part (avoids 3 index reads per n)
        x0 = x_flat
        x1 = F.pad(x_flat[:, :-1], (1, 0), value=0)
        x2 = F.pad(x_flat[:, :-2], (2, 0), value=0)
        contrib = (b0 * x0 + b1 * x1 + b2 * x2) / a0
        y[:, 0] = contrib[:, 0]
        if T > 1:
            y[:, 1] = contrib[:, 1] - (a1 / a0) * y[:, 0]
        for n in range(2, T):
            y[:, n] = contrib[:, n] - (a1 / a0) * y[:, n - 1] - (a2 / a0) * y[:, n - 2]
        x_flat = y
    return x_flat.reshape(*leading, T)


# Cache SOS per sample_rate (as torch tensor on CPU; moved to device when used)
_a_weighting_sos_cache: dict[float, torch.Tensor] = {}


def a_weighted_pre_emphasis_filter(x: torch.Tensor, sample_rate: float) -> torch.Tensor:
    """
    A-weighted pre-emphasis filter (ANSI S1.4-1983 / IEC 61672-1).

    Applies A-weighting to the input signal so that the spectrum is weighted
    according to human loudness perception (flat around 1–4 kHz, attenuated
    at low and high frequencies). Useful as a pre_filter in :func:`esr_loss`
    or :func:`dc_loss` to emphasize perceptually relevant errors.

    Expects input shape (..., C, T); output has the same shape.

    Args:
        x: Input tensor (e.g. audio (B, C, T)).
        sample_rate: Sampling frequency in Hz (e.g. 48000).

    Returns:
        A-weighted tensor with the same shape as x.
    """
    global _a_weighting_sos_cache
    if sample_rate not in _a_weighting_sos_cache:
        sos_np = _a_weighting_sos_design(sample_rate)
        _a_weighting_sos_cache[sample_rate] = torch.from_numpy(sos_np).to(dtype=x.dtype)
    sos = _a_weighting_sos_cache[sample_rate].to(device=x.device)
    # Apply along time (last dim); shape is (..., C, T)
    return _sos_filt_torch(x, sos)


def center_crop(x: torch.Tensor, target_size: int) -> torch.Tensor:
    """
    Center-crop along the last dimension to `target_size`.
    Matches the behavior used in micro-TCN (`microtcn.utils.center_crop`).
    """
    current_size = x.shape[-1]
    if current_size == target_size:
        return x
    if current_size < target_size:
        raise ValueError(
            f"Cannot center-crop from length {current_size} to larger length {target_size}."
        )
    start = (current_size - target_size) // 2
    end = start + target_size
    return x[..., start:end]


def causal_crop(x: torch.Tensor, target_size: int) -> torch.Tensor:
    """
    Causal crop along the last dimension to `target_size`, keeping the most recent samples.
    Matches the behavior used in micro-TCN (`microtcn.utils.causal_crop`).
    """
    current_size = x.shape[-1]
    if current_size == target_size:
        return x
    if current_size < target_size:
        raise ValueError(
            f"Cannot causal-crop from length {current_size} to larger length {target_size}."
        )
    return x[..., -target_size:]


def esr_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-4,
    pre_filter: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    r"""
    Energy-based Signal-to-Reconstruction (ESR) loss.

    .. math::
        \mathcal{E}_{ESR} = \frac{\sum_{n=0}^{N-1} |y_p[n] - \hat{y}_p[n]|^2}
                                {\sum_{n=0}^{N-1} |y_p[n]|^2}

    Normalized squared error: error energy divided by signal energy.
    Scale-invariant; a scalar in [0, +inf), with 0 only when pred equals target.

    Args:
        pred: Predicted values :math:`\hat{y}_p` (any shape).
        target: Target values :math:`y_p` (same shape as pred).
        eps: Small constant added to the denominator for numerical stability.
        pre_filter: Optional callable applied to both pred and target before
            computing the loss (e.g. :func:`pre_emphasis_filter` or an A-weighting
            filter). Signature: ``(tensor) -> tensor``.

    Returns:
        Scalar tensor :math:`\mathcal{E}_{ESR}`.
    """
    if pre_filter is not None:
        pred = pre_filter(pred)
        target = pre_filter(target)
    err = pred - target
    numerator = (err.abs().pow(2)).sum()
    denominator = (target.abs().pow(2)).sum()
    return numerator / (denominator + eps)


def dc_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-4,
    pre_filter: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    r"""
    DC loss: squared magnitude of mean error normalized by mean signal power.

    .. math::
        \varepsilon_{DC} = \frac{\left| \frac{1}{N} \sum_{n=0}^{N-1} (y[n] - \hat{y}[n]) \right|^2}
                                {\frac{1}{N} \sum_{n=0}^{N-1} |y[n]|^2}

    Penalizes DC (bias) mismatch; scale-invariant. Returns a scalar in [0, +inf).

    Args:
        pred: Predicted values :math:`\hat{y}` (any shape).
        target: Target values :math:`y` (same shape as pred).
        eps: Small constant added to the denominator for numerical stability.
        pre_filter: Optional callable applied to both pred and target before
            computing the loss (e.g. :func:`pre_emphasis_filter` or an A-weighting
            filter). Signature: ``(tensor) -> tensor``.

    Returns:
        Scalar tensor :math:`\varepsilon_{DC}`.
    """
    if pre_filter is not None:
        pred = pre_filter(pred)
        target = pre_filter(target)
    mean_err = (pred - target).mean()
    numerator = mean_err.abs().pow(2)
    denominator = (target.abs().pow(2)).mean()
    return numerator / (denominator + eps)


class ResidualLSTM(pl.LightningModule):
    """
    Single-step LSTM with residual connection (model from [21] with residual).

    At each time step, the raw waveform sample and (optionally) tiled hardware
    parameters are fed as input channels. The LSTM output is passed through a
    fully connected layer to produce one scalar. A residual connection adds the
    input sample to this scalar (output = input + delta).

    Input:
        - x: (B, 1, T) - batch, mono channel, time samples
        - params: (B, P) - optional hardware parameters (tiled to each time step as extra channels)
    Output: (B, 1, T) - same shape as audio input
    """

    def __init__(
        self,
        n_params: int = 2,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        learning_rate: float = 1e-4,
    ):
        """
        Args:
            n_params: Number of hardware parameters (concatenated as extra input channels).
            hidden_size: Hidden size of the LSTM layer.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout probability between LSTM layers (used when num_layers > 1).
            learning_rate: Learning rate for optimizer.
        """
        super().__init__()
        self.save_hyperparameters()

        self.n_params = n_params
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # LSTM input = 1 audio channel + n_params (params tiled along time)
        lstm_input_size = 1 + n_params

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)
        # Use default init (not zero): zero-init blocks gradient flow to LSTM

    def forward(
        self,
        x: torch.Tensor,
        params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass. Output = input + fc(lstm([input, params_tiled])) at each time step.

        Args:
            x: Input audio (B, 1, T).
            params: Hardware parameters (B, n_params). If None, uses zeros.

        Returns:
            Output audio (B, 1, T).
        """
        B, C, T = x.shape
        assert C == 1, f"Expected mono audio (C=1), got C={C}"

        x_seq = x.transpose(1, 2).contiguous()  # (B, T, 1)

        if params is not None and self.n_params > 0:
            # (B, n_params) -> (B, 1, n_params) -> (B, T, n_params)
            params_tiled = params.unsqueeze(1).expand(-1, T, -1)
            lstm_in = torch.cat([x_seq, params_tiled], dim=-1)
        else:
            params_tiled = torch.zeros(B, T, self.n_params, device=x.device, dtype=x.dtype)
            lstm_in = torch.cat([x_seq, params_tiled], dim=-1)

        lstm_out, _ = self.lstm(lstm_in)        # (B, T, hidden_size)
        delta = self.fc(lstm_out)               # (B, T, 1)
        output = x_seq + delta                  # residual: learn difference
        return output.transpose(1, 2).contiguous()  # (B, 1, T)

    def training_step(self, batch, batch_idx):
        """Training step."""
        input_audio, target_audio, params = batch
        pred_audio = self(input_audio, params)
        loss = F.mse_loss(pred_audio, target_audio)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        input_audio, target_audio, params = batch
        pred_audio = self(input_audio, params)
        loss = F.mse_loss(pred_audio, target_audio)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class ReferenceLSTM(pl.LightningModule):
    """
    LSTM model closely matching the reference micro-TCN LSTM implementation
    (`LSTMModel` in `microtcn.lstm`), adapted to this codebase.

    Shapes:
        Input:
            - x: (B, 1, T)  mono audio
            - params: (B, P) hardware parameters (optional)
        Output:
            - y: (B, 1, T)  transformed audio
    """

    def __init__(
        self,
        n_params: int = 2,
        n_inputs: int = 1,
        n_outputs: int = 1,
        hidden_size: int = 32,
        num_layers: int = 1,
        learning_rate: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.n_params = n_params
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate

        input_size = n_inputs + n_params

        # Match reference: batch_first=False, (seq, batch, feature)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
            bidirectional=False,
        )

        self.linear = nn.Linear(hidden_size, n_outputs)

    def forward(
        self,
        x: torch.Tensor,
        params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, 1, T) input audio
            params: (B, P) hardware parameters (optional)

        Returns:
            (B, 1, T) output audio
        """
        B, C, T = x.shape
        assert C == self.n_inputs, f"Expected {self.n_inputs} input channel(s), got {C}"

        # (B, C, T) -> (T, B, C) for LSTM (seq, batch, feature)
        x_seq = x.permute(2, 0, 1).contiguous()

        if params is not None and self.n_params > 0:
            # (B, P) -> (B, 1, P) -> (1, B, P) -> (T, B, P)
            p = params.unsqueeze(1).permute(1, 0, 2)
            p = p.repeat(T, 1, 1)
            lstm_in = torch.cat((x_seq, p), dim=-1)
        else:
            zeros_params = torch.zeros(
                T,
                B,
                self.n_params,
                device=x.device,
                dtype=x.dtype,
            )
            lstm_in = torch.cat((x_seq, zeros_params), dim=-1)

        out, _ = self.lstm(lstm_in)  # (T, B, hidden_size)
        out = torch.tanh(self.linear(out))  # (T, B, n_outputs)
        out = out.permute(1, 2, 0).contiguous()  # (B, n_outputs, T)
        return out

    def training_step(self, batch, batch_idx):
        input_audio, target_audio, params = batch
        pred_audio = self(input_audio, params)
        loss = F.mse_loss(pred_audio, target_audio)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_audio, target_audio, params = batch
        pred_audio = self(input_audio, params)
        loss = F.mse_loss(pred_audio, target_audio)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class CausalBlockLSTM(pl.LightningModule):
    """
    Causal LSTM that processes long sequences in blocks, persisting hidden state
    across blocks to handle arbitrarily long signals without truncating context.

    Uses PyTorch's native LSTM. Block size is configurable (default 22050 samples).
    LSTM states are carried from one block to the next, so the model effectively
    processes the whole signal in chunks.

    Input:
        - x: (B, 1, T) mono audio
        - params: (B, P) hardware parameters (optional, tiled to each time step)
    Output: (B, 1, T) transformed audio
    """

    def __init__(
        self,
        n_params: int = 2,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        block_size: int = 22050,
        learning_rate: float = 1e-4,
        causal: bool = True,
    ):
        """
        Args:
            n_params: Number of hardware parameters (concatenated as extra input channels).
            hidden_size: Hidden size of the LSTM layer.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout probability between LSTM layers (used when num_layers > 1).
            block_size: Max sequence length per block; sequences longer than this are
                processed in chunks with hidden state carried across blocks.
            learning_rate: Learning rate for optimizer.
            causal: Whether the model is causal (used for eval cropping). Default True.
        """
        super().__init__()
        self.save_hyperparameters()

        self.n_params = n_params
        self.hidden_size = hidden_size
        self.block_size = block_size
        self.learning_rate = learning_rate

        lstm_input_size = 1 + n_params

        self._core = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self._head = nn.Linear(hidden_size, 1)
        # Use default init (not zero): zero-init blocks gradient flow to LSTM

    def _initial_state(self, batch_size: int):
        """Return zero hidden/cell state for the LSTM."""
        device = next(self._core.parameters()).device
        dtype = next(self._core.parameters()).dtype
        h = torch.zeros(
            self._core.num_layers,
            batch_size,
            self.hidden_size,
            device=device,
            dtype=dtype,
        )
        c = torch.zeros(
            self._core.num_layers,
            batch_size,
            self.hidden_size,
            device=device,
            dtype=dtype,
        )
        return (h, c)

    def _apply_head(self, x: torch.Tensor) -> torch.Tensor:
        """Apply output projection: (B, L, H) -> (B, L, 1)."""
        return self._head(x)

    def _forward(
        self,
        x: torch.Tensor,
        params: Optional[torch.Tensor] = None,
        initial_state: Optional[tuple] = None,
    ) -> torch.Tensor:
        """
        Forward pass with block-wise processing.

        Args:
            x: (B, 1, T) input audio.
            params: (B, n_params) hardware parameters. If None, uses zeros.
            initial_state: Optional (h, c) tuple for LSTM. If None, uses zeros.

        Returns:
            (B, 1, T) output audio.
        """
        B, C, T = x.shape
        assert C == 1, f"Expected mono audio (C=1), got C={C}"

        # (B, 1, T) -> (B, T, 1)
        x_seq = x.transpose(1, 2).contiguous()

        if params is not None and self.n_params > 0:
            params_tiled = params.unsqueeze(1).expand(-1, T, -1)
            lstm_in = torch.cat([x_seq, params_tiled], dim=-1)
        else:
            params_tiled = torch.zeros(
                B, T, self.n_params, device=x.device, dtype=x.dtype
            )
            lstm_in = torch.cat([x_seq, params_tiled], dim=-1)

        block_size = self.block_size
        last_hidden_state = (
            self._initial_state(B) if initial_state is None else initial_state
        )

        def process_in_blocks(seq: torch.Tensor, hidden_state=None):
            outputs = []
            for i in range(0, seq.shape[1], block_size):
                chunk = seq[:, i : i + block_size, :]
                out, hidden_state = self._core(chunk, hidden_state)
                outputs.append(out)
            return torch.cat(outputs, dim=1), hidden_state

        output_features, _ = process_in_blocks(lstm_in, last_hidden_state)
        delta = self._apply_head(output_features)
        output = x_seq + delta
        return output.transpose(1, 2).contiguous()

    def forward(
        self,
        x: torch.Tensor,
        params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass. Output = input + head(lstm([input, params_tiled])).

        Args:
            x: Input audio (B, 1, T).
            params: Hardware parameters (B, n_params). If None, uses zeros.

        Returns:
            Output audio (B, 1, T).
        """
        return self._forward(x, params=params)

    def training_step(self, batch, batch_idx):
        """Training step."""
        input_audio, target_audio, params = batch
        pred_audio = self(input_audio, params)
        loss = F.mse_loss(pred_audio, target_audio)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        input_audio, target_audio, params = batch
        pred_audio = self(input_audio, params)
        loss = F.mse_loss(pred_audio, target_audio)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class TestLSTM(pl.LightningModule):
    """
    Simple LSTM model for audio processing with PyTorch Lightning.
    
    This model uses a single LSTM layer to process audio vectors of any size.
    Input: (B, 1, T) - batch, channels (mono), time samples
    Output: (B, 1, T) - same shape as input
    """
    
    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        learning_rate: float = 1e-4,
    ):
        """
        Args:
            hidden_size: Hidden size of the LSTM layer
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout probability between LSTM layers (used when num_layers > 1).
            learning_rate: Learning rate for optimizer
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        
        # Input size is 1 (mono audio), output is hidden_size
        self.lstm = torch.nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        # Output projection: hidden_size -> 1 (back to mono audio)
        self.output_proj = torch.nn.Linear(hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input audio tensor of shape (B, 1, T)
        
        Returns:
            Output audio tensor of shape (B, 1, T)
        """
        # x shape: (B, 1, T)
        B, C, T = x.shape
        assert C == 1, f"Expected mono audio (C=1), got C={C}"
        
        # Reshape to (B, T, 1) for LSTM (batch_first=True)
        x_seq = x.transpose(1, 2).contiguous()  # (B, T, 1)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x_seq)  # (B, T, hidden_size)
        
        # Project back to audio dimension
        output = self.output_proj(lstm_out)  # (B, T, 1)
        
        # Reshape back to (B, 1, T)
        output = output.transpose(1, 2).contiguous()  # (B, 1, T)
        
        return output
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        input_audio, target_audio, params = batch
        
        # Forward pass
        pred_audio = self(input_audio)
        
        # Compute loss (MSE)
        loss = torch.nn.functional.mse_loss(pred_audio, target_audio)
        
        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        input_audio, target_audio, params = batch
        
        # Forward pass
        pred_audio = self(input_audio)
        
        # Compute loss (MSE)
        loss = torch.nn.functional.mse_loss(pred_audio, target_audio)
        
        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class SimpleHardwareEmulationLSTM(pl.LightningModule):
    """
    Simple LSTM for audio hardware emulation with no FiLM layers.
    
    Hardware parameters are concatenated with audio as plain input features at each
    time step (no embedding). Input size = n_audio_channels + n_params.
    
    Input:
        - input_audio: (B, C, T) - batch, audio channels, time samples
        - params: (B, P) - batch, hardware parameter vector
    Output:
        - output_audio: (B, C, T) - transformed audio
    """

    def __init__(
        self,
        n_audio_channels: int = 1,
        n_params: int = 2,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,    
    ):
        """
        Args:
            n_audio_channels: Number of audio input channels (e.g. 1 for mono).
            n_params: Number of hardware parameters (e.g., gain, ratio).
            hidden_size: Hidden size of LSTM layers.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout probability between LSTM layers.
            learning_rate: Learning rate for optimizer.
        """
        super().__init__()
        self.save_hyperparameters()

        self.n_audio_channels = n_audio_channels
        self.n_params = n_params
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate

        # LSTM input = audio channels + params (no embedding)
        lstm_input_size = n_audio_channels + n_params

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_proj = nn.Linear(hidden_size, n_audio_channels)

    def forward(
        self,
        x: torch.Tensor,
        params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input audio (B, C, T).
            params: Hardware parameters (B, n_params). If None, uses zeros.

        Returns:
            Output audio (B, C, T).
        """
        B, C, T = x.shape
        assert C == self.n_audio_channels, (
            f"Expected {self.n_audio_channels} audio channels, got {C}"
        )

        # (B, T, C)
        x_seq = x.transpose(1, 2).contiguous()

        if params is not None and self.n_params > 0:
            # (B, n_params) -> (B, 1, n_params) -> (B, T, n_params)
            params_tiled = params.unsqueeze(1).expand(-1, T, -1)
            lstm_in = torch.cat([x_seq, params_tiled], dim=-1)
        else:
            params_tiled = torch.zeros(
                B, T, self.n_params, device=x.device, dtype=x.dtype
            )
            lstm_in = torch.cat([x_seq, params_tiled], dim=-1)

        lstm_out, _ = self.lstm(lstm_in)
        output = self.output_proj(lstm_out)

        return output.transpose(1, 2).contiguous()

    def training_step(self, batch, batch_idx):
        """Training step."""
        input_audio, target_audio, params = batch
        pred_audio = self(input_audio, params)
        loss = F.mse_loss(pred_audio, target_audio)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        input_audio, target_audio, params = batch
        pred_audio = self(input_audio, params)
        loss = F.mse_loss(pred_audio, target_audio)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class HardwareEmulationLSTM(pl.LightningModule):
    """
    Advanced LSTM model for audio hardware emulation.
    
    This model learns to transform input audio to match the output of hardware
    audio processors (e.g., LA2A compressor) based on hardware parameter settings.
    
    Architecture:
    - Parameter embedding network to learn rich representations of hardware settings
    - Bidirectional LSTM layers for temporal context (past and future)
    - Feature-wise Linear Modulation (FiLM) to condition LSTM on hardware params
    - Residual connections for gradient flow and identity mapping
    - Skip connection from input to output for learning residual transformations
    
    Input: 
        - input_audio: (B, 1, T) - batch, mono channel, time samples
        - params: (B, P) - batch, hardware parameter vector
    Output: 
        - output_audio: (B, 1, T) - transformed audio
    """
    
    def __init__(
        self,
        n_params: int = 2,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        param_embed_dim: int = 64,
        use_bidirectional: bool = True,
        use_residual: bool = True,
        use_skip_connection: bool = True,
        learning_rate: float = 1e-4,
    ):
        """
        Args:
            n_params: Number of hardware parameters (e.g., gain, ratio for LA2A)
            hidden_size: Hidden size of LSTM layers
            num_layers: Number of LSTM layers
            dropout: Dropout probability (applied between LSTM layers)
            param_embed_dim: Dimension of parameter embedding
            use_bidirectional: Whether to use bidirectional LSTM
            use_residual: Whether to use residual connections between LSTM layers
            use_skip_connection: Whether to add skip connection from input to output
            learning_rate: Learning rate for optimizer
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.n_params = n_params
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.param_embed_dim = param_embed_dim
        self.use_bidirectional = use_bidirectional
        self.use_residual = use_residual
        self.use_skip_connection = use_skip_connection
        self.learning_rate = learning_rate
        
        # Parameter embedding network
        # Maps hardware parameters to a learned representation
        self.param_embedding = nn.Sequential(
            nn.Linear(n_params, param_embed_dim),
            nn.ReLU(),
            nn.Linear(param_embed_dim, param_embed_dim),
            nn.ReLU(),
            nn.Linear(param_embed_dim, param_embed_dim),
        )
        
        # FiLM (Feature-wise Linear Modulation) generators for conditioning
        # Each LSTM layer gets its own FiLM parameters
        self.film_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(param_embed_dim, hidden_size * 2),  # scale and shift
            ) for _ in range(num_layers)
        ])
        
        # Bidirectional LSTM layers
        lstm_input_size = 1  # mono audio
        lstm_hidden_size = hidden_size
        lstm_num_directions = 2 if use_bidirectional else 1
        
        self.lstm_layers = nn.ModuleList()
        self.residual_projections = nn.ModuleList()  # For residual connections
        
        for i in range(num_layers):
            lstm = nn.LSTM(
                input_size=lstm_input_size if i == 0 else lstm_hidden_size * lstm_num_directions,
                hidden_size=lstm_hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=use_bidirectional,
                dropout=0.0,  # We'll apply dropout manually
            )
            self.lstm_layers.append(lstm)
            
            # Residual projection layers (if needed for dimension matching)
            if i > 0 and use_residual:
                # After layer 0, all layers output hidden_size * num_directions
                # So prev_size is always hidden_size * num_directions for i > 0
                prev_size = lstm_hidden_size * lstm_num_directions
                curr_size = lstm_hidden_size * lstm_num_directions
                # Dimensions should always match after first layer, but keep projection for flexibility
                if prev_size != curr_size:
                    self.residual_projections.append(nn.Linear(prev_size, curr_size))
                else:
                    self.residual_projections.append(nn.Identity())
            else:
                self.residual_projections.append(nn.Identity())
        
        # Output projection
        # Maps from LSTM hidden state to audio output
        final_hidden_size = lstm_hidden_size * lstm_num_directions
        self.output_proj = nn.Sequential(
            nn.Linear(final_hidden_size, final_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_hidden_size // 2, 1),
        )
        
    def apply_film(self, x: torch.Tensor, film_params: torch.Tensor) -> torch.Tensor:
        """
        Apply Feature-wise Linear Modulation (FiLM) to condition features on parameters.
        
        Args:
            x: Features to modulate, shape (B, T, H)
            film_params: FiLM parameters [scale, shift], shape (B, 2*H)
        
        Returns:
            Modulated features, shape (B, T, H)
        """
        B, T, H = x.shape
        scale = film_params[:, :H].unsqueeze(1)  # (B, 1, H)
        shift = film_params[:, H:].unsqueeze(1)  # (B, 1, H)
        
        # Apply FiLM: output = scale * x + shift
        return scale * x + shift
    
    def forward(
        self, 
        x: torch.Tensor, 
        params: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input audio tensor of shape (B, 1, T)
            params: Hardware parameter tensor of shape (B, n_params), optional
        
        Returns:
            Output audio tensor of shape (B, 1, T)
        """
        B, C, T = x.shape
        assert C == 1, f"Expected mono audio (C=1), got C={C}"
        
        # Reshape to (B, T, 1) for LSTM
        x_seq = x.transpose(1, 2).contiguous()  # (B, T, 1)
        
        # Embed hardware parameters
        if params is not None and self.n_params > 0:
            param_embed = self.param_embedding(params)  # (B, param_embed_dim)
            
            # Generate FiLM parameters for each LSTM layer
            film_params_list = [
                film_gen(param_embed) for film_gen in self.film_generators
            ]  # List of (B, 2*hidden_size)
        else:
            # If no params, use zero modulation (identity)
            film_params_list = [
                torch.zeros(B, 2 * self.hidden_size, device=x.device)
                for _ in range(self.num_layers)
            ]
        
        # Process through LSTM layers with FiLM conditioning
        current_seq = x_seq  # (B, T, 1)
        prev_seq = None  # Store previous layer output for residual
        
        for i, (lstm, film_params) in enumerate(zip(self.lstm_layers, film_params_list)):
            # LSTM forward pass
            lstm_out, _ = lstm(current_seq)  # (B, T, H) or (B, T, 2*H) if bidirectional
            
            # Apply FiLM conditioning if using bidirectional, split and condition separately
            if self.use_bidirectional:
                H = self.hidden_size
                forward_out = lstm_out[:, :, :H]  # (B, T, H)
                backward_out = lstm_out[:, :, H:]  # (B, T, H)
                
                # Apply FiLM to both directions
                forward_out = self.apply_film(forward_out, film_params)
                backward_out = self.apply_film(backward_out, film_params)
                
                # Concatenate back
                lstm_out = torch.cat([forward_out, backward_out], dim=-1)
            else:
                lstm_out = self.apply_film(lstm_out, film_params)
            
            # Residual connection (if enabled)
            if self.use_residual and i > 0 and prev_seq is not None:
                # Use pre-defined projection layer for dimension matching
                # prev_seq should be (B, T, prev_size), lstm_out should be (B, T, curr_size)
                residual = self.residual_projections[i](prev_seq)
                # Ensure shapes match for addition
                assert residual.shape == lstm_out.shape, (
                    f"Residual shape mismatch at layer {i}: "
                    f"residual {residual.shape} vs lstm_out {lstm_out.shape}"
                )
                lstm_out = lstm_out + residual
            
            # Apply dropout
            if i < self.num_layers - 1:  # No dropout after last layer
                lstm_out = F.dropout(lstm_out, p=self.dropout, training=self.training)
            
            # Update for next iteration
            prev_seq = lstm_out
            current_seq = lstm_out
        
        # Output projection
        output = self.output_proj(current_seq)  # (B, T, 1)
        
        # Skip connection from input to output (learn residual transformation)
        if self.use_skip_connection:
            output = output + x_seq
        
        # Reshape back to (B, 1, T)
        output = output.transpose(1, 2).contiguous()  # (B, 1, T)
        
        return output
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        input_audio, target_audio, params = batch
        
        # Forward pass
        pred_audio = self(input_audio, params)
        
        # Compute loss (MSE)
        loss = F.mse_loss(pred_audio, target_audio)
        
        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        input_audio, target_audio, params = batch
        
        # Forward pass
        pred_audio = self(input_audio, params)
        
        # Compute loss (MSE)
        loss = F.mse_loss(pred_audio, target_audio)
        
        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class FiLM(torch.nn.Module):
    """
    Feature-wise Linear Modulation layer from micro-TCN.
    Applies conditional affine transformation after BatchNorm.
    """

    def __init__(self, num_features: int, cond_dim: int):
        super().__init__()
        self.num_features = num_features
        # BatchNorm without affine; FiLM provides the affine parameters.
        self.bn = torch.nn.BatchNorm1d(num_features, affine=False)
        self.adaptor = torch.nn.Linear(cond_dim, num_features * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) feature map.
            cond: (B, 1, cond_dim) conditioning features.
        """
        cond = self.adaptor(cond)
        g, b = torch.chunk(cond, 2, dim=-1)
        g = g.permute(0, 2, 1)
        b = b.permute(0, 2, 1)

        x = self.bn(x)
        x = (x * g) + b
        return x


class TCNBlock(torch.nn.Module):
    """
    Single temporal convolutional block from micro-TCN with FiLM conditioning.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        padding: str = "same",
        dilation: int = 1,
        grouped: bool = False,
        causal: bool = False,
        conditional: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.grouped = grouped
        self.causal = causal
        self.conditional = conditional

        groups = out_ch if grouped and (in_ch % out_ch == 0) else 1

        if padding == "same":
            pad_value = (kernel_size - 1) + ((kernel_size - 1) * (dilation - 1))
        elif padding in ["none", "valid"]:
            pad_value = 0
        else:
            raise ValueError(f"Unsupported padding mode: {padding!r}")

        self.pad_value = pad_value

        # NOTE: micro-TCN uses padding=0 on the conv and applies manual padding in forward.
        self.conv1 = torch.nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=False,
        )

        if grouped:
            self.conv1b = torch.nn.Conv1d(out_ch, out_ch, kernel_size=1)
        else:
            self.conv1b = None

        if conditional:
            self.film = FiLM(out_ch, 32)
        else:
            self.bn = torch.nn.BatchNorm1d(out_ch)

        self.relu = torch.nn.PReLU(out_ch)
        self.res = torch.nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=1,
            groups=in_ch,
            bias=False,
        )

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        x_in = x

        if self.pad_value > 0:
            if self.causal:
                x = F.pad(x, (self.pad_value, 0), mode="constant", value=0)
            else:
                pad_left = self.pad_value // 2
                pad_right = self.pad_value - pad_left
                x = F.pad(x, (pad_left, pad_right), mode="constant", value=0)
        x = self.conv1(x)
        if self.grouped and self.conv1b is not None:
            x = self.conv1b(x)

        # micro-TCN applies FiLM conditioning unconditionally when conditional=True.
        x = self.film(x, p)
        x = self.relu(x)

        x_res = self.res(x_in)
        if self.causal:
            x = x + causal_crop(x_res, x.shape[-1])
        else:
            x = x + center_crop(x_res, x.shape[-1])

        return x


class TCNModel(pl.LightningModule):
    """
    Temporal Convolutional Network with global conditioning, adapted from micro-TCN.

    The core architecture (FiLM, block structure, convolutions, and output head)
    matches `microtcn.tcn.TCNModel` exactly, with a LightningModule wrapper and a
    `forward(x, params)` signature compatible with this codebase.

    Shapes:
        x: (B, ninputs, T) audio input (usually mono, ninputs=1)
        params: (B, nparams) hardware parameters (tiled and embedded internally)
        output: (B, noutputs, T)
    """

    def __init__(
        self,
        nparams: int,
        ninputs: int = 1,
        noutputs: int = 1,
        nblocks: int = 10,
        kernel_size: int = 3,
        dilation_growth: int = 1,
        channel_growth: int = 1,
        channel_width: int = 32,
        stack_size: int = 10,
        grouped: bool = False,
        causal: bool = False,
        skip_connections: bool = False,
        num_examples: int = 4,
        learning_rate: float = 1e-4,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Global conditioning network (from micro-TCN)
        if self.hparams.nparams > 0:
            self.gen = torch.nn.Sequential(
                torch.nn.Linear(nparams, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 32),
                torch.nn.ReLU(),
            )
        else:
            self.gen = None

        self.blocks = torch.nn.ModuleList()
        out_ch = None

        for n in range(nblocks):
            in_ch = out_ch if n > 0 else ninputs

            if self.hparams.channel_growth > 1:
                out_ch = in_ch * self.hparams.channel_growth
            else:
                out_ch = self.hparams.channel_width

            dilation = self.hparams.dilation_growth ** (n % self.hparams.stack_size)
            self.blocks.append(
                TCNBlock(
                    in_ch,
                    out_ch,
                    kernel_size=self.hparams.kernel_size,
                    dilation=dilation,
                    padding="same" if self.hparams.causal else "valid",
                    causal=self.hparams.causal,
                    grouped=self.hparams.grouped,
                    conditional=True if self.hparams.nparams > 0 else False,
                )
            )

        self.output = torch.nn.Conv1d(out_ch, noutputs, kernel_size=1)
        self.learning_rate = learning_rate

    def forward(
        self,
        x: torch.Tensor,
        params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, ninputs, T) input audio.
            params: (B, nparams) hardware parameters. If None, zeros are used.
        """
        B, C, T = x.shape
        assert C == self.hparams.ninputs, (
            f"Expected {self.hparams.ninputs} input channels, got {C}"
        )

        if self.hparams.nparams > 0:
            if params is None:
                # Zero conditioning when params are not provided.
                p = torch.zeros(
                    B,
                    1,
                    self.hparams.nparams,
                    device=x.device,
                    dtype=x.dtype,
                )
            else:
                assert params.shape[1] == self.hparams.nparams, (
                    f"Expected params of shape (B, {self.hparams.nparams}), "
                    f"got {tuple(params.shape)}"
                )
                # micro-TCN uses a single time step for conditioning (broadcast across T).
                p = params.unsqueeze(1)  # (B, 1, nparams)

            cond = self.gen(p)
        else:
            cond = None

        skips = 0
        for idx, block in enumerate(self.blocks):
            # cond is shared across all blocks, as in micro-TCN.
            x = block(x, cond)
            if self.hparams.skip_connections:
                if idx == 0:
                    skips = x
                else:
                    skips = center_crop(skips, x.shape[-1]) + x

        if self.hparams.skip_connections:
            x = x + skips

        out = torch.tanh(self.output(x))
        return out

    def compute_receptive_field(self) -> int:
        """
        Compute the receptive field in samples, matching micro-TCN's implementation.
        """
        rf = self.hparams.kernel_size
        for n in range(1, self.hparams.nblocks):
            dilation = self.hparams.dilation_growth ** (n % self.hparams.stack_size)
            rf = rf + ((self.hparams.kernel_size - 1) * dilation)
        return int(rf)

    def training_step(self, batch, batch_idx):
        """
        Basic MSE training step.
        Note: In this project, training is typically done via LossConfigWrapper,
        which overrides the training_step and validation_step with configurable
        audio losses, but this keeps the model self-contained for standalone use.
        """
        input_audio, target_audio, params = batch
        pred_audio = self(input_audio, params)
        loss = F.mse_loss(pred_audio, target_audio)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_audio, target_audio, params = batch
        pred_audio = self(input_audio, params)
        loss = F.mse_loss(pred_audio, target_audio)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
