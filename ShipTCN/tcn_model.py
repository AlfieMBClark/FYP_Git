"""
tcn_model.py
------------
Temporal Convolutional Network (TCN) encoder-decoder for ship trajectory
prediction. Fair comparison to ShipTrajectoryTransformer and the GRU baseline —
same data, same loss, same metrics. The only difference is the architecture:
stacked causal, dilated 1-D convolutions with residual connections instead of
recurrence or attention.

  * Encoder — a causal dilated TCN over the (60, 14) input window. Enough dilation
    levels are used to give the last timestep a receptive field covering the whole
    window; that final timestep's features are the context vector.
  * Decoder — a causal dilated TCN over the decoder-input sequence, conditioned on
    the context (concatenated to every step's channels). Causality means step t only
    sees steps <= t, so teacher-forced training decodes all steps in parallel while
    autoregressive inference re-runs the conv over the generated buffer.

The encode / decode_step / forward / predict interface matches the GRU and LSTM
models so the training and evaluation pipelines are shared.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


class Chomp1d(nn.Module):
    """Trim the right-hand padding so a Conv1d is strictly causal."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: Tensor) -> Tensor:
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    """Two causal dilated convolutions + residual (Bai et al. 2018)."""
    def __init__(self, n_in: int, n_out: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(n_in, n_out, kernel, padding=pad, dilation=dilation),
            Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(n_out, n_out, kernel, padding=pad, dilation=dilation),
            Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """Stack of TemporalBlocks with exponentially increasing dilation (1, 2, 4, ...)."""
    def __init__(self, n_in: int, channels: list[int], kernel: int, dropout: float):
        super().__init__()
        layers = []
        for i, ch in enumerate(channels):
            in_ch = n_in if i == 0 else channels[i - 1]
            layers.append(TemporalBlock(in_ch, ch, kernel, dilation=2 ** i, dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)  # (B, channels[-1], T)


def _levels_for_receptive_field(seq_len: int, kernel: int) -> int:
    """Smallest number of dilated levels whose receptive field covers seq_len.

    RF = 1 + 2*(kernel-1)*(2^L - 1)  (two conv layers per level).
    """
    need = (seq_len - 1) / (2 * (kernel - 1)) + 1
    return max(1, math.ceil(math.log2(need)))


class ShipTCNModel(nn.Module):
    def __init__(
        self,
        n_features:   int,   # 14 — encoder input width (per timestep)
        dec_features: int,   # 6  — decoder input width (per step)
        out_features: int,   # 5  — output (mu/log_var) width
        hidden_size:  int,   # 256 — channels per TCN level
        num_layers:   int,   # 2  — decoder TCN levels (encoder grows to cover the window)
        dropout:      float, # 0.2
        seq_len_enc:  int,   # 60 — encoder window length (sizes the receptive field)
        kernel_size:  int = 3,
    ):
        super().__init__()
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.out_features = out_features
        self.seq_len_enc  = seq_len_enc
        self.kernel_size  = kernel_size

        # Encoder: enough levels for the last step to see the whole input window.
        enc_levels = max(num_layers, _levels_for_receptive_field(seq_len_enc, kernel_size))
        self.encoder_tcn = TemporalConvNet(
            n_in=n_features, channels=[hidden_size] * enc_levels,
            kernel=kernel_size, dropout=dropout,
        )

        # Decoder: causal TCN over (context + decoder input) channels.
        self.decoder_tcn = TemporalConvNet(
            n_in=hidden_size + dec_features, channels=[hidden_size] * num_layers,
            kernel=kernel_size, dropout=dropout,
        )

        self.mu_proj      = nn.Linear(hidden_size, out_features)
        self.log_var_proj = nn.Linear(hidden_size, out_features)

    def encode(self, src: Tensor) -> tuple[Tensor, tuple[Tensor, None]]:
        """
        src: (B, 60, 14)
        Returns:
          context:    (B, hidden_size) — features of the final (fully-informed) timestep
          dec_state:  (context, buffer) — decoder state; buffer of generated inputs
                      starts empty (None) and grows during autoregression.
        """
        feats   = self.encoder_tcn(src.transpose(1, 2))  # (B, hidden, 60)
        context = feats[:, :, -1]                         # (B, hidden)
        return context, (context, None)

    def _project(self, dec_out: Tensor) -> tuple[Tensor, Tensor]:
        mu      = self.mu_proj(dec_out)
        log_var = self.log_var_proj(dec_out).clamp(-4.0, 4.0)
        return mu, log_var

    def _decode_seq(self, context: Tensor, dec_seq: Tensor) -> Tensor:
        """
        context: (B, hidden)   dec_seq: (B, T, dec_features)
        Returns per-step decoder features (B, T, hidden), causal in T.
        """
        T   = dec_seq.size(1)
        ctx = context.unsqueeze(1).expand(-1, T, -1)      # (B, T, hidden)
        x   = torch.cat([ctx, dec_seq], dim=-1)           # (B, T, hidden+dec_features)
        y   = self.decoder_tcn(x.transpose(1, 2))         # (B, hidden, T)
        return y.transpose(1, 2)                           # (B, T, hidden)

    def decode_step(
        self,
        dec_input:  Tensor,                  # (B, 1, 6)
        dec_state:  tuple[Tensor, Tensor],   # (context, buffer)
    ) -> tuple[Tensor, Tensor, Tensor, tuple[Tensor, Tensor]]:
        """
        One autoregressive step. Appends dec_input to the buffer, re-runs the causal
        decoder TCN, and returns the last step's outputs.
        Returns: mu (B,1,5), log_var (B,1,5), dec_out (B,1,256), new_state.
        """
        context, buf = dec_state
        buf = dec_input if buf is None else torch.cat([buf, dec_input], dim=1)  # (B, t, 6)
        dec_out = self._decode_seq(context, buf)[:, -1:, :]                      # (B, 1, hidden)
        mu, log_var = self._project(dec_out)
        return mu, log_var, dec_out, (context, buf)

    def forward(
        self,
        src:       Tensor,   # (B, 60, 14)
        tgt_input: Tensor,   # (B, 10, 6) — teacher-forced decoder input
    ) -> tuple[Tensor, Tensor]:
        """Teacher-forced forward pass (all steps in parallel). Returns mu, log_var (B,10,5)."""
        context, _ = self.encode(src)
        dec_out = self._decode_seq(context, tgt_input)    # (B, 10, hidden)
        return self._project(dec_out)

    def predict(
        self,
        src:    Tensor,   # (B, 60, 14)
        tgt_dt: Tensor,   # (B, 10, 1) — ground-truth DT for each future step
        seed:   Tensor,   # (B, 1, 5)  — last encoder step (LAT,LON,SOG,COG_SIN,COG_COS)
    ) -> tuple[Tensor, Tensor]:
        """
        Autoregressive inference — no ground truth motion used, only DT.
        Step i: dec_input = concat(prev_mu, dt_i) → decode_step → mu_i → next input
        """
        _, dec_state = self.encode(src)
        prev_mu = seed
        mu_list, lv_list = [], []

        for t in range(tgt_dt.size(1)):
            dt_t      = tgt_dt[:, t:t+1, :]                    # (B, 1, 1)
            dec_input = torch.cat([prev_mu, dt_t], dim=-1)      # (B, 1, 6)
            mu, log_var, _, dec_state = self.decode_step(dec_input, dec_state)
            mu_list.append(mu)
            lv_list.append(log_var)
            prev_mu = mu.clamp(0.0, 1.0)

        return torch.cat(mu_list, dim=1), torch.cat(lv_list, dim=1)

    def anomaly_score(
        self,
        mu: Tensor, log_var: Tensor, target: Tensor,  # all (B, T, 5)
    ) -> Tensor:
        """Mean abs z-score per timestep: |target-mu| / sigma, mean over features → (B, T)"""
        sigma = (log_var * 0.5).exp()
        return ((target - mu).abs() / (sigma + 1e-8)).mean(dim=-1)


if __name__ == "__main__":
    model = ShipTCNModel(14, 6, 5, 256, 2, 0.2, 60)
    src = torch.randn(4, 60, 14)
    tgt = torch.randn(4, 10, 6)
    mu, lv = model(src, tgt)
    assert mu.shape == (4, 10, 5), f"forward shape wrong: {mu.shape}"
    mu2, lv2 = model.predict(src, tgt[:, :, 5:6], src[:, -1:, :5])
    assert mu2.shape == (4, 10, 5), f"predict shape wrong: {mu2.shape}"
    # Teacher-forced and autoregressive must agree when fed the same inputs (causality check).
    print("tcn_model.py: OK")
