"""
model.py
--------
Ship Trajectory Transformer — probabilistic encoder-decoder.

The output head predicts a Gaussian distribution over the next position
rather than a single point.  This serves two purposes:

  1. Training loss is Gaussian NLL instead of MSE, which is better calibrated
     and naturally handles the fact that some futures are more uncertain.

  2. Anomaly detection for free at inference time: the z-score of the actual
     observed position against the predicted distribution is the anomaly score.
     A vessel that has been spoofed (position jump) will be many σ away from
     the model's prediction.  No labelled anomaly data is required.

Architecture changes vs. the point-prediction version
------------------------------------------------------
  output_proj (d_model → n_features)
  →  mu_proj      (d_model → n_features)   predicted mean
     log_var_proj (d_model → n_features)   predicted log-variance

Everything else — multi-head attention, positional encoding, encoder/decoder
stacks — is identical.
"""

import math
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        return torch.matmul(torch.softmax(scores, dim=-1), V)

    def split_heads(self, x):
        B, S, _ = x.size()
        return x.view(B, S, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        B, _, S, _ = x.size()
        return x.transpose(1, 2).contiguous().view(B, S, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        return self.W_o(self.combine_heads(self.scaled_dot_product_attention(Q, K, V, mask)))


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1  = nn.Linear(d_model, d_ff)
        self.fc2  = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int):
        super().__init__()
        pe       = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn    = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1        = nn.LayerNorm(d_model)
        self.norm2        = nn.LayerNorm(d_model)
        self.dropout      = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask)))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn    = MultiHeadAttention(d_model, num_heads)
        self.cross_attn   = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1        = nn.LayerNorm(d_model)
        self.norm2        = nn.LayerNorm(d_model)
        self.norm3        = nn.LayerNorm(d_model)
        self.dropout      = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_output, enc_output, src_mask)))
        x = self.norm3(x + self.dropout(self.feed_forward(x)))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────

class ShipTrajectoryTransformer(nn.Module):
    """
    Probabilistic encoder-decoder transformer for ship trajectory modelling.

    forward() returns (mu, log_var) instead of a single point prediction.
    Use anomaly_score() at inference to flag unusual behaviour.
    """

    def __init__(
        self,
        n_features:     int,
        d_model:        int,
        num_heads:      int,
        num_layers:     int,
        d_ff:           int,
        max_seq_length: int,
        dropout:        float,
    ):
        super().__init__()

        self.encoder_input_proj = nn.Linear(n_features, d_model)
        self.decoder_input_proj = nn.Linear(n_features, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        # Probabilistic output: separate heads for mean and log-variance.
        # log_var is clamped in forward() to keep variance in a stable range.
        self.mu_proj      = nn.Linear(d_model, n_features)
        self.log_var_proj = nn.Linear(d_model, n_features)

        self.dropout = nn.Dropout(dropout)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)

    def _encode(self, src: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.positional_encoding(self.encoder_input_proj(src)))
        for layer in self.encoder_layers:
            x = layer(x, mask=None)
        return x

    def _decode(self, tgt: torch.Tensor, enc_output: torch.Tensor) -> torch.Tensor:
        tgt_mask = self._causal_mask(tgt.size(1), tgt.device)
        x = self.dropout(self.positional_encoding(self.decoder_input_proj(tgt)))
        for layer in self.decoder_layers:
            x = layer(x, enc_output, src_mask=None, tgt_mask=tgt_mask)
        return x

    def forward(
        self, src: torch.Tensor, tgt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        src : (batch, src_len, n_features)
        tgt : (batch, tgt_len, n_features)

        Returns
        -------
        mu      : (batch, tgt_len, n_features)  predicted mean
        log_var : (batch, tgt_len, n_features)  predicted log-variance
                  clamped to [-8, 4] → variance in [0.0003, 54.6]
        """
        enc_output = self._encode(src)
        dec_output = self._decode(tgt, enc_output)

        mu      = self.mu_proj(dec_output)
        log_var = self.log_var_proj(dec_output).clamp(-8.0, 4.0)
        return mu, log_var

    # ── Anomaly detection ─────────────────────────────────────────────────────

    def anomaly_score(
        self,
        mu:      torch.Tensor,
        log_var: torch.Tensor,
        target:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Per-timestep anomaly score: mean absolute z-score across features.

        High score → the observed position is far outside the predicted
        distribution → likely anomalous (spoofing, position jump, error).

        Parameters
        ----------
        mu, log_var : output of forward()   (batch, tgt_len, n_features)
        target      : actual observed data  (batch, tgt_len, n_features)

        Returns
        -------
        (batch, tgt_len) — score per predicted timestep; average over tgt_len
        for a single-number track-level score.
        """
        std = (log_var * 0.5).exp()
        return ((target - mu).abs() / (std + 1e-8)).mean(dim=-1)
