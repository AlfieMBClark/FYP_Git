"""
baseline_model.py
-----------------
GRU encoder-decoder baseline for ship trajectory prediction.
Fair comparison to ShipTrajectoryTransformer — same data, same loss, same metrics.
"""

import torch
import torch.nn as nn
from torch import Tensor


class ShipGRUBaseline(nn.Module):
    def __init__(
        self,
        n_features:   int,   # 14 — encoder input width
        dec_features: int,   # 6  — decoder input width
        out_features: int,   # 5  — output (mu/log_var) width
        hidden_size:  int,   # 256
        num_layers:   int,   # 2
        dropout:      float, # 0.2
    ):
        super().__init__()
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.out_features = out_features

        self.encoder_gru = nn.GRU(
            input_size    = n_features,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if num_layers > 1 else 0.0,
        )

        # One bridge per layer: Linear(512→256)+Tanh initialises decoder hidden state
        self.encoder_to_decoder = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.Tanh())
            for _ in range(num_layers)
        ])

        self.decoder_gru = nn.GRU(
            input_size  = dec_features,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        self.mu_proj      = nn.Linear(hidden_size, out_features)
        self.log_var_proj = nn.Linear(hidden_size, out_features)

    def encode(self, src: Tensor) -> tuple[Tensor, Tensor]:
        """
        src: (B, 90, 14)
        Returns:
          context:    (B, hidden_size*2) — final encoder hidden, both directions concat'd
          dec_hidden: (num_layers, B, hidden_size) — initialised decoder hidden state
        """
        _, h_n = self.encoder_gru(src)  # h_n: (num_layers*2, B, hidden_size)
        B = src.size(0)

        # (num_layers*2, B, H) → (num_layers, 2, B, H) → concat directions → (num_layers, B, H*2)
        h     = h_n.view(self.num_layers, 2, B, self.hidden_size)
        h_cat = torch.cat([h[:, 0, :, :], h[:, 1, :, :]], dim=-1)  # (num_layers, B, H*2)

        context = h_cat[-1]  # (B, H*2) — last layer, both directions

        dec_hidden = torch.stack([
            self.encoder_to_decoder[i](h_cat[i])
            for i in range(self.num_layers)
        ], dim=0)  # (num_layers, B, H)

        return context, dec_hidden

    def decode_step(
        self,
        dec_input:  Tensor,   # (B, 1, 6)
        dec_hidden: Tensor,   # (num_layers, B, hidden_size)
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Returns: mu (B,1,5), log_var (B,1,5), dec_out (B,1,256), new_hidden"""
        dec_out, new_hidden = self.decoder_gru(dec_input, dec_hidden)
        mu      = self.mu_proj(dec_out)
        log_var = self.log_var_proj(dec_out).clamp(-4.0, 4.0)
        return mu, log_var, dec_out, new_hidden

    def forward(
        self,
        src:       Tensor,   # (B, 90, 14)
        tgt_input: Tensor,   # (B, 10, 6) — teacher-forced decoder input
    ) -> tuple[Tensor, Tensor]:
        """Teacher-forced forward pass. Returns: mu (B,10,5), log_var (B,10,5)"""
        _, dec_hidden = self.encode(src)
        dec_out, _ = self.decoder_gru(tgt_input, dec_hidden)  # (B, 10, 256)
        mu      = self.mu_proj(dec_out)
        log_var = self.log_var_proj(dec_out).clamp(-4.0, 4.0)
        return mu, log_var

    def predict(
        self,
        src:    Tensor,   # (B, 90, 14)
        tgt_dt: Tensor,   # (B, 10, 1) — ground-truth DT for each future step
        seed:   Tensor,   # (B, 1, 5)  — last encoder step (LAT,LON,SOG,COG_SIN,COG_COS)
    ) -> tuple[Tensor, Tensor]:
        """
        Autoregressive inference — no ground truth motion used, only DT.
        Step i: dec_input = concat(prev_mu, dt_i) → decode_step → mu_i → next input
        """
        _, dec_hidden = self.encode(src)
        prev_mu = seed
        mu_list, lv_list = [], []

        for t in range(tgt_dt.size(1)):
            dt_t      = tgt_dt[:, t:t+1, :]                    # (B, 1, 1)
            dec_input = torch.cat([prev_mu, dt_t], dim=-1)      # (B, 1, 6)
            mu, log_var, _, dec_hidden = self.decode_step(dec_input, dec_hidden)
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
    model = ShipGRUBaseline(14, 6, 5, 256, 2, 0.2)
    src = torch.randn(4, 90, 14)
    tgt = torch.randn(4, 10, 6)
    mu, lv = model(src, tgt)
    assert mu.shape == (4, 10, 5), f"forward shape wrong: {mu.shape}"
    mu2, lv2 = model.predict(src, tgt[:, :, 5:6], src[:, -1:, :5])
    assert mu2.shape == (4, 10, 5), f"predict shape wrong: {mu2.shape}"
    print("baseline_model.py: OK")
