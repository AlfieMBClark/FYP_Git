"""
model_registry.py
-----------------
All models share the same interface the sim's AR decoder relies on:
    forward(src (B, SEQ_ENC, 14), dec_input (B, t, 6)) -> (mu, log_var)

import the *model class* modules (gru_model.py,tcn_model.py) 
"""

import os
import sys

import torch

_HERE        = os.path.dirname(os.path.abspath(__file__))
_TRANSFORMER = os.path.abspath(os.path.join(_HERE, "..", "ShipTransformer"))
_GRU         = os.path.abspath(os.path.join(_HERE, "..", "ShipGRU"))
_TCN         = os.path.abspath(os.path.join(_HERE, "..", "ShipTCN"))

for _p in (_TRANSFORMER, _GRU, _TCN):
    if _p not in sys.path:
        sys.path.append(_p)

from config import cfg   # noqa: E402  (ShipTransformer/config.py)

_NF       = cfg.n_features            # 14
_N_DEC    = cfg.n_dec_features        # 5
_N_DEC_IN = cfg.n_dec_input_features  # 6
_SEQ_ENC  = cfg.seq_len_enc           # 90


# --- loaders (build from the checkpoint's config) ---

def _load_transformer(ckpt_path: str, device: str):
    from predict import load_model            # ShipTransformer/predict.py
    return load_model(ckpt_path, device)


def _load_gru(ckpt_path: str, device: str):
    from gru_model import ShipGRUBaseline
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    c    = ckpt.get("config", {})
    model = ShipGRUBaseline(
        n_features   = c.get("n_features",   _NF),
        dec_features = c.get("dec_features", _N_DEC_IN),
        out_features = c.get("out_features", _N_DEC),
        hidden_size  = c.get("hidden_size",  256),
        num_layers   = c.get("num_layers",   2),
        dropout      = 0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _load_tcn(ckpt_path: str, device: str):
    from tcn_model import ShipTCNModel
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    c    = ckpt.get("config", {})
    model = ShipTCNModel(
        n_features   = c.get("n_features",   _NF),
        dec_features = c.get("dec_features", _N_DEC_IN),
        out_features = c.get("out_features", _N_DEC),
        hidden_size  = c.get("hidden_size",  256),
        num_layers   = c.get("num_layers",   2),
        dropout      = 0.0,
        seq_len_enc  = c.get("seq_len_enc",  _SEQ_ENC),
        kernel_size  = c.get("kernel_size",  3),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ---registry ---
# `ade`/`fde` are the val-set figures shown in the Model panel.

MODELS = {
    "gru": {
        "label":  "GRU",
        "arch":   "2 layers · h=256",
        "params": "2.46M",
        "ade":    1.30,
        "fde":    2.48,
        "ckpt":   os.path.join(_GRU, "checkpoints", "gru_model.pt"),
        "loader": _load_gru,
    },
    "tcn": {
        "label":  "TCN",
        "arch":   "5 levels · 256 ch",
        "params": "2.65M",
        "ade":    1.38,
        "fde":    2.78,
        "ckpt":   os.path.join(_TCN, "checkpoints", "tcn_model.pt"),
        "loader": _load_tcn,
    },
    "transformer": {
        "label":  "Transformer",
        "arch":   "5 layers · d=128",
        "params": "2.3M",
        "ade":    1.117,
        "fde":    2.02,
        "ckpt":   os.path.join(_TRANSFORMER, "checkpoints", "transformer_model.pt"),
        "loader": _load_transformer,
    },
}

DEFAULT_MODEL = "transformer"


def public_list(active_key: str) -> list:
    """Serialisable model list for the UI """
    return [
        {
            "key":     k,
            "label":   m["label"],
            "arch":    m["arch"],
            "params":  m["params"],
            "ade":     m["ade"],
            "fde":     m["fde"],
            "active":  k == active_key,
            "missing": not os.path.exists(m["ckpt"]),
        }
        for k, m in MODELS.items()
    ]
