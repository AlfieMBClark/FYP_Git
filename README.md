# Maritime AIS Trajectory Prediction & Anomaly Detection
- **Region:** North Sea / Danish waters (lat 50–66°N, lon 5°W–20°E).
- **Task:** from 90 pings of history, predict the next 10 steps of the track.
---

## Requirements

- **Python 3.12** .
- **~8 GB RAM**; a **CUDA GPU is optional** 
- **Internet access** 
- The **AIS databases are not in the repo** (tens of GB). You supply them — see
  [Data](#data).

## Installation
```bash
# from the repository root
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# PyTorch first — pick the build for your machine (see note in requirements.txt):
#   CPU:  pip install "torch>=2.2"
#   CUDA: https://pytorch.org/get-started/locally/
pip install -r requirements.txt

# Optional: only to REBUILD the coastline/land assets (ShipTransformer/utils.py)
# pip install -r requirements-geo.txt
```

All code resolves its paths relative to each file's location, so the repository
runs from any directory on any machine once the databases and checkpoints are in
place.

## Data

Two SQLite databases (Danish Maritime Authority AIS), placed under
`DataHandling/`:

| File | Role | Approx. size |
|---|---|---|
| `DataHandling/training/dma.db` | training + validation | ~65 GB |
| `DataHandling/testing/2023.db` | held-out 2023 traffic (eval + live sim) | ~4.1 GB |

Paths are defined in [`ShipTransformer/config.py`](ShipTransformer/config.py)
(`train_db_path`, `test_db_path`). Build raw AIS into these databases with
`DataHandling/Data_Processing.py`; then cut the model-ready windowed `.bin`
files with `DataHandling/prepare_dataset.py`.

Trained model checkpoints live beside each model
(`Ship*/checkpoints/*.pt`) and are loaded by the dashboard and eval scripts.

---

## Repository layout

Six sibling packages sharing the model + a single-source-of-truth config. Cross-
folder imports are resolved via `__file__`-anchored `sys.path` additions, so what
is evaluated is exactly what is trained and exactly what the dashboard runs.

| Folder | Role | Key modules |
|---|---|---|
| **ShipTransformer/** | Core Transformer model + the single config | `model.py`, `config.py`, `train.py`, `predict.py`, `utils.py` |
| **ShipGRU/** | GRU baseline | `gru_model.py` (`ShipGRUBaseline`), `train_gru.py`, `predict_gru.py` |
| **ShipTCN/** | Temporal-convolutional baseline | `tcn_model.py`, `train_tcn.py`, `predict_tcn.py` |
| **ShipEval/** | Held-out evaluation suite (prediction + anomaly metrics) | `build_eval_set.py`, `evaluate.py`, `evaluate_anomaly.py`, `inject_anomalies.py`, `common.py`, `eda/eda.py` |
| **ShipDashboard/** | Flask + SSE dashboard (Replay & Live-Sim) | `server.py`, `precompute.py`, `sim_engine.py`, `model_registry.py`, `inject_replay.py`, `static/` |
| **DataHandling/** | Data ingestion + preprocessing, SQLite DBs, land raster | `Data_Processing.py`, `prepare_dataset.py`, `dataset.py`, `DataCollection/collect.py` |

---

## Running

Run scripts from inside their own folder (they self-locate, but relative CLI
paths like `--db` are resolved from your working directory). With the venv
active:

### 1. Prepare data (one-time)
```bash
cd DataHandling
python Data_Processing.py --dma <raw-dma-dir> --dma-output training/dma.db --out-dir training/
python prepare_dataset.py  --db training/dma.db --test-db testing/2023.db
```

### 2. Train
```bash
cd ShipTransformer && python train.py            # two-phase Transformer
cd ShipGRU         && python train_gru.py        # GRU baseline (--smoke-test for 1 epoch)
cd ShipTCN         && python train_tcn.py        # TCN baseline
```

### 3. Evaluate (held-out 2023)
```bash
cd ShipEval
python build_eval_set.py       # cache a reproducible eval set (eval_set.npz)
python evaluate.py             # ADE/FDE per horizon, per ship-type, seen/unseen
python inject_anomalies.py     # build labelled synthetic anomalies
python evaluate_anomaly.py     # precision/recall/F1/AUC of the detector
```

### 4. Dashboard
```bash
cd ShipDashboard
python precompute.py --inject-frac 0.5 --n-faulty 5   # builds static/data/ships.json
./run-dashboard.sh                                    # serves http://localhost:8050
#   PYTHON=/path/to/venv/bin/python ./run-dashboard.sh   # choose the interpreter
#   ./run-dashboard.sh --port 9000                        # change the port
```

### 5. Quick visual inference / EDA
```bash
cd ShipTransformer && python predict.py --random --count 10   # writes an HTML map
cd ShipEval/eda    && python eda.py                           # dataset figures
```

---

## Dependencies

Installed by `requirements.txt` (core):

| Package | Used by |
|---|---|
| **torch** | all model code (train / predict / eval / dashboard inference) |
| **numpy** | everywhere (arrays, windowing, metrics) |
| **scipy** | training utilities and metrics |
| **matplotlib** | training curves, prediction maps, EDA figures |
| **folium** | interactive HTML maps from `predict.py` / `predict_*.py` |
| **flask** | dashboard web server (`ShipDashboard/server.py`) |
| **requests** | AIS download in `DataHandling/DataCollection/collect.py` |
| **global-land-mask** | on-land anomaly rule (`precompute.py`, `sim_engine.py`) |

Optional, `requirements-geo.txt` — only to rebuild land assets via
`ShipTransformer/utils.py`: **geopandas, osmnx, pyproj, shapely** (heavy native
GEOS/GDAL/PROJ deps).


The rest of the imports are Python standard library (`argparse`, `sqlite3`,
`json`, `multiprocessing`, `pathlib`, …).
