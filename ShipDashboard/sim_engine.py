"""
sim_engine.py
-------------
Frame-based AIS streaming simulation engine (v2).

Reads a working set of vessels from the AIS database, replays their pings in
timestamp order on an *accumulator* sim-clock, runs transformer inference per
vessel at a configurable cadence, keeps a rolling set of overlapping
predictions per vessel, and broadcasts events to SSE subscribers.

Key properties (changed from v1):
  * Accumulator clock  — sim_ts advances by  speed * frame_dt  each tick,
    rather than being derived from wall-clock elapsed. This makes the clock
    seekable and pause-exact.
  * Seekable           — seek(unix_ts) re-opens the ping cursor at (or before)
    a target time and replays forward from there. Backward seeks re-read the DB.
  * Thread-safe state  — every read/write of VesselState crosses _vessels_lock.
  * Configurable cadence — re-predict every `predict_every` pings per vessel,
    decoupled from the SEQ_DEC horizon, so predictions overlap and can be
    compared against each other and against ground truth.
  * Rolling predictions — each vessel keeps up to `MAX_PREDS_PER_VESSEL` recent
    predictions, each tagged with the sim-time and ping index it was issued at.
  * Full reset         — reset() returns the engine to a clean stopped state.

Speed multiplier: how many database-seconds advance per real second.
    1728x  → one sim-day per ~50 real seconds
    3600x  → one sim-hour per real second
    7200x  → two sim-hours per real second
"""

import heapq
import math
import os
import queue
import sqlite3
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import torch

# ── path bootstrap ─────────────────────────────────────────────────────────────
# Import from ShipTransformer/ — new architecture with TemporalPositionalEncoding
# and predict_deltas mode matching the 1429km checkpoint.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ST   = os.path.abspath(os.path.join(_HERE, "..", "ShipTransformer"))
if _ST not in sys.path:
    sys.path.insert(0, _ST)

from config  import cfg                                    # noqa: E402
from predict import load_model, rows_to_array, denormalise_dec  # noqa: E402

# Land/water mask for the "on land" detector. Bundled ~1km global mask, no data
# files needed. Optional: if unavailable the on-land detector is skipped.
try:
    from global_land_mask import globe as _land_globe      # noqa: E402
    _LAND_OK = True
except Exception:                                          # pragma: no cover
    _land_globe = None
    _LAND_OK = False

# ── constants ─────────────────────────────────────────────────────────────────
SEQ_ENC   = cfg.seq_len_enc           # 60
SEQ_DEC   = cfg.seq_len_dec           # 10
N_DEC     = cfg.n_dec_features        # 5
N_DEC_IN  = cfg.n_dec_input_features  # 6

LAT_LO, LAT_HI = cfg.norm_bounds["LAT"]
LON_LO, LON_HI = cfg.norm_bounds["LON"]
LAT_RNG = LAT_HI - LAT_LO
LON_RNG = LON_HI - LON_LO

# Delta-prediction bounds (needed when cfg.predict_deltas=True)
_PREDICT_DELTAS = getattr(cfg, "predict_deltas", False)
if _PREDICT_DELTAS:
    _DLAT_LO, _DLAT_HI = cfg.norm_bounds["dLAT"]
    _DLON_LO, _DLON_HI = cfg.norm_bounds["dLON"]
    _DLAT_RNG = _DLAT_HI - _DLAT_LO
    _DLON_RNG = _DLON_HI - _DLON_LO

_REGION_PARAMS = (
    cfg.region_bounds[0], cfg.region_bounds[1],
    cfg.region_bounds[2], cfg.region_bounds[3],
)

# Speed presets exposed to the UI (sim-seconds per real second).
SPEED_STEPS   = [60, 300, 900, 1728, 3600, 7200, 14400]
SPEED_DEFAULT = 900
FRAME_HZ      = 10             # frames per real second
FRAME_SEC     = 1.0 / FRAME_HZ

# Cadence / prediction bookkeeping
DEFAULT_PREDICT_EVERY   = 5    # re-run inference every N pings per vessel
MAX_PREDS_PER_VESSEL    = 4    # rolling window of overlapping predictions kept
MIN_PREDICT_EVERY       = 1
MAX_PREDICT_EVERY       = 30

# Batched inference — group up to N vessels per GPU/CPU call, wait up to T sec
INFER_BATCH_SIZE = 8
INFER_WAIT_SEC   = 0.03

# Frame budget: max pings processed per 100ms frame. At extreme speed
# multipliers (14400x) a frame can span 24 sim-minutes of a dense dataset;
# capping the work keeps the loop responsive — the sim clock is simply held
# back to the last processed ping and catches up over following frames.
MAX_PINGS_PER_FRAME = 4000

# Working-set stats cache table built once per DB file (full-table GROUP BY
# takes ~40s on a 25M-row DB; reading the cache takes microseconds).
_STATS_TABLE = "dashboard_vessel_stats"

# Data-derived "normal stationary zones" (anchorages/ports). Cells where many
# vessels are historically stationary; built once per DB, read into a set.
_ANCHOR_TABLE = "dashboard_anchorages"
# Per-cell normal dwell (how long ships usually stay stationary in each place),
# so loitering is flagged relative to what is normal *for that location*.
_DWELL_TABLE  = "dashboard_dwell"

# ── rule-based physical-anomaly detectors ──────────────────────────────────────
# These run per ping, independently of the model, and raise *typed* alerts.
LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = cfg.region_bounds   # (50, 66, -5, 20)

JUMP_MIN_KM        = 5.0     # ignore sub-5km moves (normal reporting noise)
JUMP_MAX_GAP_SEC   = 900     # only "impossible" within 15 min — longer = coverage gap
JUMP_MAX_KN        = 80.0    # implied speed above this is physically impossible
SOG_MAX_KN         = 60.0    # a reported SOG above this is a sensor/spoof value
STATIONARY_SOG_KN  = 0.5     # below this the vessel is "not moving"
ONLAND_MIN_RUN     = 2       # consecutive on-land-while-moving pings before flagging
ONLAND_MIN_SOG     = 1.0     # only land-check moving vessels (docked ships sit on port pixels)
ANCHOR_CELL_DEG    = 0.02    # ~1–2 km anchorage grid resolution
ANCHOR_MIN_PINGS   = 100     # stationary pings in a cell to call it a normal zone
ALERT_COOLDOWN     = 30      # pings before the same alert kind re-fires for a vessel

# Movement gate — the Transformer was trained only on moving vessels
# (cfg.max_sog_minimum etc.), so a stationary ship is out of distribution and
# produces garbage predictions + false anomalies. Below this recent-mean SOG we
# skip inference entirely and treat the vessel as "holding position".
MOVING_SOG_MIN   = 1.0      # knots; recent-mean SOG below this → hold, don't predict
HOLD_SOG_WINDOW  = 12       # pings averaged for the movement test

# Loitering — confined to a small area longer than is normal for that location.
LOITER_RADIUS_KM  = 0.75    # stay within this radius of an anchor point = "same spot"
LOITER_MARGIN     = 1.2     # flag once dwell exceeds the cell's normal dwell × this
LOITER_OPEN_SEC   = 7200    # threshold where the location has no learned norm (2 h)
LOITER_MIN_SEC    = 1800    # never flag a stay shorter than 30 min
BERTH_LAND_DEG    = 0.01    # ~1 km: land this close counts as "alongside a quay"
DWELL_RUN_GAP_SEC = 3600    # >1 h between stationary pings = a separate visit
DWELL_PCTL        = 90      # per-cell "normal max" dwell = this percentile of visits
DWELL_MIN_SAMPLES = 5       # visits needed before a per-type norm is trusted
_DWELL_ALL        = -1      # type_group sentinel for a cell's all-type norm

# Loitering is only meaningful for vessels that are *supposed* to be transiting.
# Working vessels hold station as their job: measured on this dataset, tugs and
# fishing vessels tripped the alert 100% of the time (median dwell 56 h and 96 h
# respectively) and together produced 51% of all loitering alerts, while cargo
# and tankers produced none. The per-cell dwell norm is averaged over all types,
# so it can never accommodate them — the type has to be gated, not tuned.
# Groups: 0 Unknown · 1 Cargo · 2 Tanker · 3 Passenger · 4 Fishing
#         5 Tug/Service · 6 Pleasure/Sail · 7 Other
LOITER_SHIP_TYPES = frozenset({1, 2, 3})   # Cargo, Tanker, Passenger

GROUP_NAMES = ["Unknown", "Cargo", "Tanker", "Passenger",
               "Fishing", "Tug/Service", "Pleasure/Sail", "Other"]

# Accepted timestamp formats (first match wins)
_TS_FORMATS = (
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S.%f",
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_ts(ts_str) -> float:
    """Parse a DB timestamp to a UTC unix float. Returns NaN on failure so
    unparseable rows can be filtered rather than silently sorted to the epoch."""
    if ts_str is None:
        return float("nan")
    # Fast path: already numeric (unix seconds)
    if isinstance(ts_str, (int, float)):
        return float(ts_str)
    s = str(ts_str).strip()
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp()
        except (ValueError, TypeError):
            continue
    return float("nan")


def _fmt_ts(ts: float) -> str:
    if not ts or math.isnan(ts):
        return "—"
    return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _fmt_ts_sql(ts: float) -> str:
    """Format for SQL comparison against stored TIMESTAMP values.
    The DB stores 'YYYY-MM-DDTHH:MM:SS'; using a space separator here would
    sort *before* 'T' and silently pull in up to a day of earlier pings."""
    return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%S")


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in km between two lat/lon points (degrees)."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi   = math.radians(lat2 - lat1)
    dlmb   = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2)
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class _DictRow:
    """sqlite3.Row-compatible wrapper for a plain dict."""
    __slots__ = ("_d",)

    def __init__(self, d: dict):
        self._d = d

    def __getitem__(self, key):
        return self._d.get(key)

    def keys(self):
        return self._d.keys()


class _MergedPingCursor:
    """Streams pings for a set of vessels in global TIMESTAMP order.

    Opens one range cursor per MMSI (each hits the UNIQUE(MMSI, TIMESTAMP)
    index directly — no scan, no temp-sort) and heap-merges them on the
    TIMESTAMP string, which sorts correctly in ISO-8601 form. The previous
    single `MMSI IN (...) ORDER BY TIMESTAMP` query forced SQLite to
    materialise and sort every remaining row before returning the first one,
    which froze the UI for seconds on every seek."""

    def __init__(self, conn, mmsis, from_ts=None):
        self._heap    = []       # (timestamp_str, cursor_id, row)
        self._cursors = {}
        start = _fmt_ts_sql(from_ts) if from_ts is not None else None
        for m in mmsis:
            cur = conn.cursor()
            if start is not None:
                cur.execute(
                    "SELECT * FROM ais WHERE MMSI=? AND FLAGS=0"
                    "  AND LAT BETWEEN ? AND ? AND LON BETWEEN ? AND ?"
                    "  AND TIMESTAMP >= ? ORDER BY TIMESTAMP",
                    (m, *_REGION_PARAMS, start),
                )
            else:
                cur.execute(
                    "SELECT * FROM ais WHERE MMSI=? AND FLAGS=0"
                    "  AND LAT BETWEEN ? AND ? AND LON BETWEEN ? AND ?"
                    "  ORDER BY TIMESTAMP",
                    (m, *_REGION_PARAMS),
                )
            row = cur.fetchone()
            if row is not None:
                cid = id(cur)
                self._cursors[cid] = cur
                heapq.heappush(self._heap, (row["TIMESTAMP"], cid, row))

    def fetchone(self):
        if not self._heap:
            return None
        _, cid, row = heapq.heappop(self._heap)
        cur = self._cursors[cid]
        nxt = cur.fetchone()
        if nxt is not None:
            heapq.heappush(self._heap, (nxt["TIMESTAMP"], cid, nxt))
        else:
            cur.close()
            del self._cursors[cid]
        return row

    def close(self):
        for cur in self._cursors.values():
            cur.close()
        self._cursors.clear()
        self._heap.clear()


# ── vessel state ──────────────────────────────────────────────────────────────

class Prediction:
    """A single issued prediction, tagged with when it was made."""
    __slots__ = ("pred_id", "issued_ts", "issued_ping_idx",
                 "mu", "sigma", "dec_raw", "steps_matched")

    def __init__(self, pred_id, issued_ts, issued_ping_idx, mu, sigma, dec_raw):
        self.pred_id         = pred_id
        self.issued_ts       = issued_ts
        self.issued_ping_idx = issued_ping_idx
        self.mu              = mu       # (SEQ_DEC, N_DEC) normalised mean
        self.sigma           = sigma    # (SEQ_DEC, N_DEC) normalised std
        self.dec_raw         = dec_raw  # (SEQ_DEC, N_DEC) denormalised
        self.steps_matched   = 0        # how many future pings compared so far


class VesselState:
    __slots__ = (
        "mmsi", "ship_type_group", "rows",
        "ping_idx", "last_pred_ping_idx",
        "preds", "next_pred_id", "inferring",
        "last_lat", "last_lon", "last_sog", "last_cog",
        # rule-based physical-anomaly detector state
        "last_ts", "onland_run", "alerts", "holding",
        "loiter_lat", "loiter_lon", "loiter_since",
    )

    def __init__(self, mmsi: int):
        self.mmsi               = mmsi
        self.ship_type_group    = 0       # Unknown until a real SHIP_TYPE arrives
        self.rows: list         = []      # raw row dicts for rows_to_array
        self.ping_idx           = 0       # total pings seen for this vessel
        self.last_pred_ping_idx = -10**9  # ping_idx at last inference trigger
        self.preds: list        = []      # rolling list[Prediction], newest last
        self.next_pred_id       = 0
        self.inferring          = False
        self.last_lat           = None
        self.last_lon           = None
        self.last_sog           = 0.0
        self.last_cog           = 0.0
        self.last_ts            = None    # unix ts of previous ping (speed-jump)
        self.onland_run         = 0       # consecutive on-land-while-moving pings
        self.alerts: dict       = {}      # kind -> ping_idx it last fired (cooldown)
        self.holding            = False   # stationary → model skipped, holding position
        self.loiter_lat         = None    # anchor point of the current confinement
        self.loiter_lon         = None
        self.loiter_since       = None    # unix ts the confinement started


# ── simulation engine ─────────────────────────────────────────────────────────

class SimEngine:
    """Thread-safe, frame-based, seekable AIS replay simulation engine."""

    def __init__(self, db_path: str, checkpoint_path: str, device: str = "cpu",
                 model_key: str = "transformer", loader=None):
        self.db_path         = db_path
        self.checkpoint_path = checkpoint_path
        self.device          = device
        self.model_key       = model_key   # which registry entry is loaded
        # How to build the model from a checkpoint. Defaults to the transformer
        # loader; set_model() swaps it along with the weights.
        self._loader         = loader or load_model

        # user-configurable
        self.speed             = float(SPEED_DEFAULT)
        self.max_vessels       = 50
        self.anomaly_threshold = 3.0
        self.predict_every     = DEFAULT_PREDICT_EVERY

        # data time span (filled on first model/db touch)
        self.data_start_ts     = None
        self.data_end_ts       = None

        # rule-based detector resources (filled by _prepare_db)
        self._anchor_cells: set = set()   # {(ilat, ilon)} = normal stationary zones
        self._cell_dwell: dict  = {}      # (ilat, ilon) -> normal dwell seconds

        # runtime
        self._state         = "stopped"   # stopped | running | paused | done
        self._model         = None
        self._model_lock    = threading.Lock()
        self._start_lock    = threading.Lock()

        self._stop_evt      = threading.Event()
        self._pause_evt     = threading.Event()
        self._pause_evt.set()

        # A pending seek target (unix ts). Picked up by the sim loop, which
        # restarts its cursor. None means "no seek pending".
        self._seek_target   = None
        self._seek_lock     = threading.Lock()

        self._vessels: dict[int, VesselState] = {}
        self._vessels_lock  = threading.Lock()

        # Monotonic generation counter. Bumped on start/seek/reset so inference
        # results computed from pre-seek snapshots are recognised as stale and
        # dropped instead of being attached to the freshly rebuilt VesselStates.
        self._epoch         = 0

        # In-memory working-set cache (fallback when the DB is read-only and
        # the stats table can't be created): max_vessels -> [mmsi, ...]
        self._ws_cache: dict[int, list] = {}
        self._db_prepared   = False

        self._sim_ts        = 0.0
        self._n_pings       = 0
        self._n_anomalies   = 0        # total anomalous events (all kinds)
        self._n_predictions = 0
        self._anom_by_kind: dict = {}   # kind -> count, for the status readout
        self._anom_ships: set  = set()  # distinct MMSIs that have been flagged

        # Online deviation calibration. The detector scores raw km-deviation from
        # the predicted position (the model's variance head is over-dispersed and
        # unusable for a z-score, which is why replay switched to km-deviation too).
        # We rescale that raw km by the fleet's recent 90th-percentile so the score
        # behaves like a unit-normal deviate: a single fleet-wide scale (not
        # per-ship-type), so threshold 3σ means "3× the typical current deviation".
        self._z_recent: list = []       # rolling raw km-deviations (bounded)
        self._z_scale        = None     # raw km p90 / 1.2816
        self._z_since_calib  = 0

        self._clients: list[queue.Queue] = []
        self._clients_lock  = threading.Lock()

        self._infer_q: queue.Queue     = queue.Queue()
        self._infer_thread: Optional[threading.Thread] = None
        self._sim_thread:   Optional[threading.Thread] = None

    # ── public API ─────────────────────────────────────────────────────────────

    def configure(self, speed=None, max_vessels=None,
                  anomaly_threshold=None, predict_every=None):
        """Live-applicable where possible. speed/threshold/predict_every take
        effect on the next frame. max_vessels triggers a working-set refresh
        the next time the sim loop reopens its cursor (or immediately if it
        forces a soft seek to the current clock)."""
        if speed is not None:
            self.speed = float(max(1.0, speed))
        if anomaly_threshold is not None:
            self.anomaly_threshold = float(anomaly_threshold)
        if predict_every is not None:
            self.predict_every = int(
                max(MIN_PREDICT_EVERY, min(MAX_PREDICT_EVERY, predict_every))
            )
        if max_vessels is not None:
            new_mv = int(max_vessels)
            changed = new_mv != self.max_vessels
            self.max_vessels = new_mv
            # If running, apply immediately by soft-seeking to the current clock,
            # which reopens the working set with the new vessel count.
            if changed and self._state in ("running", "paused"):
                self.seek(self._sim_ts)

    def start(self):
        with self._start_lock:
            if self._state in ("running", "paused"):
                return
            self._teardown_threads()
            self._drain_infer_q()
            self._reset_runtime()
            self._epoch += 1
            self._stop_evt.clear()
            self._pause_evt.set()
            with self._seek_lock:
                self._seek_target = None
            self._state = "running"
            self._spawn_threads()

    def stop(self):
        self._stop_evt.set()
        self._pause_evt.set()
        self._state = "stopped"

    def pause(self):
        if self._state == "running":
            self._pause_evt.clear()
            self._state = "paused"

    def resume(self):
        if self._state == "paused":
            self._pause_evt.set()
            self._state = "running"

    def reset(self):
        """Full reset to a clean stopped state: stop threads, clear vessels,
        counters, feed state, and broadcast a reset event so clients wipe."""
        self._stop_evt.set()
        self._pause_evt.set()
        self._teardown_threads()
        self._drain_infer_q()
        self._reset_runtime()
        self._epoch += 1
        self._state = "stopped"
        with self._seek_lock:
            self._seek_target = None
        self._broadcast([{"type": "reset"},
                         {"type": "status", "state": "stopped",
                          "sim_time": "—", "n_pings": 0,
                          "n_vessels": 0, "n_anomalies": 0}])

    def seek(self, unix_ts: float):
        """Request the sim loop to jump the clock to unix_ts and replay from
        there. Clears live vessel render state (predictions/history reset)."""
        try:
            target = float(unix_ts)
        except (TypeError, ValueError):
            return
        with self._seek_lock:
            self._seek_target = target
        # Instant UI feedback: tell clients a seek is in flight before the sim
        # loop (which may be mid-frame) picks it up.
        self._broadcast([{"type": "status", "state": "seeking",
                          "sim_time": _fmt_ts(target), "sim_ts": target}])
        # Ensure the loop is not blocked in pause so it can pick up the seek.
        self._pause_evt.set()
        if self._state == "paused":
            self._state = "running"

    def status(self) -> dict:
        return {
            "state":             self._state,
            "model":             self.model_key,
            "speed":             self.speed,
            "sim_time":          _fmt_ts(self._sim_ts),
            "sim_ts":            self._sim_ts,
            "pings_processed":   self._n_pings,
            "active_vessels":    len(self._vessels),
            "anomalies_flagged": self._n_anomalies,
            "anomalous_ships":   len(self._anom_ships),
            "anomaly_kinds":     dict(self._anom_by_kind),
            "predictions_made":  self._n_predictions,
            "anomaly_threshold": self.anomaly_threshold,
            "max_vessels":       self.max_vessels,
            "predict_every":     self.predict_every,
            "data_start_ts":     self.data_start_ts,
            "data_end_ts":       self.data_end_ts,
            "data_start":        _fmt_ts(self.data_start_ts) if self.data_start_ts else "—",
            "data_end":          _fmt_ts(self.data_end_ts) if self.data_end_ts else "—",
            "speed_steps":       SPEED_STEPS,
        }

    def subscribe(self) -> "queue.Queue":
        q: queue.Queue = queue.Queue(maxsize=1000)
        with self._clients_lock:
            self._clients.append(q)
        return q

    def unsubscribe(self, q: "queue.Queue"):
        with self._clients_lock:
            try:
                self._clients.remove(q)
            except ValueError:
                pass

    # ── internals: lifecycle ─────────────────────────────────────────────────

    def _load_model(self):
        with self._model_lock:
            if self._model is None:
                print(f"[SimEngine] loading model ({self.model_key}) ...", flush=True)
                self._model = self._loader(self.checkpoint_path, self.device)
                print("[SimEngine] model ready", flush=True)
        self._prepare_db()
        # Opportunistically discover the data time span once.
        if self.data_start_ts is None:
            self._discover_time_span()

    def set_model(self, key: str, loader, ckpt_path: str):
        """Hot-swap the inference model, safe to call while the sim is running.

        Everything derived from the previous model is invalidated: in-flight
        inference is dropped (epoch bump), live predictions are cleared so the
        old model's tracks can't raise anomalies against new pings, and the
        deviation calibration is reset because its scale is model-specific.
        Raises on load failure, leaving the previous model in place."""
        model = loader(ckpt_path, self.device)   # build first — don't clobber on error
        with self._model_lock:
            self._model          = model
            self._loader         = loader
            self.checkpoint_path = ckpt_path
            self.model_key       = key

        self._epoch += 1          # results computed with the old model are stale
        self._drain_infer_q()
        with self._vessels_lock:
            for vs in self._vessels.values():
                vs.preds     = []
                vs.inferring = False   # never leave a vessel stuck mid-inference
        self._z_recent      = []
        self._z_scale       = None
        self._z_since_calib = 0

        print(f"[SimEngine] model switched -> {key} ({ckpt_path})", flush=True)
        self._broadcast([{"type": "reset_preds"},
                         {"type": "status", "state": self._state, "model": key}])

    # ── one-time DB preparation ────────────────────────────────────────────────

    def _prepare_db(self):
        """One-time per DB file: verify the indexes the hot path relies on and
        build the vessel-stats cache table.

        Required indexes: (MMSI, TIMESTAMP) — satisfied by the table's
        UNIQUE(MMSI, TIMESTAMP) autoindex — and TIMESTAMP. Both exist in the
        standard schema; if missing they are created here (a one-time cost).

        The stats table replaces a full-table GROUP BY (~40s on 25M rows) that
        used to run on every working-set selection, i.e. on every start *and*
        every seek. With the cache, selection is a sub-millisecond lookup."""
        if self._db_prepared:
            return
        try:
            conn = sqlite3.connect(self.db_path)
            cur  = conn.cursor()

            # Index check: collect the leading columns of every index on ais.
            cur.execute("PRAGMA index_list('ais')")
            idx_names = [r[1] for r in cur.fetchall()]
            leading_cols = set()
            pair_indexed = False
            for name in idx_names:
                cur.execute(f"PRAGMA index_info('{name}')")
                cols = [r[2] for r in cur.fetchall()]
                if cols:
                    leading_cols.add(cols[0])
                if cols[:2] == ["MMSI", "TIMESTAMP"]:
                    pair_indexed = True
            if not pair_indexed:
                print("[SimEngine] creating (MMSI, TIMESTAMP) index (one-time) ...", flush=True)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_mmsi_ts ON ais (MMSI, TIMESTAMP)")
            if "TIMESTAMP" not in leading_cols:
                print("[SimEngine] creating TIMESTAMP index (one-time) ...", flush=True)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON ais (TIMESTAMP)")

            cur.execute(
                f"SELECT name FROM sqlite_master WHERE type='table' AND name='{_STATS_TABLE}'")
            have_stats = cur.fetchone() is not None
            if have_stats:
                cur.execute(f"SELECT COUNT(*) FROM {_STATS_TABLE}")
                have_stats = cur.fetchone()[0] > 0
            if not have_stats:
                print("[SimEngine] building vessel-stats cache (one-time, ~1 min on large DBs) ...",
                      flush=True)
                cur.execute(f"DROP TABLE IF EXISTS {_STATS_TABLE}")
                cur.execute(
                    f"CREATE TABLE {_STATS_TABLE} AS"
                    "  SELECT MMSI,"
                    "         COUNT(*) AS cnt,"
                    "         COUNT(DISTINCT ROUND(LAT,2)||','||ROUND(LON,2)) AS uniq_pos,"
                    "         MIN(TIMESTAMP) AS min_ts,"
                    "         MAX(TIMESTAMP) AS max_ts"
                    "  FROM ais"
                    "  WHERE LAT BETWEEN ? AND ? AND LON BETWEEN ? AND ? AND FLAGS=0"
                    "  GROUP BY MMSI",
                    _REGION_PARAMS,
                )
                conn.commit()
                print("[SimEngine] vessel-stats cache ready", flush=True)

            # Anchorage / normal-stationary-zone cache: grid cells where many
            # vessels are historically stationary (ports, anchorages, moorings).
            # A sustained-stationary vessel *outside* these cells is anomalous.
            cur.execute(
                f"SELECT name FROM sqlite_master WHERE type='table' AND name='{_ANCHOR_TABLE}'")
            have_anchor = cur.fetchone() is not None
            if not have_anchor:
                print("[SimEngine] building anchorage-zone cache (one-time, ~1 min) ...",
                      flush=True)
                cur.execute(f"DROP TABLE IF EXISTS {_ANCHOR_TABLE}")
                cur.execute(
                    f"CREATE TABLE {_ANCHOR_TABLE} AS"
                    "  SELECT CAST((LAT - ?) / ? AS INT) AS ilat,"
                    "         CAST((LON - ?) / ? AS INT) AS ilon,"
                    "         COUNT(*) AS cnt"
                    "  FROM ais"
                    "  WHERE SOG < ? AND FLAGS=0"
                    "    AND LAT BETWEEN ? AND ? AND LON BETWEEN ? AND ?"
                    "  GROUP BY ilat, ilon HAVING cnt >= ?",
                    (LAT_MIN, ANCHOR_CELL_DEG, LON_MIN, ANCHOR_CELL_DEG,
                     STATIONARY_SOG_KN, *_REGION_PARAMS, ANCHOR_MIN_PINGS),
                )
                conn.commit()
                print("[SimEngine] anchorage-zone cache ready", flush=True)

            cur.execute(f"SELECT ilat, ilon FROM {_ANCHOR_TABLE}")
            self._anchor_cells = {(r[0], r[1]) for r in cur.fetchall()}

            # Normal dwell per (cell, ship-type): the 90th-percentile duration of
            # *contiguous* stationary visits. Loitering is judged relative to what
            # is normal for that location *and that vessel type* — a ferry berthed
            # at its terminal is measured against other ferries there, not against
            # the cell's all-type average. Open water has no norm (LOITER_OPEN_SEC).
            self._build_dwell_table(conn, cur)
            cur.execute(f"SELECT ilat, ilon, type_group, thresh_sec FROM {_DWELL_TABLE}")
            self._cell_dwell = {(r[0], r[1], r[2]): r[3] for r in cur.fetchall()}
            _n_typed = sum(1 for k in self._cell_dwell if k[2] != _DWELL_ALL)
            _n_cells = len({(k[0], k[1]) for k in self._cell_dwell})
            print(f"[SimEngine] {len(self._anchor_cells)} anchorage cells, "
                  f"{_n_cells} dwell cells ({_n_typed} per-type norms); "
                  f"land mask {'on' if _LAND_OK else 'OFF'}", flush=True)

            conn.close()
            self._db_prepared = True
        except Exception as exc:
            # Read-only DB or locked file: fall back to per-process in-memory
            # caching (_select_working_set handles it).
            print(f"[SimEngine] DB preparation skipped: {exc}", flush=True)

    def _build_dwell_table(self, conn, cur):
        """Learn normal dwell per (cell, ship-type) from contiguous stationary visits.

        A 'visit' is a maximal run of low-SOG pings with < DWELL_RUN_GAP_SEC
        between consecutive pings; its dwell is (last - first) timestamp. For
        each cell we store the DWELL_PCTL percentile of visit dwells, both
        per ship-type group and — as row type_group=_DWELL_ALL — across all
        types, so a type with too few local samples can fall back to the
        location's overall norm.

        Learning per type matters: tugs and fishing vessels routinely hold
        station for days while cargo ships pass through in hours, so a single
        all-type norm per cell is wrong for both. Built once (~1–2 min on a
        large DB) and cached in _DWELL_TABLE."""
        # The table gained a type_group column — rebuild any older schema.
        cur.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{_DWELL_TABLE}'")
        if cur.fetchone() is not None:
            cols = {r[1] for r in cur.execute(f"PRAGMA table_info('{_DWELL_TABLE}')")}
            cur.execute(f"SELECT COUNT(*) FROM {_DWELL_TABLE}")
            if "type_group" in cols and cur.fetchone()[0] > 0:
                return
            print("[SimEngine] dwell cache has an old schema — rebuilding", flush=True)

        print("[SimEngine] learning per-(cell, ship-type) dwell norms "
              "(one-time, ~1–2 min) ...", flush=True)
        rows = cur.execute(
            "WITH s AS ("
            "  SELECT MMSI, SHIP_TYPE,"
            "         CAST((LAT - ?) / ? AS INT) AS ilat,"
            "         CAST((LON - ?) / ? AS INT) AS ilon,"
            "         julianday(TIMESTAMP)*86400.0 AS t,"
            "         LAG(julianday(TIMESTAMP)*86400.0)"
            "           OVER (PARTITION BY MMSI ORDER BY TIMESTAMP) AS pt"
            "  FROM ais"
            "  WHERE SOG < ? AND FLAGS=0"
            "    AND LAT BETWEEN ? AND ? AND LON BETWEEN ? AND ?"
            "),"
            "runs AS ("
            "  SELECT MMSI, ilat, ilon, SHIP_TYPE, t,"
            "         SUM(CASE WHEN pt IS NULL OR (t - pt) > ? THEN 1 ELSE 0 END)"
            "             OVER (PARTITION BY MMSI ORDER BY t) AS run_id"
            "  FROM s"
            ")"
            "SELECT ilat, ilon, MAX(SHIP_TYPE) AS st,"
            "       (MAX(t) - MIN(t)) AS dwell_sec, COUNT(*) n"
            "  FROM runs GROUP BY MMSI, run_id, ilat, ilon HAVING n >= 3",
            (LAT_MIN, ANCHOR_CELL_DEG, LON_MIN, ANCHOR_CELL_DEG,
             STATIONARY_SOG_KN, *_REGION_PARAMS, DWELL_RUN_GAP_SEC),
        ).fetchall()

        by_cell: dict = {}   # (ilat, ilon)        -> [dwell, ...]  (all types)
        by_type: dict = {}   # (ilat, ilon, group) -> [dwell, ...]
        for ilat, ilon, st, dwell_sec, _ in rows:
            try:
                grp = cfg.ship_type_groups.get(int(float(st or 0)), 7)
            except (ValueError, TypeError):
                grp = 7
            by_cell.setdefault((ilat, ilon), []).append(dwell_sec)
            by_type.setdefault((ilat, ilon, grp), []).append(dwell_sec)

        out = [(ilat, ilon, _DWELL_ALL, float(np.percentile(v, DWELL_PCTL)), len(v))
               for (ilat, ilon), v in by_cell.items()]
        # A per-type norm is only trusted with enough local visits; otherwise the
        # vessel falls back to the cell's all-type norm at lookup time.
        out += [(ilat, ilon, grp, float(np.percentile(v, DWELL_PCTL)), len(v))
                for (ilat, ilon, grp), v in by_type.items()
                if len(v) >= DWELL_MIN_SAMPLES]

        cur.execute(f"DROP TABLE IF EXISTS {_DWELL_TABLE}")
        cur.execute(f"CREATE TABLE {_DWELL_TABLE} "
                    "(ilat INT, ilon INT, type_group INT, thresh_sec REAL, n INT)")
        cur.executemany(f"INSERT INTO {_DWELL_TABLE} VALUES (?,?,?,?,?)", out)
        conn.commit()
        print(f"[SimEngine] dwell norms ready ({len(by_cell)} cells)", flush=True)

    def _discover_time_span(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cur  = conn.cursor()
            if self._db_prepared:
                cur.execute(f"SELECT MIN(min_ts), MAX(max_ts) FROM {_STATS_TABLE}")
            else:
                cur.execute(
                    "SELECT MIN(TIMESTAMP), MAX(TIMESTAMP) FROM ais "
                    "WHERE LAT BETWEEN ? AND ? AND LON BETWEEN ? AND ? AND FLAGS=0",
                    _REGION_PARAMS,
                )
            row = cur.fetchone()
            conn.close()
            if row and row[0] and row[1]:
                s, e = _parse_ts(row[0]), _parse_ts(row[1])
                if not math.isnan(s):
                    self.data_start_ts = s
                if not math.isnan(e):
                    self.data_end_ts = e
                print(f"[SimEngine] data span {_fmt_ts(s)} → {_fmt_ts(e)}", flush=True)
        except Exception as exc:
            print(f"[SimEngine] time-span discovery failed: {exc}", flush=True)

    def _teardown_threads(self):
        self._stop_evt.set()
        self._pause_evt.set()
        if self._sim_thread and self._sim_thread.is_alive():
            self._sim_thread.join(timeout=3.0)
        if self._infer_thread and self._infer_thread.is_alive():
            self._infer_thread.join(timeout=3.0)

    def _spawn_threads(self):
        self._infer_thread = threading.Thread(
            target=self._infer_loop, daemon=True, name="sim-infer")
        self._sim_thread = threading.Thread(
            target=self._sim_loop, daemon=True, name="sim-stream")
        self._infer_thread.start()
        self._sim_thread.start()

    def _drain_infer_q(self):
        while True:
            try:
                self._infer_q.get_nowait()
            except queue.Empty:
                break

    def _reset_runtime(self):
        with self._vessels_lock:
            self._vessels.clear()
        self._sim_ts        = 0.0
        self._n_pings       = 0
        self._n_anomalies   = 0
        self._n_predictions = 0
        self._anom_by_kind  = {}
        self._anom_ships    = set()
        self._z_recent      = []
        self._z_scale       = None
        self._z_since_calib = 0

    def _broadcast(self, events: list):
        if not events:
            return
        with self._clients_lock:
            for q in self._clients:
                try:
                    q.put_nowait(events)
                except queue.Full:
                    pass

    # ── internals: DB working set + cursor ───────────────────────────────────

    def _select_working_set(self, cur) -> list:
        """Pick the max_vessels busiest non-stationary vessels. Reads the
        one-time stats cache table when available; otherwise falls back to the
        full GROUP BY, memoised per max_vessels for the process lifetime so it
        never runs on the seek path more than once."""
        mv = self.max_vessels
        if self._db_prepared:
            cur.execute(
                f"SELECT MMSI FROM {_STATS_TABLE}"
                "  WHERE uniq_pos >= 10 ORDER BY cnt DESC LIMIT ?",
                (mv,),
            )
            return [r[0] for r in cur.fetchall()]
        if mv in self._ws_cache:
            return self._ws_cache[mv]
        cur.execute(
            "SELECT MMSI FROM ("
            "  SELECT MMSI, COUNT(*) AS cnt,"
            "    COUNT(DISTINCT ROUND(LAT,2)||','||ROUND(LON,2)) AS uniq_pos"
            "  FROM ais"
            "  WHERE LAT BETWEEN ? AND ? AND LON BETWEEN ? AND ?"
            "  AND FLAGS=0"
            "  GROUP BY MMSI"
            "  HAVING uniq_pos >= 10"
            "  ORDER BY cnt DESC LIMIT ?"
            ")",
            (*_REGION_PARAMS, mv),
        )
        result = [r[0] for r in cur.fetchall()]
        self._ws_cache[mv] = result
        return result

    def _open_ping_cursor(self, conn, mmsis, from_ts=None):
        """Open a merged per-vessel ping stream, optionally starting at or
        after from_ts (used for seeking). See _MergedPingCursor for why this
        is not a single IN(...) ORDER BY query."""
        return _MergedPingCursor(conn, mmsis, from_ts=from_ts)

    # ── simulation loop ──────────────────────────────────────────────────────

    def _sim_loop(self):
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            setup_cur = conn.cursor()

            selected_mmsis = self._select_working_set(setup_cur)
            if not selected_mmsis:
                print("[SimEngine] no vessels found in region", flush=True)
                self._broadcast([{"type": "status", "state": "done",
                                  "sim_time": "—"}])
                return

            with self._vessels_lock:
                self._vessels.clear()
                for m in selected_mmsis:
                    self._vessels[m] = VesselState(m)

            print(f"[SimEngine] tracking {len(selected_mmsis)} vessels", flush=True)

            cur = self._open_ping_cursor(conn, selected_mmsis)
            first_row = cur.fetchone()
            if first_row is None:
                self._broadcast([{"type": "status", "state": "done",
                                  "sim_time": "—"}])
                return

            sim_origin   = _parse_ts(first_row["TIMESTAMP"])
            self._sim_ts = sim_origin
            pending_rows = [first_row]

            self._broadcast([{
                "type": "status", "state": "running",
                "sim_time": _fmt_ts(sim_origin),
                "n_vessels": len(selected_mmsis),
                "data_start_ts": self.data_start_ts,
                "data_end_ts": self.data_end_ts,
            }])

            while not self._stop_evt.is_set():
                self._pause_evt.wait()
                if self._stop_evt.is_set():
                    break

                # ── handle a pending seek ──────────────────────────────────
                seek_to = None
                with self._seek_lock:
                    if self._seek_target is not None:
                        seek_to = self._seek_target
                        self._seek_target = None
                if seek_to is not None:
                    # Invalidate any in-flight inference snapshots from before
                    # the jump so their results are dropped, not attached to
                    # the rebuilt vessel states.
                    self._epoch += 1
                    # Refresh working set (max_vessels may have changed) and
                    # reopen the cursor at the target time. Both are now
                    # sub-millisecond (stats cache + per-MMSI index cursors).
                    selected_mmsis = self._select_working_set(setup_cur) or selected_mmsis
                    with self._vessels_lock:
                        self._vessels.clear()
                        for m in selected_mmsis:
                            self._vessels[m] = VesselState(m)
                    self._drain_infer_q()
                    cur.close()
                    cur = self._open_ping_cursor(conn, selected_mmsis, from_ts=seek_to)
                    pending_rows = []
                    self._sim_ts = seek_to
                    sim_origin   = seek_to

                    # Pre-process one frame of pings so "reset" and initial
                    # vessel positions arrive in the same SSE batch — eliminates
                    # the blank-map gap after a timeline seek.
                    self._sim_ts += self.speed * FRAME_SEC
                    seed_events: list = []
                    self._process_pings_until(cur, pending_rows, self._sim_ts, seed_events)
                    self._broadcast([
                        {"type": "reset"},
                        {"type": "status", "state": self._state,
                         "sim_time": _fmt_ts(seek_to),
                         "n_vessels": len(selected_mmsis),
                         "seeked": True},
                        *seed_events,
                    ])

                # ── advance accumulator clock ──────────────────────────────
                frame_start = time.time()
                self._sim_ts += self.speed * FRAME_SEC
                current_sim  = self._sim_ts

                frame_events: list = []
                exhausted = self._process_pings_until(
                    cur, pending_rows, current_sim, frame_events)
                # _process_pings_until may have pulled the clock back if the
                # per-frame ping budget was exhausted.
                current_sim = self._sim_ts

                # periodic status
                with self._vessels_lock:
                    active = sum(1 for v in self._vessels.values() if v.last_lat is not None)
                frame_events.append({
                    "type": "status", "state": self._state,
                    "sim_time": _fmt_ts(current_sim), "sim_ts": current_sim,
                    "n_pings": self._n_pings, "n_vessels": active,
                    "n_anomalies": self._n_anomalies,
                    "n_anom_ships": len(self._anom_ships),
                    "n_predictions": self._n_predictions,
                    "progress": self._progress_fraction(current_sim),
                })
                self._broadcast(frame_events)

                if exhausted:
                    self._state = "done"
                    self._broadcast([{"type": "status", "state": "done",
                                      "sim_time": _fmt_ts(current_sim),
                                      "sim_ts": current_sim}])
                    return

                # sleep remainder of frame
                sleep = FRAME_SEC - (time.time() - frame_start)
                if sleep > 0:
                    time.sleep(sleep)

        except Exception as exc:
            print(f"[SimEngine] sim_loop crashed: {exc}", flush=True)
            import traceback; traceback.print_exc()
        finally:
            if conn is not None:
                conn.close()
            if self._state == "running":
                self._state = "done"
            self._stop_evt.set()
            self._broadcast([{"type": "status", "state": self._state,
                              "sim_time": _fmt_ts(self._sim_ts)}])

    def _calibrate_z(self, raw: float) -> float:
        """Rescale a raw km-deviation by the fleet's recent 90th-percentile
        deviation (see __init__ note). The raw distribution is heavy-tailed
        (median-anchored scaling flagged ~27% of pings at 3σ), so we anchor
        at p90 → 1.2816 (the unit-normal p90), which puts the 3σ flag rate
        around 1% — rare enough that the anomaly feed highlights genuine
        outliers. Called under _vessels_lock from the sim thread only.
        Until enough samples exist the raw score is passed through."""
        self._z_recent.append(raw)
        if len(self._z_recent) > 8000:
            self._z_recent = self._z_recent[-4000:]
        self._z_since_calib += 1
        if self._z_since_calib >= 500 and len(self._z_recent) >= 200:
            self._z_since_calib = 0
            p90 = sorted(self._z_recent)[int(len(self._z_recent) * 0.9)]
            if p90 > 1e-9:
                self._z_scale = p90 / 1.2816
        return raw / self._z_scale if self._z_scale else raw

    def _progress_fraction(self, current_sim):
        if not self.data_start_ts or not self.data_end_ts:
            return None
        span = self.data_end_ts - self.data_start_ts
        if span <= 0:
            return None
        return max(0.0, min(1.0, (current_sim - self.data_start_ts) / span))

    def _process_pings_until(self, cur, pending_rows, current_sim, frame_events):
        """Consume pings up to current_sim, bounded by MAX_PINGS_PER_FRAME.
        If the budget is hit, the sim clock (self._sim_ts) is pulled back to
        the last processed ping so no data is skipped — the clock simply
        advances more slowly than requested under extreme speed multipliers.
        Returns True if the DB is exhausted."""
        processed = 0
        while True:
            if pending_rows:
                row = pending_rows.pop(0)
            else:
                row = cur.fetchone()
                if row is None:
                    return True

            ping_ts = _parse_ts(row["TIMESTAMP"])
            if math.isnan(ping_ts):
                continue  # skip unparseable rows rather than mis-order them
            if ping_ts > current_sim:
                pending_rows.append(row)
                return False

            self._handle_ping(row, ping_ts, frame_events)
            processed += 1
            if processed >= MAX_PINGS_PER_FRAME:
                self._sim_ts = ping_ts   # hold clock back; catch up next frame
                return False

    def _handle_ping(self, row, ping_ts, frame_events):
        mmsi = int(row["MMSI"])
        with self._vessels_lock:
            vs = self._vessels.get(mmsi)
            if vs is None:
                return

            # Previous ping (for the speed-jump detector) before we overwrite.
            prev_lat, prev_lon, prev_ts = vs.last_lat, vs.last_lon, vs.last_ts

            vs.last_lat = float(row["LAT"])
            vs.last_lon = float(row["LON"])
            vs.last_sog = float(row["SOG"] or 0)
            vs.last_cog = float(row["COG"] or 0)
            vs.last_ts  = ping_ts
            # AIS SHIP_TYPE=0 means "not available", not "this vessel is of
            # unknown type" — most vessels send a few 0s before their static
            # report arrives (7,993 vessels in this DB report both 0 and a real
            # type). Latch the first real type so the vessel doesn't flicker to
            # Unknown, which would grey out its icon and silently exempt it from
            # the ship-type-gated detectors (e.g. loitering).
            try:
                grp = cfg.ship_type_groups.get(int(float(row["SHIP_TYPE"] or 0)), 7)
            except (ValueError, TypeError):
                grp = 7
            if grp:                       # group 0 = "no type reported" — keep what we know
                vs.ship_type_group = grp

            vs.rows.append(dict(row))
            trim = SEQ_ENC + SEQ_DEC + 5
            if len(vs.rows) > trim:
                vs.rows = vs.rows[-trim:]
            vs.ping_idx += 1
            self._n_pings += 1

            # ── movement gate ──────────────────────────────────────────────
            # The model was trained only on moving vessels; a stationary ship is
            # out of distribution. If it isn't really moving, skip inference and
            # drop any stale predictions so it doesn't self-flag as it stops.
            recent = vs.rows[-HOLD_SOG_WINDOW:]
            mean_sog = sum(float(r["SOG"] or 0) for r in recent) / len(recent)
            vs.holding = len(vs.rows) >= 3 and mean_sog < MOVING_SOG_MIN
            if vs.holding:
                vs.preds = []

            # ── anomaly check against every live prediction ────────────────
            best_score = None
            best_step  = None
            per_pred_scores = []

            still_live = []
            for pr in vs.preds:
                step = pr.steps_matched
                if step < SEQ_DEC:
                    # Km-deviation scoring, matching the replay detector: the
                    # anomaly signal is how far the vessel is from the model's
                    # predicted position (mu), NOT the model's own σ z-score (its
                    # variance head is over-dispersed and unusable). Unlike replay
                    # this uses a single fleet-wide calibration (no per-ship-type
                    # threshold) — a unified operating point is enough for the sim.
                    pred_lat = float(pr.mu[step, 0]) * LAT_RNG + LAT_LO
                    pred_lon = float(pr.mu[step, 1]) * LON_RNG + LON_LO
                    raw   = _haversine_km(vs.last_lat, vs.last_lon, pred_lat, pred_lon)
                    score = round(self._calibrate_z(raw), 3)
                    per_pred_scores.append({"pred_id": pr.pred_id, "step": step, "z": score})
                    if best_score is None or score > best_score:
                        best_score = score
                        best_step  = step
                    pr.steps_matched += 1
                if pr.steps_matched < SEQ_DEC:
                    still_live.append(pr)
            vs.preds = still_live[-MAX_PREDS_PER_VESSEL:]

            flagged = best_score is not None and best_score > self.anomaly_threshold

            # ── rule-based physical detectors (independent of the model) ───
            rule_alerts = self._run_detectors(vs, prev_lat, prev_lon, prev_ts, ping_ts)

            # ── cadence-based inference trigger ────────────────────────────
            pending_pred_id = None
            due = (vs.ping_idx - vs.last_pred_ping_idx) >= self.predict_every
            if len(vs.rows) >= SEQ_ENC and due and not vs.inferring and not vs.holding:
                vs.inferring          = True
                vs.last_pred_ping_idx = vs.ping_idx
                snapshot = list(vs.rows[-SEQ_ENC:])
                pred_id  = vs.next_pred_id
                vs.next_pred_id += 1
                pending_pred_id = pred_id
                self._infer_q.put(
                    (mmsi, snapshot, pred_id, ping_ts, vs.ping_idx, self._epoch))

            n_rows     = len(vs.rows)
            ship_type  = vs.ship_type_group
            last_lat, last_lon = vs.last_lat, vs.last_lon
            last_sog, last_cog = vs.last_sog, vs.last_cog
            holding    = vs.holding

        # ── build events (outside lock) ────────────────────────────────────
        frame_events.append({
            "type": "ping", "mmsi": mmsi,
            "lat": round(last_lat, 5), "lon": round(last_lon, 5),
            "sog": round(last_sog, 1), "cog": round(last_cog, 1),
            "ship_type": ship_type, "n_pings": n_rows,
            "anomaly_score": best_score, "flagged": flagged,
            "holding": holding,
            "pred_scores": per_pred_scores,
        })

        # Signal that an async inference was just queued for this vessel so
        # the UI can show a "predicting…" state before the result arrives.
        if pending_pred_id is not None:
            frame_events.append({
                "type": "infer_pending", "mmsi": mmsi,
                "pred_id": pending_pred_id, "sim_time": _fmt_ts(ping_ts),
            })

        if flagged:
            self._n_anomalies += 1
            self._anom_ships.add(mmsi)
            self._anom_by_kind["model_deviation"] = \
                self._anom_by_kind.get("model_deviation", 0) + 1
            frame_events.append({
                "type": "anomaly", "kind": "model_deviation",
                "mmsi": mmsi, "z_score": best_score,
                "lat": round(last_lat, 5), "lon": round(last_lon, 5),
                "ship_type": ship_type, "sim_time": _fmt_ts(ping_ts),
                "sog": round(last_sog, 1), "cog": round(last_cog, 1),
                "step": best_step,
                "reason": "Off predicted path",
            })

        # ── typed physical anomalies (speed jump, on-land, idle, bad SOG) ──
        for kind, severity, reason in rule_alerts:
            self._n_anomalies += 1
            self._anom_ships.add(mmsi)
            self._anom_by_kind[kind] = self._anom_by_kind.get(kind, 0) + 1
            frame_events.append({
                "type": "anomaly", "kind": kind,
                "mmsi": mmsi, "z_score": severity, "severity": severity,
                "lat": round(last_lat, 5), "lon": round(last_lon, 5),
                "ship_type": ship_type, "sim_time": _fmt_ts(ping_ts),
                "sog": round(last_sog, 1), "cog": round(last_cog, 1),
                "reason": reason,
            })

    # ── rule-based detectors ──────────────────────────────────────────────────

    def _cell(self, lat, lon):
        """Grid cell (ilat, ilon) matching the anchorage-zone table."""
        return (int((lat - LAT_MIN) / ANCHOR_CELL_DEG),
                int((lon - LON_MIN) / ANCHOR_CELL_DEG))

    def _cooldown_ok(self, vs, kind) -> bool:
        """True if `kind` may fire for this vessel now (respecting ALERT_COOLDOWN),
        recording the firing. Prevents a persistent condition (on land, parked)
        from emitting an alert on every single ping."""
        last = vs.alerts.get(kind)
        if last is not None and (vs.ping_idx - last) < ALERT_COOLDOWN:
            return False
        vs.alerts[kind] = vs.ping_idx
        return True

    def _run_detectors(self, vs, prev_lat, prev_lon, prev_ts, ping_ts):
        """Return a list of (kind, severity, reason) for the physical anomalies
        this ping triggers. Called under _vessels_lock (mutates vs counters)."""
        alerts = []
        lat, lon, sog = vs.last_lat, vs.last_lon, vs.last_sog

        # 1) Impossible position jump — big distance in a short time.
        if prev_lat is not None and prev_ts is not None and not math.isnan(prev_ts):
            dt   = ping_ts - prev_ts
            dist = _haversine_km(prev_lat, prev_lon, lat, lon)
            if dist >= JUMP_MIN_KM and 0 < dt <= JUMP_MAX_GAP_SEC:
                implied_kn = (dist / 1.852) / (dt / 3600.0)
                if implied_kn > JUMP_MAX_KN and self._cooldown_ok(vs, "speed_jump"):
                    alerts.append(("speed_jump", round(implied_kn, 1),
                                   f"Jumped {dist:.0f} km in {dt/60:.0f} min "
                                   f"(~{implied_kn:.0f} kn — impossible)"))
            elif dist >= JUMP_MIN_KM and dt <= 0 and self._cooldown_ok(vs, "speed_jump"):
                alerts.append(("speed_jump", 99.0,
                               f"Jumped {dist:.0f} km with no time advance"))

        # 2) Impossible reported speed — the SOG field itself is unphysical.
        if sog > SOG_MAX_KN and self._cooldown_ok(vs, "impossible_sog"):
            alerts.append(("impossible_sog", round(sog, 1),
                           f"Reported speed {sog:.0f} kn exceeds any real vessel"))

        # 3) On land — vessel tracking across land while moving (skip docked ships).
        if _LAND_OK and sog > ONLAND_MIN_SOG:
            try:
                on_land = bool(_land_globe.is_land(lat, lon))
            except Exception:
                on_land = False
            vs.onland_run = vs.onland_run + 1 if on_land else 0
            if vs.onland_run >= ONLAND_MIN_RUN and self._cooldown_ok(vs, "on_land"):
                alerts.append(("on_land", 0.0,
                               f"Tracking over land at {sog:.0f} kn"))
        elif sog <= ONLAND_MIN_SOG:
            vs.onland_run = 0

        # 4) Loitering — confined to a small area longer than is normal here.
        #    Track an anchor point; while the vessel stays within LOITER_RADIUS_KM
        #    of it, dwell accumulates (real time). Reset the anchor when it leaves.
        #    The threshold is the location's learned normal dwell (a busy port ~
        #    days, open water ~none), so waiting where ships normally wait is fine.
        #    The anchor is tracked for *every* vessel (cheap, and keeps the dwell
        #    intact if a vessel's reported type is corrected mid-track), but only
        #    transiting types (LOITER_SHIP_TYPES) can raise the alert.
        if (vs.loiter_since is None
                or _haversine_km(vs.loiter_lat, vs.loiter_lon, lat, lon) > LOITER_RADIUS_KM):
            vs.loiter_lat, vs.loiter_lon, vs.loiter_since = lat, lon, ping_ts
        elif not math.isnan(ping_ts) and vs.ship_type_group in LOITER_SHIP_TYPES:
            dwell = ping_ts - vs.loiter_since
            # Ask the location questions about where the vessel *is*, not where
            # it first stopped. The anchor is set as a ship decelerates on
            # approach — typically a few hundred metres offshore, outside the
            # port cell — and never moves again while it berths inside the
            # confinement radius. Using it made docked ships report "open water".
            cell        = self._cell(lat, lon)
            norm, typed = self._dwell_norm(cell, vs.ship_type_group)
            thresh = max(LOITER_MIN_SEC, norm * LOITER_MARGIN) if norm else LOITER_OPEN_SEC
            # A vessel berthed in a known port/anchorage, or alongside a quay, is
            # cleared to be there — sitting still carries no signal. Checked only
            # once the dwell threshold is passed, and *before* _cooldown_ok so a
            # docked vessel never burns its cooldown slot.
            if (dwell >= thresh
                    and not self._is_docked(cell, lat, lon)
                    and self._cooldown_ok(vs, "loitering")):
                hrs = dwell / 3600.0
                if norm:
                    who  = GROUP_NAMES[vs.ship_type_group] if typed else "ships"
                    note = f"normal here for {who} ~{norm/3600:.0f} h"
                else:
                    note = "open water — no normal stop"
                alerts.append(("loitering", round(hrs, 1),
                               f"Loitering {hrs:.1f} h in one spot ({note})"))

        return alerts

    def _is_docked(self, cell, lat, lon) -> bool:
        """True if the vessel is somewhere it is cleared to sit: inside a known
        port/anchorage cell, or alongside land (a quay/berth).

        A ship at a berth usually reports a position the land mask calls *water*,
        because at ~1 km resolution the quay's cell is majority-water. So we also
        treat land in the immediate neighbourhood as being alongside. The cost is
        that genuine loitering within ~BERTH_LAND_DEG of a coast is not flagged."""
        if cell in self._anchor_cells:
            return True
        if not _LAND_OK:
            return False
        try:
            if _land_globe.is_land(lat, lon):
                return True
            d = BERTH_LAND_DEG
            for dlat in (-d, 0.0, d):
                for dlon in (-d, 0.0, d):
                    if (dlat or dlon) and _land_globe.is_land(lat + dlat, lon + dlon):
                        return True
        except Exception:
            return False
        return False

    def _dwell_norm(self, cell, group):
        """Normal dwell (seconds) for this location and vessel type.

        Prefers the per-type norm; falls back to the cell's all-type norm when
        that type has too few local visits. Returns (norm, is_type_specific);
        norm is None for a cell with no stationary history at all (open water),
        where any sustained stop is suspicious regardless of type."""
        ilat, ilon = cell
        norm = self._cell_dwell.get((ilat, ilon, group))
        if norm is not None:
            return norm, True
        return self._cell_dwell.get((ilat, ilon, _DWELL_ALL)), False

    # ── batched autoregressive inference ──────────────────────────────────────

    @torch.no_grad()
    def _batch_ar_predict(self, enc_arrays):
        """Batched AR decode for SEQ_DEC steps.

        enc_arrays : list of (SEQ_ENC, N_ENC) numpy float32 arrays.
        Returns    : list of (mu, sigma) pairs, each (SEQ_DEC, N_DEC) numpy array.
        Handles cfg.predict_deltas — accumulates dLAT/dLON onto the previous
        absolute position at each step before feeding back into the decoder.
        """
        # Snapshot the model reference: set_model() may swap self._model while
        # this batch is decoding. The whole batch then runs on one consistent
        # model, and the epoch bump makes the sim discard the stale result.
        model = self._model

        B   = len(enc_arrays)
        src = torch.from_numpy(np.stack(enc_arrays)).to(self.device)  # (B, SEQ_ENC, N_ENC)
        last_dt     = src[:, -1:, N_DEC:N_DEC_IN]    # (B, 1, 1) — reuse last observed DT
        dec_input   = src[:, -1:, :N_DEC_IN]         # (B, 1, N_DEC_IN)
        prev_motion = src[:, -1:, :N_DEC].float()    # (B, 1, N_DEC) last absolute position

        mu_acc  = []
        sig_acc = []
        # Delta mode: the variance head parameterises per-step dLAT/dLON, so
        # absolute-position uncertainty is the accumulated (summed) variance
        # of the deltas, converted from delta-normalised units (range 4°) to
        # absolute-normalised units (LAT range 16°, LON range 25°). Without
        # this conversion z-scores were computed against a sigma in the wrong
        # units and came out ~100× too small — anomalies never fired.
        var_lat_deg2 = torch.zeros((B, 1, 1), device=self.device)
        var_lon_deg2 = torch.zeros((B, 1, 1), device=self.device)

        for _ in range(SEQ_DEC):
            mu_raw, log_var = model(src, dec_input)
            mu_last  = mu_raw[:, -1:, :].float()                   # (B, 1, N_DEC)
            std_last = (log_var[:, -1:, :].float() * 0.5).exp()

            if _PREDICT_DELTAS:
                # Accumulate dLAT/dLON offset onto previous absolute position
                prev_lat = prev_motion[:, :, 0:1] * LAT_RNG + LAT_LO
                prev_lon = prev_motion[:, :, 1:2] * LON_RNG + LON_LO
                dlat = mu_last[:, :, 0:1] * _DLAT_RNG + _DLAT_LO
                dlon = mu_last[:, :, 1:2] * _DLON_RNG + _DLON_LO
                abs_lat = ((prev_lat + dlat - LAT_LO) / LAT_RNG).clamp(0.0, 1.0)
                abs_lon = ((prev_lon + dlon - LON_LO) / LON_RNG).clamp(0.0, 1.0)
                sog_cog = mu_last[:, :, 2:].clamp(0.0, 1.0)
                next_motion = torch.cat([abs_lat, abs_lon, sog_cog], dim=-1)
            else:
                next_motion = mu_last.clamp(0.0, 1.0)

            # Renormalise COG to the unit circle (vectorised across batch)
            sin_raw = next_motion[:, :, 3] * 2.0 - 1.0
            cos_raw = next_motion[:, :, 4] * 2.0 - 1.0
            mag     = (sin_raw.pow(2) + cos_raw.pow(2)).sqrt().clamp(min=1e-8)
            next_motion = next_motion.clone()
            next_motion[:, :, 3] = (sin_raw / mag + 1.0) * 0.5
            next_motion[:, :, 4] = (cos_raw / mag + 1.0) * 0.5

            prev_motion = next_motion
            next_step   = torch.cat([next_motion, last_dt], dim=-1)  # (B, 1, N_DEC_IN)

            if _PREDICT_DELTAS:
                var_lat_deg2 = var_lat_deg2 + (std_last[:, :, 0:1] * _DLAT_RNG) ** 2
                var_lon_deg2 = var_lon_deg2 + (std_last[:, :, 1:2] * _DLON_RNG) ** 2
                std_abs = torch.cat([
                    var_lat_deg2.sqrt() / LAT_RNG,
                    var_lon_deg2.sqrt() / LON_RNG,
                    std_last[:, :, 2:],
                ], dim=-1)
            else:
                std_abs = std_last

            mu_acc.append(next_motion.cpu().numpy())    # (B, 1, N_DEC)
            sig_acc.append(std_abs.cpu().numpy())
            dec_input = torch.cat([dec_input, next_step], dim=1)

        mu_batch  = np.concatenate(mu_acc,  axis=1)   # (B, SEQ_DEC, N_DEC)
        sig_batch = np.concatenate(sig_acc, axis=1)
        return [(mu_batch[i], sig_batch[i]) for i in range(B)]

    # ── inference loop ─────────────────────────────────────────────────────────

    def _infer_loop(self):
        while not self._stop_evt.is_set():
            # ── wait for first item ────────────────────────────────────────
            try:
                first = self._infer_q.get(timeout=0.5)
            except queue.Empty:
                continue

            # ── collect more items up to INFER_BATCH_SIZE ─────────────────
            batch   = [first]
            deadline = time.time() + INFER_WAIT_SEC
            while len(batch) < INFER_BATCH_SIZE and time.time() < deadline:
                try:
                    batch.append(self._infer_q.get_nowait())
                except queue.Empty:
                    break

            # ── convert DB rows → normalised encoder arrays ────────────────
            valid        = []   # [(item, enc_array), ...]
            skip_mmsis   = []
            for item in batch:
                mmsi, rows_snapshot, pred_id, issued_ts, issued_idx, epoch = item
                if epoch != self._epoch:
                    continue    # queued before a seek/reset — vessel state is gone
                try:
                    row_objs   = [_DictRow(r) for r in rows_snapshot]
                    track_norm = rows_to_array(row_objs)
                    if len(track_norm) < SEQ_ENC:
                        skip_mmsis.append(mmsi)
                        continue
                    valid.append((item, track_norm[-SEQ_ENC:]))
                except Exception as exc:
                    print(f"[SimEngine] rows_to_array error MMSI {mmsi}: {exc}", flush=True)
                    skip_mmsis.append(mmsi)

            with self._vessels_lock:
                for mmsi in skip_mmsis:
                    if mmsi in self._vessels:
                        self._vessels[mmsi].inferring = False

            if not valid:
                continue

            # ── single batched forward pass ────────────────────────────────
            try:
                items, enc_arrays = zip(*valid)
                results = self._batch_ar_predict(list(enc_arrays))
            except Exception as exc:
                print(f"[SimEngine] batch inference error: {exc}", flush=True)
                import traceback; traceback.print_exc()
                with self._vessels_lock:
                    for item, _ in valid:
                        mmsi = item[0]
                        if mmsi in self._vessels:
                            self._vessels[mmsi].inferring = False
                continue

            # ── broadcast one prediction event per vessel ──────────────────
            for item, (mu_pred, sig_pred) in zip(items, results):
                mmsi, _, pred_id, issued_ts, issued_idx, epoch = item
                try:
                    if epoch != self._epoch:
                        continue   # completed after a seek/reset — discard
                    dec_raw = denormalise_dec(mu_pred)
                    pr = Prediction(pred_id, issued_ts, issued_idx,
                                    mu_pred, sig_pred, dec_raw)
                    with self._vessels_lock:
                        vs = self._vessels.get(mmsi)
                        if vs is not None:
                            vs.preds.append(pr)
                            vs.preds = vs.preds[-MAX_PREDS_PER_VESSEL:]
                            vs.inferring = False
                            self._n_predictions += 1

                    self._broadcast([{
                        "type": "prediction", "mmsi": mmsi, "pred_id": pred_id,
                        "issued_ts": _fmt_ts(issued_ts),
                        "issued_unix": issued_ts,
                        "issued_idx": issued_idx,
                        "pred": [[round(float(dec_raw[i, 0]), 5),
                                  round(float(dec_raw[i, 1]), 5)]
                                 for i in range(SEQ_DEC)],
                        "sogs": [round(float(dec_raw[i, 2]), 2) for i in range(SEQ_DEC)],
                    }])
                except Exception as exc:
                    print(f"[SimEngine] post-infer error MMSI {mmsi}: {exc}", flush=True)
                    with self._vessels_lock:
                        if mmsi in self._vessels:
                            self._vessels[mmsi].inferring = False
