"""
utils.py
--------
Shared helper functions used across dataset, training, and prediction.
"""

import os
import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Distance metric
# ─────────────────────────────────────────────────────────────────────────────

def haversine_tensor(pred, target):
    """
    Haversine distance for batched torch tensors.

    pred, target : (..., 2)  where last dim is [lat, lon] in degrees.
    Returns a tensor of the same leading shape containing distances in km.
    """
    R = 6371.0
    pred   = torch.deg2rad(pred)
    target = torch.deg2rad(target)

    dlat = target[..., 0] - pred[..., 0]
    dlon = target[..., 1] - pred[..., 1]

    a = (
        torch.sin(dlat / 2) ** 2
        + torch.cos(pred[..., 0]) * torch.cos(target[..., 0]) * torch.sin(dlon / 2) ** 2
    )
    return 2 * R * torch.asin(torch.sqrt(a.clamp(0, 1)))


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Water mask  (OSM land polygons + OSM inland waterways)
# ─────────────────────────────────────────────────────────────────────────────

# Data lives in the sibling DataHandling/ folder; shared land reference data
# (water mask + OSM land polygons) sits in its Land/ subfolder.
_DATA              = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DataHandling"))
WATER_MASK_CACHE   = os.path.join(_DATA, "Land", "water_mask.npz")
WATER_MASK_RES     = 0.005  # degrees per cell (~500 m at these latitudes)
_OSM_LAND_DIR      = os.path.join(_DATA, "Land", "osm_land")
_OSM_LAND_SHP      = os.path.join(_OSM_LAND_DIR,
                         "simplified-land-polygons-complete-3857",
                         "simplified_land_polygons.shp")
# Simplified version (~24 MB) is Mercator; geopandas reprojects to WGS84 on load.
# Full WGS84 version is 918 MB — not practical.
_OSM_LAND_URL      = ("https://osmdata.openstreetmap.de/download/"
                      "simplified-land-polygons-complete-3857.zip")


def _get_osm_land_shp() -> str:
    """Download and unzip the OSM land-polygon shapefile if not already cached.

    Same vector data the OSM tile servers render from. One-time ~60 MB download.
    """
    if os.path.exists(_OSM_LAND_SHP):
        return _OSM_LAND_SHP
    import urllib.request, zipfile
    os.makedirs(_OSM_LAND_DIR, exist_ok=True)
    zip_path = os.path.join(_OSM_LAND_DIR, "land_polygons.zip")
    print(f"  [water mask] Downloading OSM land polygons from osmdata.openstreetmap.de (~60 MB) ...",
          flush=True)
    urllib.request.urlretrieve(_OSM_LAND_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(_OSM_LAND_DIR)
    os.remove(zip_path)
    return _OSM_LAND_SHP


def _burn_polygons(geoms, xs: np.ndarray, ys: np.ndarray, flat: np.ndarray, value: bool) -> None:
    """Burn shapely geometries into flat bool array using vectorised contains.

    Uses bounding-box pre-filter so only relevant points are tested per polygon —
    ~10-50× faster than testing all points for every polygon.
    """
    from shapely.vectorized import contains as sv_contains
    for geom in geoms:
        if geom is None or geom.is_empty:
            continue
        parts = list(geom.geoms) if geom.geom_type.startswith("Multi") else [geom]
        for poly in parts:
            if poly.is_empty:
                continue
            minx, miny, maxx, maxy = poly.bounds
            mask = (xs >= minx) & (xs <= maxx) & (ys >= miny) & (ys <= maxy)
            if not mask.any():
                continue
            hits = np.zeros(len(xs), dtype=bool)
            hits[mask] = sv_contains(poly, xs[mask], ys[mask])
            flat[hits] = value


def _apply_osmnx_waterways(
    raster: np.ndarray,
    lat_lo: float, lat_hi: float,
    lon_lo: float, lon_hi: float,
    lat_lo_raster: float, lon_lo_raster: float,
    res: float,
) -> np.ndarray:
    """Override land cells with OSM waterway features (canals, rivers, docks, harbours).

    Line geometries (rivers, canals) are buffered by 2 cells so narrow channels
    are fully opened even when the centreline is only 1 cell wide.  Polygon
    features (docks, basins, water areas) are burned directly.

    Falls back gracefully if osmnx is not installed or the query fails.
    """
    try:
        import osmnx as ox
    except ImportError:
        print("  [water mask]   2/3  osmnx not installed — skipping waterway step "
              "(pip install osmnx to enable)", flush=True)
        return raster

    H, W = raster.shape
    lats = np.arange(lat_lo_raster, lat_lo_raster + H * res, res)[:H]
    lons = np.arange(lon_lo_raster, lon_lo_raster + W * res, res)[:W]
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    xs = lon_grid.ravel()
    ys = lat_grid.ravel()

    tags = {
        "waterway": ["canal", "river", "dock", "fairway", "tidal_channel",
                     "navigable_channel"],
        "natural":  ["water", "bay"],
        "harbour":  True,
        "landuse":  ["basin", "harbour"],
    }

    print("  [water mask]   2/3  OSM waterways via osmnx ...", flush=True)
    try:
        # osmnx 2.x bbox = (left, bottom, right, top) = (lon_lo, lat_lo, lon_hi, lat_hi)
        gdf = ox.features_from_bbox(
            bbox=(lon_lo, lat_lo, lon_hi, lat_hi), tags=tags,
        )
    except Exception as exc:
        print(f"  [water mask]   osmnx query failed ({exc}) — skipping", flush=True)
        return raster

    if gdf is None or gdf.empty:
        print("  [water mask]   no waterway features returned", flush=True)
        return raster

    gdf = gdf.to_crs("EPSG:4326")
    print(f"  [water mask]         {len(gdf):,} waterway features", flush=True)

    buffer_deg = res * 2   # 2-cell buffer for line features (~1 km at 0.005°)
    water_override = np.zeros(len(xs), dtype=bool)

    from shapely.vectorized import contains as sv_contains

    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type in ("LineString", "MultiLineString", "Point", "MultiPoint"):
            geom = geom.buffer(buffer_deg)
        parts = list(geom.geoms) if geom.geom_type.startswith("Multi") else [geom]
        for poly in parts:
            if poly.is_empty:
                continue
            minx, miny, maxx, maxy = poly.bounds
            mask = (xs >= minx) & (xs <= maxx) & (ys >= miny) & (ys <= maxy)
            if not mask.any():
                continue
            hits = np.zeros(len(xs), dtype=bool)
            hits[mask] = sv_contains(poly, xs[mask], ys[mask])
            water_override |= hits

    water_override = water_override.reshape(H, W)
    corrected = water_override & ~raster
    raster    = raster | water_override
    print(f"  [water mask]   waterway correction: {corrected.sum():,} land cells "
          f"overridden as water  ({corrected.mean()*100:.2f}% of raster)", flush=True)
    return raster


def build_water_raster(
    lat_lo: float, lat_hi: float, lon_lo: float, lon_hi: float,
    cache_path: str = WATER_MASK_CACHE,
    res: float = WATER_MASK_RES,
) -> tuple:
    """Build, save, and return a binary water raster for the given region.

    Built in three passes: (1) OSM land polygons set the coastline, (2) OSM
    waterways (canals, rivers, docks) are burned back in as water, (3) any cell a
    ship track visits is forced to water. Returns (raster, lat_lo, lon_lo) with
    raster (H, W) bool, True = water. Requires: pip install geopandas osmnx
    """
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise RuntimeError(
            "Water mask requires: pip install geopandas\n" + str(exc)
        )

    lats = np.arange(lat_lo, lat_hi, res)
    lons = np.arange(lon_lo, lon_hi, res)
    H, W = len(lats), len(lons)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    xs = lon_grid.ravel()
    ys = lat_grid.ravel()

    print(f"  [water mask] Building {H}×{W} raster at {res}° (~{res*111:.0f} km/cell) "
          f"— cached after first run ...", flush=True)

    flat = np.ones(len(xs), dtype=bool)   # start all-water; burn land below

    # ── Step 1: OSM land polygons ─────────────────────────────────────────────
    print("  [water mask]   1/3  OSM land polygons ...", flush=True)
    shp = _get_osm_land_shp()
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x_lo, y_lo = t.transform(lon_lo, lat_lo)
    x_hi, y_hi = t.transform(lon_hi, lat_hi)
    land_gdf = gpd.read_file(shp, bbox=(x_lo, y_lo, x_hi, y_hi)).to_crs("EPSG:4326")
    print(f"  [water mask]         {len(land_gdf):,} polygons in region", flush=True)
    _burn_polygons(land_gdf.geometry.values, xs, ys, flat, value=False)

    raster = flat.reshape(H, W)

    # ── Step 2: OSM waterways (canals, rivers, docks, harbours) ──────────────
    raster = _apply_osmnx_waterways(
        raster, lat_lo, lat_hi, lon_lo, lon_hi, lat_lo, lon_lo, res,
    )

    # ── Step 3: AIS track correction (train + val + test windows) ────────────
    raster = _apply_track_corrections(raster, lat_lo, lon_lo, res)

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.savez(cache_path, raster=raster, lat_lo=lat_lo, lon_lo=lon_lo)
    print(f"  [water mask] Saved to {cache_path}")
    return raster, lat_lo, lon_lo


def _apply_track_corrections(
    raster: np.ndarray,
    lat_lo: float,
    lon_lo: float,
    res: float,
    meta_path:   str = os.path.join(_DATA, "training", "dataset_meta.json"),
    buffer_cells: int = 3,
) -> np.ndarray:
    """Override land cells that lie on recorded ship tracks with water.

    Uses all three window splits (train, val, test) so every observed ship
    position — regardless of split — contributes to the correction.  A
    3-cell dilation ensures narrow channels are fully opened even when the
    track runs along the centreline.

    Safe to call even if the data files do not yet exist.
    """
    if not os.path.exists(meta_path):
        print("  [water mask]   track correction: meta not found, skipped")
        return raster

    import json
    from scipy.ndimage import binary_dilation

    meta  = json.load(open(meta_path))
    W_win = meta["window_len"]
    F     = meta["n_features"]
    nb    = meta["norm_bounds"]
    lat_lo_norm, lat_hi_norm = nb["LAT"]
    lon_lo_norm, lon_hi_norm = nb["LON"]
    H, W_rast = raster.shape

    splits = [
        (os.path.join(_DATA, "training", "train_windows.bin"), meta.get("n_train", 0)),
        (os.path.join(_DATA, "training", "val_windows.bin"),   meta.get("n_val",   0)),
        (os.path.join(_DATA, "training", "test_windows.bin"),  meta.get("n_test",  0)),
    ]

    visited = np.zeros((H, W_rast), dtype=bool)
    total_pings = 0

    for path, N in splits:
        if N == 0 or not os.path.exists(path):
            continue
        data     = np.memmap(path, dtype="float32", mode="r", shape=(N, W_win, F))
        lat_vals = data[:, :, 0] * (lat_hi_norm - lat_lo_norm) + lat_lo_norm
        lon_vals = data[:, :, 1] * (lon_hi_norm - lon_lo_norm) + lon_lo_norm
        row_idx  = ((lat_vals - lat_lo) / res).astype(np.int32).clip(0, H - 1)
        col_idx  = ((lon_vals - lon_lo) / res).astype(np.int32).clip(0, W_rast - 1)
        visited[row_idx.ravel(), col_idx.ravel()] = True
        total_pings += N * W_win
        print(f"  [water mask]         {path}: {N:,} windows ({N*W_win:,} pings)",
              flush=True)

    if total_pings == 0:
        print("  [water mask]   track correction: no window files found, skipped")
        return raster

    if buffer_cells > 0:
        struct  = np.ones((2 * buffer_cells + 1, 2 * buffer_cells + 1), dtype=bool)
        visited = binary_dilation(visited, structure=struct)

    corrected = visited & ~raster
    raster    = raster | visited
    print(f"  [water mask]   3/3  track correction: {corrected.sum():,} land cells "
          f"overridden as water  ({corrected.mean()*100:.2f}% of raster)", flush=True)
    return raster


def load_water_mask(
    lat_lo: float, lat_hi: float, lon_lo: float, lon_hi: float,
    cache_path: str = WATER_MASK_CACHE,
) -> tuple:
    """Load cached water raster or build it on first call.

    Returns (raster, lat_lo, lon_lo) where raster is (H, W) bool, True = water.
    """
    if os.path.exists(cache_path):
        d      = np.load(cache_path)
        raster = d["raster"].astype(bool)
        lat_lo = float(d["lat_lo"])
        lon_lo = float(d["lon_lo"])
        H, W   = raster.shape
        print(f"  [water mask] Loaded {H}×{W} raster from {cache_path}")
        return raster, lat_lo, lon_lo
    return build_water_raster(lat_lo, lat_hi, lon_lo, lon_hi, cache_path)
