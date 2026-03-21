# -*- coding: utf-8 -*-
"""
TESS Phase 1: Download labeled light curves (planets vs. field stars)

RETRAIN VERSION — key changes vs original:
  - want=1500 field stars (was 300) for stable CNN training
  - probe_limit=8000 (was 5000) to find 1500 downloadable targets
  - max_samples=1500 for nonplanets (was 300)
  - 10 sky patches instead of 5 for more diversity
  - Checkpoint/resume: saves progress to phase1_field_checkpoint.pkl
    so a crash or interruption doesn't lose hours of downloads
  - Everything else identical to original
"""

import time
import random
import re
import pickle
import os
from io import StringIO
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import astropy.units as u
from astropy.coordinates import SkyCoord

import lightkurve as lk
from astroquery.mast import Observations


# =============================================================================
# Reproducibility
# =============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# =============================================================================
# Networking: retries + timeouts
# =============================================================================
def make_retry_session(
    total: int = 8,
    backoff_factor: float = 1.5,
    user_agent: str = "tess-downloader/1.0",
) -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=total,
        connect=total,
        read=total,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": user_agent})
    return s


Observations._portal_api_connection._session = make_retry_session()
Observations._portal_api_connection.TIMEOUT = 300


def polite_sleep(lo: float = 1.0, hi: float = 2.0) -> None:
    time.sleep(random.uniform(lo, hi))


def normalize_target_name(target: str) -> str:
    t = str(target).strip()
    if t.upper().startswith("TOI-"):
        return "TOI " + t[4:]
    return t


# =============================================================================
# Confirmed TOIs (planets)
# =============================================================================
def get_confirmed_toi_targets() -> List[str]:
    """Return confirmed TOI identifiers (tfopwg_disp == 'CP')."""
    base = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"
    params = {"table": "toi", "select": "tidstr,tfopwg_disp", "format": "csv"}

    r = requests.get(base, params=params, timeout=60)
    r.raise_for_status()

    df = pd.read_csv(StringIO(r.text))
    df = df[df["tfopwg_disp"].astype(str).str.upper().str.strip() == "CP"].copy()

    targets = pd.unique(df["tidstr"].astype(str).str.strip()).tolist()
    targets = [normalize_target_name(t) for t in targets if t and t.lower() != "nan"]

    print(f"✓ Loaded {len(targets)} confirmed TOI targets")
    print("Example targets:", targets[:5])
    return targets


# =============================================================================
# Negatives: pool of TIC targets from sky patches
# =============================================================================
def _extract_tics_from_search(search) -> List[int]:
    tbl = getattr(search, "table", None)
    if tbl is None:
        return []

    col_candidates = [c for c in ("target_name", "target", "objectname",
                                   "obs_id", "productFilename") if c in tbl.colnames]
    rows = ([str(v) for v in tbl[col_candidates[0]]]
            if col_candidates else [str(row) for row in tbl])

    tic_ids = set()
    for s in rows:
        if not s or s.lower() == "nan":
            continue
        m = re.search(r"\bTIC\s*([0-9]+)\b", s, flags=re.IGNORECASE)
        if m:
            tic_ids.add(int(m.group(1)))
            continue
        s2 = s.strip()
        if s2.isdigit():
            tic_ids.add(int(s2))

    return list(tic_ids)


def get_real_field_stars_pool(
    total_needed: int = 10000,    # ← increased from 5000
    radius_deg: float = 0.25,
    limit_per_patch: int = 2000,
) -> List[str]:
    """
    Return a large pool of TIC targets from multiple sky patches.
    Uses 10 patches (was 5) to get enough diversity for 1500 backgrounds.
    """
    # 10 patches spread across the sky for maximum diversity
    patches = [
        # Original 5
        (100.0,  20.0),
        (250.0, -20.0),
        (180.0,   0.0),
        (300.0,  30.0),
        ( 50.0, -10.0),
        # 5 new patches
        ( 20.0,  45.0),
        (150.0, -45.0),
        (200.0,  60.0),
        (330.0, -30.0),
        ( 80.0,  10.0),
    ]

    all_tics = set()

    for ra_deg, dec_deg in patches:
        coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
        search = lk.search_lightcurve(
            coord, radius=radius_deg * u.deg, mission="TESS", limit=limit_per_patch
        )
        polite_sleep()

        for tic in _extract_tics_from_search(search):
            all_tics.add(tic)

        print(f"Field-star pool size now: {len(all_tics)} "
              f"(after RA={ra_deg}, Dec={dec_deg})")

        if len(all_tics) >= total_needed:
            break

    tic_ids = list(all_tics)
    np.random.shuffle(tic_ids)
    return [f"TIC {tic}" for tic in tic_ids]


# =============================================================================
# Download + preprocess one light curve
# =============================================================================
def download_tess_lc(
    target: str,
    target_length: int = 2001,
    min_points: int = 1000,
    prefer_spoc: bool = False,
) -> np.ndarray | None:
    target = normalize_target_name(target)

    try:
        search = lk.search_lightcurve(target, mission="TESS", limit=50)
        if len(search) == 0:
            return None

        idx = 0
        if prefer_spoc:
            for i in range(len(search)):
                if str(search[i].author).upper() in ("SPOC", "TESS-SPOC"):
                    idx = i
                    break

        lc = search[idx].download()
        polite_sleep()
        if lc is None:
            return None

        lc = lc.remove_nans()
        if len(lc) < min_points:
            return None

        flux = lc.flux.value
        std = np.std(flux)
        if std == 0 or np.isnan(std):
            return None

        flux = (flux - np.median(flux)) / std
        resample_idx = np.linspace(0, len(flux) - 1, target_length).astype(int)
        return flux[resample_idx]

    except Exception:
        return None


# =============================================================================
# Filter to downloadable targets — WITH checkpoint/resume
#
# Why checkpoint?  With want=1500 this function runs for several hours.
# If Spyder crashes, your kernel restarts, or your laptop sleeps, you'd
# normally lose all progress.  The checkpoint saves every 25 successful
# downloads to phase1_field_checkpoint.pkl so you can resume where you
# left off just by running Phase_1.py again.
# =============================================================================
FIELD_CHECKPOINT = "phase1_field_checkpoint.pkl"

def filter_downloadable_targets(
    targets: List[str],
    want: int = 1500,             # ← increased from 300
    probe_limit: int = 8000,      # ← increased from 5000
    target_length: int = 2001,
    min_points_probe: int = 800,
) -> List[str]:
    """
    Return the first `want` targets that successfully download a usable LC.
    Saves progress every 25 downloads so a crash doesn't lose your work.

    To start completely fresh: delete phase1_field_checkpoint.pkl
    To resume after a crash:   just run Phase_1.py again
    """
    # ── Load checkpoint if it exists ────────────────────────────────────────
    good: List[str] = []
    tried_targets: set = set()

    if os.path.exists(FIELD_CHECKPOINT):
        try:
            with open(FIELD_CHECKPOINT, "rb") as f:
                ckpt = pickle.load(f)
            good = ckpt.get("good", [])
            tried_targets = set(ckpt.get("tried", []))
            print(f"✓ Resuming from checkpoint: {len(good)} already downloaded, "
                  f"{len(tried_targets)} already tried")
        except Exception as e:
            print(f"Warning: could not load checkpoint ({e}), starting fresh")
            good = []
            tried_targets = set()

    if len(good) >= want:
        print(f"✓ Already have {len(good)} targets from checkpoint. Done.")
        return good[:want]

    # ── Continue downloading ─────────────────────────────────────────────────
    tried = len(tried_targets)

    for t in targets:
        if len(good) >= want or tried >= probe_limit:
            break

        if t in tried_targets:
            continue  # already attempted in a previous run

        tried += 1
        tried_targets.add(t)

        arr = download_tess_lc(
            t,
            target_length=target_length,
            min_points=min_points_probe,
            prefer_spoc=False,
        )
        if arr is not None:
            good.append(t)

        # Progress report + checkpoint save every 25 probes
        if tried % 25 == 0:
            print(f"Probed {tried}, kept {len(good)} / {want}")
            _save_field_checkpoint(good, list(tried_targets))
            polite_sleep(0.2, 0.6)

    # Final checkpoint save
    _save_field_checkpoint(good, list(tried_targets))
    print(f"✓ Filtered downloadable targets: {len(good)}/{want} (probed {tried})")
    return good


def _save_field_checkpoint(good: List[str], tried: List[str]) -> None:
    """Save current progress so we can resume after a crash."""
    with open(FIELD_CHECKPOINT, "wb") as f:
        pickle.dump({"good": good, "tried": tried}, f)


# =============================================================================
# Build dataset
# =============================================================================
def build_dataset(
    targets: List[str],
    label: int,
    max_samples: int,
    target_length: int = 2001,
    min_points: int = 1000,
    prefer_spoc: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X, y, failed = [], [], []

    n = min(len(targets), max_samples)
    for i in range(n):
        if i % 25 == 0:
            print(f"label={label}: {i}/{n}")

        t = targets[i]
        arr = download_tess_lc(
            t,
            target_length=target_length,
            min_points=min_points,
            prefer_spoc=prefer_spoc,
        )
        if arr is None:
            failed.append(t)
            continue

        X.append(arr)
        y.append(label)

    X = np.array(X)
    y = np.array(y)
    print(f"✓ label={label} downloaded {len(X)} / {n} (failed {len(failed)})")
    return X, y, failed


# =============================================================================
# Save helpers
# =============================================================================
def save_phase1_pickle(
    filename: str,
    X_p: np.ndarray,
    y_p: np.ndarray,
    X_n: np.ndarray,
    y_n: np.ndarray,
    failed_planets: List[str],
    failed_nonplanets: List[str],
    meta: Dict[str, Any],
) -> None:
    payload = {
        "X_planets": X_p,
        "y_planets": y_p,
        "X_nonplanets": X_n,
        "y_nonplanets": y_n,
        "failed_planets": failed_planets,
        "failed_nonplanets": failed_nonplanets,
        "meta": meta,
    }
    with open(filename, "wb") as f:
        pickle.dump(payload, f)
    print(f"✓ Saved: {filename}")


def save_combined_pickle(
    filename: str, X: np.ndarray, y: np.ndarray, meta: Dict[str, Any]
) -> None:
    payload = {"X": X, "y": y, "meta": meta}
    with open(filename, "wb") as f:
        pickle.dump(payload, f)
    print(f"✓ Saved: {filename}")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":

    # ── Planets (confirmed TOIs) ─────────────────────────────────────────────
    planets = get_confirmed_toi_targets()

    # ── Field stars: large pool -> filter to 1500 that actually download ─────
    # NOTE: If this was interrupted before, it resumes automatically from
    # phase1_field_checkpoint.pkl — no need to change anything.
    # To start completely fresh, delete phase1_field_checkpoint.pkl first.
    field_candidates = get_real_field_stars_pool(total_needed=10000)
    field = filter_downloadable_targets(
        field_candidates,
        want=1500,          # ← KEY CHANGE: was 300
        probe_limit=8000,   # ← KEY CHANGE: was 5000
    )

    # ── Build datasets ───────────────────────────────────────────────────────
    X_p, y_p, f_p = build_dataset(
        planets, label=1, max_samples=300, min_points=1000, prefer_spoc=False
    )
    X_n, y_n, f_n = build_dataset(
        field, label=0, max_samples=1500, min_points=800, prefer_spoc=False
        #                ↑ KEY CHANGE: was 300
    )

    # ── Combine ──────────────────────────────────────────────────────────────
    if len(X_p) and len(X_n):
        X = np.vstack([X_p, X_n])
        y = np.hstack([y_p, y_n])
    else:
        X, y = None, None

    print(f"\nPlanets:     {len(X_p)}")
    print(f"Non-planets: {len(X_n)}")
    print(f"Failed planets:     {len(f_p)}  Examples: {f_p[:3]}")
    print(f"Failed non-planets: {len(f_n)}  Examples: {f_n[:3]}")

    # ── Save ─────────────────────────────────────────────────────────────────
    meta = {
        "seed": SEED,
        "target_length": 2001,
        "planet_max_samples": 300,
        "nonplanet_max_samples": 1500,     # ← updated
        "planet_min_points": 1000,
        "nonplanet_min_points": 800,
        "prefer_spoc_planets": False,
        "prefer_spoc_nonplanets": False,
        "field_pool_total_needed": 10000,  # ← updated
        "n_patches": 10,                   # ← updated
        "retrain_version": True,
    }

    save_phase1_pickle(
        "tess_phase1_dataset.pkl",
        X_p, y_p, X_n, y_n,
        failed_planets=f_p,
        failed_nonplanets=f_n,
        meta=meta,
    )

    if X is not None and y is not None:
        save_combined_pickle("tess_phase1_dataset_combined.pkl", X, y, meta)

    # ── Quick sanity plot ────────────────────────────────────────────────────
    if len(X_n) > 0:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 3))
        axes[0].plot(X_n[0], lw=0.8)
        axes[0].set_title("Field star background (negative example)")
        axes[0].set_xlabel("Time index")
        axes[0].set_ylabel("Normalized flux")

        if len(X_p) > 0:
            axes[1].plot(X_p[0], lw=0.8, color="orange")
            axes[1].set_title("Confirmed TOI (positive example)")
            axes[1].set_xlabel("Time index")
            axes[1].set_ylabel("Normalized flux")

        plt.tight_layout()
        plt.savefig("phase1_sample_lightcurves.png", dpi=100)
        plt.show()
        print("✓ Saved sample plot: phase1_sample_lightcurves.png")
    else:
        print("No light curves downloaded yet.")
