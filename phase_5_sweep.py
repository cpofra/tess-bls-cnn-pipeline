# -*- coding: utf-8 -*-
"""
TESS Phase 5: Long-run planet search
- Streams sector-by-sector using MAST CAOM via astroquery (stable sector harvesting)
- download LC -> detrend -> BLS top-K -> window -> CNN score -> checkpoint CSV
- Writes run summary CSV for this run only

REVISION A (Reviewer item #1 scaffolding):
- Adds confirmed-planet benchmark mode using NASA Exoplanet Archive (astroquery 0.4.11-safe query_criteria)
- Writes tess_benchmark_confirmed.csv with scores + period-match diagnostics
"""

import os
import re
import time
import json
import random
import warnings
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares

# Quiet noisy logs (MUST be set BEFORE importing lightkurve)
warnings.filterwarnings(
    "ignore",
    message=r".*tpfmodel submodule is not available without oktopus installed.*"
)
warnings.filterwarnings("ignore", category=UserWarning, module=r"lightkurve(\..*)?$")

# Hide TF INFO spam (oneDNN etc.)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import lightkurve as lk  # noqa: E402
from tensorflow import keras  # noqa: E402
import multiprocessing as mp  # noqa: E402

# Morphology pre-filters (morphology_filters.py must be in the same directory)
try:
    from morphology_filters import compute_morph_flags, morph_flags_to_dict
    _MORPH_FILTERS_AVAILABLE = True
except ImportError:
    _MORPH_FILTERS_AVAILABLE = False
    print("WARNING: morphology_filters.py not found — morph columns will be empty. "
          "Copy morphology_filters.py to the same directory as phase_5_sweep.py.")


# =============================================================================
# Config
# =============================================================================
MODEL_PATH = "final_phase3_model.keras"
THRESHOLDS_JSON = "phase3_inference_config.json"

DEFAULT_INFERENCE_CFG = {
    "window_len": 301,
    "candidate_threshold": 0.86,
    "likely_threshold": 0.88,
    "high_confidence_threshold": 0.895,
}

CHECKPOINT_CSV = "tess_checkpoint.csv"
OUT_TOP_CSV = "tess_top_scores.csv"
OUT_SHORTLIST_CSV = "tess_candidates_086.csv"
OUT_HIGH_CSV = "tess_candidates_0895.csv"

# Run summary (this run only)
RUN_SUMMARY_CSV = "tess_run_summary.csv"

WRITE_TOP_N = 200

# Pipeline knobs
WINDOW_LEN = 301          # overwritten by phase3_inference_config if present
TARGET_LENGTH = 2001
DETREND_WIN = 101         # must be odd
CLIP_SIGMA = 8.0
MIN_FINITE_FRAC = 0.98

# BLS (time in days)
TOPK_BLS = 24
BLS_OVERSAMPLE = 5

BLS_PERIOD_MIN = 0.05   # days (1.2 hr) — captures USPs
BLS_N_PERIODS  = 3000   # denser grid helps when going short

# Candidate write thresholds (synced in main() from JSON)
WRITE_SHORTLIST_SCORE = DEFAULT_INFERENCE_CFG["candidate_threshold"]
WRITE_HIGH_SCORE = DEFAULT_INFERENCE_CFG["high_confidence_threshold"]

# Baseline control
MAX_SPAN_DAYS = 27.4  # 1 sector

# Networking / throttling
SLEEP_LO = 0.6
SLEEP_HI = 1.6
MAX_CONSECUTIVE_TIMEOUTS = 6


DISABLE_HEURISTICS_IN_BENCH = False

# ----------------------------
# Planet hunt mode (sector streaming)
# ----------------------------
USE_TEST_TARGETS = False

SECTOR_PROGRESS_JSON = "sector_progress.json"
SECTOR_START = 1
SECTOR_END = 200          # set high; code keeps going
SECTORS_PER_BATCH = 5     # search multiple sectors per batch
TARGETS_PER_SECTOR = 800  # cap per sector harvested
EMPTY_SECTOR_STOP = 999999  # effectively never stop for empties

MAX_WALLTIME_HOURS = 24

# Filters on MAST rows (safe defaults)
HARVEST_AUTHOR_WHITELIST = ["SPOC"]   # [] to allow all
HARVEST_EXPTIME_SECONDS = [120]       # [] to allow any cadence
HARVEST_DEBUG_SCHEMA = True           # print row/schema diagnostics

# Optional: brightness filter (slow; not implemented here)
HARVEST_TMAG_MAX = None

# ----------------------------
# Benchmark mode (Reviewer #1 scaffolding)
# ----------------------------
# NOTE:
# - BENCHMARK_TESS_ONLY=False by default to avoid empty results due to facility naming differences.
# - Set BENCHMARK_CONFIRMED=False for your original 24h sector-sweep behavior.
#
# ══════════════════════════════════════════════════════════
#  MODE SWITCH — change exactly these two lines:
#
#  BENCHMARK mode (confirmed-planet evaluation):
#    BENCHMARK_CONFIRMED = True
#    USE_TEST_TARGETS    = False
#
#  SWEEP mode (live TESS sector search):
#    BENCHMARK_CONFIRMED = False
#    USE_TEST_TARGETS    = False   (or True for quick smoke-test)
# ══════════════════════════════════════════════════════════
BENCHMARK_CONFIRMED = True            # True => run benchmark instead of sector sweep
BENCHMARK_LIMIT = 2000
BENCHMARK_OUT_CSV = "tess_benchmark_confirmed.csv"
BENCHMARK_TESS_ONLY = False           # FIX: default False to avoid 0-row filter
BENCHMARK_REQUIRE_2_TRANSITS = True   # True => P_true <= 0.5*MAX_SPAN_DAYS
BENCHMARK_PERIOD_FRAC_TOL = 0.02
BENCHMARK_MAX_HARMONIC = 6

# ----------------------------
# Test targets (optional)
# ----------------------------
TEST_TARGETS = [
    "TIC 7422496",
    "TIC 26547036",
    "TIC 27064468",
    "TIC 48018596",
    "TIC 49428710",
    "TIC 52368076",
    "TIC 91987762",
    "TIC 149601126",
    "TIC 192415680",
    "TIC 201248411",
]

# Checkpoint schema (includes sector)
CHECKPOINT_FIELDS = [
    "sector",
    "target",
    "score",
    "label",
    "download_status",

    "tspan_days",
    "median_dt_days",

    "bls_period",
    "bls_t0",
    "bls_duration",
    "bls_depth",
    "bls_power",

    "polarity_flipped",
    "score_dip",
    "score_bump",
    "score_used",
    "score_injected",
    "score_delta_injected",

    "w_nan_frac",
    "w_std",
    "min_abs_phase",

    # Morphology filter scores
    "morph_v_score",
    "morph_asymmetry",
    "morph_secondary_dip",
    "morph_depth_cv",
    "morph_duty_cycle",
    "morph_grazing_ratio",
    "morph_grazing_flag",
    "morph_n_flags",
    "morph_is_strong_fp",
    "morph_flags",
]


# =============================================================================
# Utilities
# =============================================================================
def ensure_odd(k: int) -> int:
    k = int(k)
    return k if (k % 2 == 1) else (k + 1)


DETREND_WIN = ensure_odd(DETREND_WIN)


def polite_sleep(lo: float = SLEEP_LO, hi: float = SLEEP_HI) -> None:
    time.sleep(random.uniform(lo, hi))


def normalize_target_name(target: str) -> str:
    t = str(target).strip()
    if t.upper().startswith("TOI-"):
        return "TOI " + t[4:]
    return t


def load_thresholds(path: str) -> Tuple[float, float, float]:
    """Returns (candidate, likely, high) and updates WINDOW_LEN. Writes defaults if missing."""
    global WINDOW_LEN
    if not os.path.exists(path):
        print(f"Warning: thresholds file not found: {path} (creating from defaults)")
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w") as f:
            json.dump(DEFAULT_INFERENCE_CFG, f, indent=2)

        WINDOW_LEN = int(DEFAULT_INFERENCE_CFG["window_len"])
        return (
            float(DEFAULT_INFERENCE_CFG["candidate_threshold"]),
            float(DEFAULT_INFERENCE_CFG["likely_threshold"]),
            float(DEFAULT_INFERENCE_CFG["high_confidence_threshold"]),
        )

    with open(path, "r") as f:
        cfg = json.load(f)

    WINDOW_LEN = int(cfg.get("window_len", DEFAULT_INFERENCE_CFG["window_len"]))
    t_candidate = float(cfg.get("candidate_threshold", DEFAULT_INFERENCE_CFG["candidate_threshold"]))
    t_likely = float(cfg.get("likely_threshold", DEFAULT_INFERENCE_CFG["likely_threshold"]))
    t_high = float(cfg.get("high_confidence_threshold", DEFAULT_INFERENCE_CFG["high_confidence_threshold"]))

    print(f"✓ Loaded thresholds from {path}: {t_candidate}/{t_likely}/{t_high}")
    print(f"✓ WINDOW_LEN={WINDOW_LEN}")
    return t_candidate, t_likely, t_high


def label_scores(scores: np.ndarray, t_candidate: float, t_likely: float, t_high: float) -> np.ndarray:
    labels = np.full(len(scores), "reject", dtype=object)
    labels[scores >= t_candidate] = "candidate"
    labels[scores >= t_likely] = "likely_planet"
    labels[scores >= t_high] = "high_confidence_planet"
    return labels


def append_checkpoint_row(csv_path: str, row: Dict[str, Any]) -> None:
    out = {k: row.get(k, np.nan) for k in CHECKPOINT_FIELDS}
    df = pd.DataFrame([out], columns=CHECKPOINT_FIELDS)
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=header, index=False)


def write_candidates(path: str, rows: List[Dict[str, Any]]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"✓ Wrote candidates: {path} ({len(df)} rows)")


def load_sector_progress(path: str, default_sector: int) -> int:
    if not os.path.exists(path):
        return int(default_sector)
    try:
        with open(path, "r") as f:
            d = json.load(f)
        return int(d.get("next_sector", default_sector))
    except Exception:
        return int(default_sector)


def save_sector_progress(path: str, next_sector: int) -> None:
    with open(path, "w") as f:
        json.dump({"next_sector": int(next_sector)}, f, indent=2)


def load_done_targets() -> set:
    done = set()
    if os.path.exists(CHECKPOINT_CSV):
        try:
            df_done = pd.read_csv(CHECKPOINT_CSV, usecols=["target"])
            done = set(str(x).strip() for x in df_done["target"].dropna().values)
            print(f"✓ Resume enabled: {len(done)} targets already in {CHECKPOINT_CSV}")
        except Exception as e:
            print(f"Warning: could not read checkpoint for resume: {e}")
    return done


# =============================================================================
# Phase-2-aligned preprocessing
# =============================================================================
def moving_median(x: np.ndarray, k: int) -> np.ndarray:
    k = ensure_odd(k)
    n = len(x)
    half = k // 2
    out = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = np.median(x[lo:hi])
    return out


def detrend_flux(flux: np.ndarray, win: int = DETREND_WIN) -> np.ndarray:
    flux = flux.astype(float)
    finite = np.isfinite(flux)
    if finite.mean() < MIN_FINITE_FRAC:
        med = np.nanmedian(flux[finite]) if finite.any() else 0.0
        flux = np.where(finite, flux, med)

    trend = moving_median(flux, win)
    detr = flux - trend

    s = np.std(detr[np.isfinite(detr)])
    if s == 0 or not np.isfinite(s):
        return detr

    detr = detr / s
    detr = np.clip(detr, -CLIP_SIGMA, CLIP_SIGMA)
    return detr


def zscore_window(w: np.ndarray) -> np.ndarray:
    w = w.astype(np.float32)
    mu = float(np.nanmean(w))
    sd = float(np.nanstd(w))
    if (not np.isfinite(sd)) or sd == 0:
        return w - mu
    return (w - mu) / sd


# =============================================================================
# Benchmark helpers (DEFINE ONCE, ABOVE run_confirmed_benchmark)
# =============================================================================
def _rel_err(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-12)


def period_match_harmonic(
    p_pred: float,
    p_true: float,
    frac_tol: float = 0.02,
    max_harm: int = 6
) -> Tuple[bool, Optional[int], Optional[str], float]:
    """
    Harmonic/alias-tolerant match.

    Returns:
      (matched, k, kind, rel_error)

    kind:
      - "fundamental" if k=1 and p_pred ~ p_true
      - "multiple"    if p_pred ~ k * p_true (k>=2)
      - "divisor"     if p_pred ~ p_true / k (k>=2)
    """
    if not (np.isfinite(p_pred) and np.isfinite(p_true) and p_pred > 0 and p_true > 0):
        return False, None, None, np.inf

    best_err = np.inf
    best_k: Optional[int] = None
    best_kind: Optional[str] = None

    for k in range(1, int(max_harm) + 1):
        # multiple: p_pred ~ k * p_true
        target = p_true * k
        err = _rel_err(p_pred, target)
        if err < best_err:
            best_err = err
            best_k = k
            best_kind = ("multiple" if k > 1 else "fundamental")

        # divisor: p_pred ~ p_true / k
        target = p_true / k
        err = _rel_err(p_pred, target)
        if err < best_err:
            best_err = err
            best_k = k
            best_kind = ("divisor" if k > 1 else "fundamental")

    if best_err > frac_tol:
        return False, None, None, float(best_err)

    if best_k == 1:
        best_kind = "fundamental"

    return True, int(best_k), str(best_kind), float(best_err)


def period_matches(p_pred: float, p_true: float, frac_tol: float = 0.02, max_harm: int = 6) -> bool:
    """
    Backwards-compatible boolean wrapper.
    This keeps your benchmark code unchanged while using the new harmonic matcher.
    """
    matched, _, _, _ = period_match_harmonic(p_pred, p_true, frac_tol=frac_tol, max_harm=max_harm)
    return bool(matched)


# =============================================================================
# BLS + windowing (real time axis, days)
# =============================================================================
def bls_top_candidates(
    t: np.ndarray,
    flux: np.ndarray,
    top_k: int = TOPK_BLS,
    period_min: float = 0.05,
    period_max: Optional[float] = None,
    n_periods: int = 3000,
    dur_min_days: float = 0.5 / 24.0,
    dur_max_days: float = 12.0 / 24.0,
    n_dur: int = 25,
    oversample: int = BLS_OVERSAMPLE,
    usp_period_max: float = 0.60,
    normal_period_min: float = 0.60,
    dedupe_rel: float = 0.002,
    debug_once: bool = True,
) -> List[Tuple[float, float, float, float, float]]:
    """
    Two-pass BLS:
      - USP/short-period pass: [period_min, usp_period_max] with short durations
      - Normal pass:          [normal_period_min, period_max] with long durations

    Astropy BoxLeastSquares requires per call:
        max(duration_grid) < min(period_grid)
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(flux, dtype=float)

    m = np.isfinite(t) & np.isfinite(y)
    t = t[m]; y = y[m]
    if t.size < 200:
        return []

    o = np.argsort(t)
    t = t[o]; y = y[o]

    tspan = float(t[-1] - t[0])
    if (not np.isfinite(tspan)) or tspan <= 0:
        return []

    if period_max is None or (not np.isfinite(period_max)) or period_max <= 0:
        period_max = 0.5 * tspan

    period_min = float(period_min)
    period_max = float(period_max)
    period_min = max(period_min if np.isfinite(period_min) and period_min > 0 else 0.05, 0.02)
    if period_max <= period_min:
        period_max = period_min * 2.0

    usp_period_max = float(usp_period_max)
    normal_period_min = float(normal_period_min)

    # Ensure normal pass is compatible with long durations
    if normal_period_min <= dur_max_days:
        normal_period_min = float(dur_max_days) * 1.20

    usp_lo = period_min
    usp_hi = min(usp_period_max, period_max)

    norm_lo = max(normal_period_min, period_min * 1.01)
    norm_hi = period_max

    bls = BoxLeastSquares(t, y)

    def run_pass(p_lo: float, p_hi: float, d_lo: float, d_hi: float, nP: int, nD: int, pass_name: str):
        if (not np.isfinite(p_lo)) or (not np.isfinite(p_hi)) or p_hi <= p_lo:
            return []

        nP = int(max(200, nP))
        nD = int(max(5, nD))

        periods = np.exp(np.linspace(np.log(p_lo), np.log(p_hi), nP)).astype(float)

        pmin = float(periods.min())
        pmin_margin = 0.98 * pmin

        d_lo = float(d_lo)
        d_hi = float(d_hi)

        if (not np.isfinite(d_lo)) or d_lo <= 0:
            d_lo = max(0.001, 0.02 * pmin)

        d_hi_eff = min(d_hi, pmin_margin)
        if (not np.isfinite(d_hi_eff)) or d_hi_eff <= d_lo:
            d_lo = max(0.001, 0.02 * pmin)
            d_hi_eff = min(0.2 * pmin, pmin_margin)

        durations = np.geomspace(d_lo, d_hi_eff, nD).astype(float)
        durations = np.unique(durations)
        durations = durations[np.isfinite(durations)]
        durations = durations[(durations > 0) & (durations < pmin_margin)]
        if durations.size < 3:
            return []

        if debug_once and not hasattr(bls_top_candidates, "_printed_passes"):
            bls_top_candidates._printed_passes = set()
        if debug_once and pass_name not in bls_top_candidates._printed_passes:
            bls_top_candidates._printed_passes.add(pass_name)
            print(f"[BLS DEBUG {pass_name}] p_lo={p_lo:.6f} p_hi={p_hi:.6f} "
                  f"min(periods)={periods.min():.6f} max(durations)={durations.max():.6f} "
                  f"nP={len(periods)} nD={len(durations)} d_lo={d_lo:.6f} d_hi_req={d_hi:.6f}")

        if durations.max() >= periods.min():
            raise RuntimeError(f"[BLS GRID BAD {pass_name}] max(dur)={durations.max()} >= min(period)={periods.min()}")

        try:
            res = bls.power(periods, durations, oversample=int(oversample))
        except Exception as e:
            raise RuntimeError(
                f"BLS power() failed ({pass_name}): {type(e).__name__}: {e} | "
                f"minP={periods.min():.6g} maxDur={durations.max():.6g}"
            )

        power = np.asarray(res.power, dtype=float)
        if power.size == 0 or not np.any(np.isfinite(power)):
            return []

        cand = []
        take = max(1, int(top_k)) * 10
        for ii in np.argsort(power)[::-1][:take]:
            p = float(res.period[ii])
            t0 = float(res.transit_time[ii])
            dur = float(res.duration[ii])
            dep = float(res.depth[ii])
            powv = float(res.power[ii])
            if not (np.isfinite(p) and np.isfinite(t0) and np.isfinite(dur) and np.isfinite(dep) and np.isfinite(powv)):
                continue
            cand.append((p, t0, dur, dep, powv))
        return cand

    def log_span(a: float, b: float) -> float:
        if (not np.isfinite(a)) or (not np.isfinite(b)) or a <= 0 or b <= 0 or b <= a:
            return 0.0
        return float(np.log(b) - np.log(a))

    usp_span = log_span(usp_lo, usp_hi)
    norm_span = log_span(norm_lo, norm_hi)
    total_span = usp_span + norm_span

    merged: List[Tuple[float, float, float, float, float]] = []

    if total_span <= 0:
        p_lo, p_hi = period_min, period_max
        d_lo = max(1.0 / 24.0 / 12.0, min(dur_min_days, 0.05 * p_lo))
        d_hi = min(dur_max_days, 0.2 * p_lo)
        merged = run_pass(p_lo, p_hi, d_lo, d_hi, n_periods, n_dur, "FALLBACK")
    else:
        nP_usp = int(round(n_periods * (usp_span / total_span))) if usp_span > 0 else 0
        nP_norm = n_periods - nP_usp

        if usp_span > 0:
            nP_usp = max(400, nP_usp)
        if norm_span > 0:
            nP_norm = max(600, nP_norm)

        if (usp_span > 0) and (norm_span > 0):
            extra = (nP_usp + nP_norm) - n_periods
            if extra > 0:
                if nP_norm >= nP_usp:
                    nP_norm = max(600, nP_norm - extra)
                else:
                    nP_usp = max(400, nP_usp - extra)

        usp_dur_lo = max(1.0 / 24.0 / 12.0, min(dur_min_days, 0.10 * usp_lo))
        usp_dur_hi = min(1.0 / 24.0, 0.20 * usp_hi)

        norm_dur_lo = dur_min_days
        norm_dur_hi = dur_max_days

        cands_usp = run_pass(usp_lo, usp_hi, usp_dur_lo, usp_dur_hi, nP_usp, max(7, n_dur // 2), "USP") if usp_span > 0 else []
        cands_norm = run_pass(norm_lo, norm_hi, norm_dur_lo, norm_dur_hi, nP_norm, n_dur, "NORMAL") if norm_span > 0 else []

        merged = cands_usp + cands_norm

    if not merged:
        return []

    merged.sort(key=lambda x: x[4], reverse=True)

    out: List[Tuple[float, float, float, float, float]] = []
    kept_periods: List[float] = []
    rel = float(dedupe_rel)

    for p, t0, dur, dep, powv in merged:
        if not np.isfinite(p):
            continue
        if any(abs(p - pk) / max(pk, 1e-9) <= rel for pk in kept_periods):
            continue
        out.append((p, t0, dur, dep, powv))
        kept_periods.append(p)
        if len(out) >= int(top_k):
            break

    return out


def fold_and_center_window(
    t: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    window_len: int = WINDOW_LEN
) -> np.ndarray:
    """
    Fold on (period, t0) and interpolate onto a fixed phase grid in [-0.5, 0.5).

    FIX for native-cadence + wrap:
      - circularly extend phase by +/-1 before interpolation so np.interp
        doesn't bridge the -0.5/0.5 discontinuity.
    """
    t = np.asarray(t, dtype=float)
    flux = np.asarray(flux, dtype=float)

    m = np.isfinite(t) & np.isfinite(flux)
    t = t[m]
    flux = flux[m]
    if t.size < 10:
        return np.full(window_len, np.nan, dtype=np.float32)

    period = float(period)
    t0 = float(t0)
    if (not np.isfinite(period)) or period <= 0:
        return np.full(window_len, np.nan, dtype=np.float32)

    phase = ((t - t0 + 0.5 * period) % period) / period - 0.5

    o = np.argsort(phase)
    phase = phase[o].astype(np.float64)
    flux = flux[o].astype(np.float64)

    phase_ext = np.concatenate([phase - 1.0, phase, phase + 1.0])
    flux_ext = np.concatenate([flux, flux, flux])

    phase_grid = np.linspace(-0.5, 0.5, int(window_len), endpoint=False, dtype=np.float64)
    w = np.interp(phase_grid, phase_ext, flux_ext).astype(np.float32)

    return w


# =============================================================================
# Download a TESS light curve (single product, no stitch)
# =============================================================================
def download_tess_lc_mainprocess(
    target: str,
    target_length: int = TARGET_LENGTH,
    min_points: int = 800,
    prefer_spoc: bool = False,
    max_span_days: float = 27.4,
) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray]], str]:
    tgt = normalize_target_name(target)

    try:
        search = lk.search_lightcurve(tgt, mission="TESS", limit=50)
        if len(search) == 0:
            return None, "no_lc"

        # Prefer SPOC if requested
        if prefer_spoc:
            spoc_idx = None
            for i in range(len(search)):
                auth = str(getattr(search[i], "author", "")).upper()
                if auth in ("SPOC", "TESS-SPOC"):
                    spoc_idx = i
                    break
            if spoc_idx is not None and spoc_idx != 0:
                search = search[[spoc_idx] + [i for i in range(len(search)) if i != spoc_idx]]

        # Single product only (NO stitch)
        lc = search[0].download()
        if lc is None:
            return None, "download_none"

        lc = lc.remove_nans()
        if len(lc) < min_points:
            return None, "too_short"

        t = np.asarray(lc.time.value, dtype=float)

        # Prefer PDCSAP/SAP columns if present, else lc.flux
        f = None
        try:
            cols = getattr(lc, "columns", [])
            if "pdcsap_flux" in cols:
                f = np.asarray(lc["pdcsap_flux"].value, dtype=float)
            elif "sap_flux" in cols:
                f = np.asarray(lc["sap_flux"].value, dtype=float)
        except Exception:
            f = None
        if f is None:
            f = np.asarray(lc.flux.value, dtype=float)

        if t.size != f.size or t.size < min_points:
            return None, "bad_time"

        m = np.isfinite(t) & np.isfinite(f)
        t = t[m]
        f = f[m]
        if t.size < min_points:
            return None, "too_short"

        o = np.argsort(t)
        t = t[o]
        f = f[o]

        tu, iu = np.unique(t, return_index=True)
        t = tu
        f = f[iu]
        if t.size < min_points:
            return None, "too_short"

        # Enforce baseline window
        if max_span_days is not None and np.isfinite(max_span_days) and max_span_days > 0:
            t_end = float(t[-1])
            t_start = t_end - float(max_span_days)
            keep = t >= t_start
            if np.sum(keep) >= min_points:
                t = t[keep]
                f = f[keep]

        if t.size < min_points:
            return None, "too_short"

        # Robust clip
        med = float(np.median(f))
        mad = float(np.median(np.abs(f - med))) + 1e-12
        lo = med - 12.0 * 1.4826 * mad
        hi = med + 12.0 * 1.4826 * mad
        f = np.clip(f, lo, hi)

        # Normalize median 0 std 1
        med = float(np.median(f))
        sd = float(np.std(f))
        if (not np.isfinite(sd)) or sd <= 0:
            return None, "bad_std"
        f = (f - med) / sd

        if len(f) < min_points:
            return None, "too_short"

        # Resample by interpolation to fixed length
        t0 = float(t[0])
        t_rel = (t - t0).astype(np.float64)
        f_rel = f.astype(np.float64)

        t_grid = np.linspace(float(t_rel[0]), float(t_rel[-1]), int(target_length), dtype=np.float64)
        f_grid = np.interp(t_grid, t_rel, f_rel)

        time_rs = t_grid.astype(np.float32)
        flux_rs = f_grid.astype(np.float32)

        if np.any(np.diff(time_rs) <= 0):
            return None, "bad_resample_time"

        if (not np.all(np.isfinite(flux_rs))) or (not np.all(np.isfinite(time_rs))):
            return None, "all_nan"

        if time_rs.size != target_length or flux_rs.size != target_length:
            return None, "bad_resample"

        return (time_rs, flux_rs), "ok"

    except Exception as e:
        msg = str(e).replace("\n", " ").strip()
        return None, f"error:{msg[:120]}"


# =============================================================================
# Scoring
# =============================================================================
def score_target_topk(
    model,
    payload: Tuple[np.ndarray, np.ndarray],
) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Single-exit scorer:
      LC -> detrend -> BLS top-K -> fold -> CNN -> select candidate

    Returns:
      best_score (float), diag (dict), cand_rows (list of dict)

    Selection (v5.x):
      1) High-CNN override: if any cand CNN >= HIGH_CNN_OVERRIDE_THRESH, pick max CNN among them.
      2) Else: build a CNN-qualified set near best gated-CNN, then maximize combined objective.
      3) Floor-guard: if chosen period is near-floor, allow plausible long-period candidate.
      4) Alias promote longer: prefer longer integer multiple when CNN is near-tied and power not too weak.
      5) Short-alias guard: after longer promotion, allow stepping down to plausible divisor (avoid over-promoting).
    """

    # -------------------------
    # Knobs
    # -------------------------
    ALPHA                    = 0.70    # hybrid diagnostics only
    USE_LOGIT_CNN            = False

    FLOOR_PERIOD_MULT        = 2.00
    FLOOR_PENALTY_SCALE      = 0.40
    HIGH_CNN_OVERRIDE_THRESH = 0.95

    # CNN-qualified set knobs
    CNN_REL_TO_BEST          = 0.02
    CNN_MIN_KEEP             = 0.00
    GAMMA_CNN                = 2.0
    BETA_POW                 = 0.10     # 0 => power-neutral (combined reduces to CNN^GAMMA)

    # Floor guard knobs
    FLOOR_GUARD_ENABLE       = True
    FLOOR_GUARD_MIN_LONG_P   = 0.15
    FLOOR_GUARD_CNN_DROP     = 0.03
    FLOOR_GUARD_POW_FLOOR    = 0.25
    FLOOR_GUARD_COMBINED_EPS = 0.85

    # Alias/harmonic promote longer knobs
    ALIAS_TIE_EPS            = 0.01   # tightened from 0.02: only promote near-tied CNN
    ALIAS_POWER_FLOOR        = 0.55   # combined guard is now the real backstop; coarse pre-filter only
    ALIAS_MAX_K              = 20
    ALIAS_RATIO_TOL          = 0.05   # widened from 0.03: BLS period noise makes true 3x appear as 2.96x etc.

    # Short-alias guard knobs (divisors)
    SHORT_ALIAS_ENABLE       = True
    SHORT_ALIAS_MIN_PERIOD   = max(float(BLS_PERIOD_MIN) * 1.10, 0.08)
    SHORT_ALIAS_TIE_EPS      = 0.03   # loosened from 0.02: more willing to step down
    SHORT_ALIAS_POWER_FLOOR  = 0.25   # loosened from 0.40: less strict power requirement
    SHORT_ALIAS_MAX_K        = 20
    SHORT_ALIAS_RATIO_TOL    = 0.03

    time_rs, flux_rs = payload
    diag: Dict[str, Any] = {}

    diag["tspan_days"]     = float(time_rs[-1] - time_rs[0])
    diag["median_dt_days"] = float(np.median(np.diff(time_rs)))

    dflux = detrend_flux(flux_rs, win=DETREND_WIN)

    cands = bls_top_candidates(
        time_rs, dflux,
        top_k=TOPK_BLS,
        period_min=BLS_PERIOD_MIN,
        n_periods=BLS_N_PERIODS,
    )

    if BENCHMARK_CONFIRMED and DISABLE_HEURISTICS_IN_BENCH:
        FLOOR_GUARD_ENABLE = False
        # alias promote & short alias guard off too
        ALIAS_TIE_EPS = 0.0          # effectively disables promote (or add a boolean)
        SHORT_ALIAS_ENABLE = True
        ALIAS_PROMOTE_ENABLE = True

    if not hasattr(score_target_topk, "_printed_version"):
        print("[score_target_topk] VERSION = single-exit v5.x (qualified+combined + floor_guard + alias_long + short_guard)")
        score_target_topk._printed_version = True

    if not cands:
        raise ValueError("BLS produced no candidates")

    # ---------------------------------------------------------------
    # Pass 1: fold candidates and batch CNN
    # ---------------------------------------------------------------
    valid_indices: List[int] = []
    windows_z: List[np.ndarray] = []
    w_raws: List[np.ndarray] = []

    for j, (period, t0, dur, depth, power) in enumerate(cands):
        w = fold_and_center_window(time_rs, dflux, float(period), float(t0), window_len=WINDOW_LEN)
        if len(w) != WINDOW_LEN:
            continue

        w_raw = w.astype(np.float32)
        w_z   = zscore_window(w_raw)
        w_z   = np.nan_to_num(w_z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if not np.all(np.isfinite(w_z)):
            continue

        valid_indices.append(j)
        windows_z.append(w_z)
        w_raws.append(w_raw)

    if not valid_indices:
        raise ValueError("No usable BLS candidates produced a valid window")

    batch      = np.stack(windows_z, axis=0)[:, :, None]
    batch_flip = np.stack([-w for w in windows_z], axis=0)[:, :, None]

    s_dips  = model.predict(batch,      verbose=0).ravel().astype(float)
    s_bumps = model.predict(batch_flip, verbose=0).ravel().astype(float)

    # ---------------------------------------------------------------
    # Build cand_rows (BLS-rank order)
    # ---------------------------------------------------------------
    cand_rows: List[Dict[str, Any]] = []
    for idx, j in enumerate(valid_indices):
        period, t0, dur, depth, power = cands[j]
        s_dip  = float(s_dips[idx])
        s_bump = float(s_bumps[idx])

        if s_bump > s_dip:
            s_used = s_bump
            polarity_flipped = True
        else:
            s_used = s_dip
            polarity_flipped = False

        w_raw      = w_raws[idx]
        w_chk      = zscore_window(w_raw)
        w_nan_frac = float(np.mean(~np.isfinite(w_chk)))
        w_std      = float(np.nanstd(w_chk))

        cand_rows.append({
            "k":                int(j + 1),  # 1-based rank in BLS output
            "bls_period":       float(period),
            "bls_t0":           float(t0),
            "bls_duration":     float(dur),
            "bls_depth":        float(depth) if depth is not None else np.nan,
            "bls_power":        float(power),
            "score_dip":        s_dip,
            "score_bump":       s_bump,
            "score_used":       float(s_used),
            "polarity_flipped": bool(polarity_flipped),
            "w_nan_frac":       w_nan_frac,
            "w_std":            w_std,
        })

    # ---------------------------------------------------------------
    # Pass 2: selection
    # ---------------------------------------------------------------
    cnn_scores  = np.array([cr["score_used"] for cr in cand_rows], dtype=float)
    bls_powers  = np.array([cr["bls_power"]  for cr in cand_rows], dtype=float)
    periods_arr = np.array([cr["bls_period"] for cr in cand_rows], dtype=float)

    # CNN term for hybrid diagnostics
    if USE_LOGIT_CNN:
        p_clip = np.clip(cnn_scores, 1e-4, 1.0 - 1e-4)
        cnn_term = np.log(p_clip / (1.0 - p_clip))
        cnn_term = (cnn_term + 6.0) / 12.0
        cnn_term = np.clip(cnn_term, 0.0, 1.0)
    else:
        cnn_term = np.clip(cnn_scores, 0.0, 1.0)

    # Power term
    log_pow  = np.log1p(bls_powers)
    pow_max  = float(log_pow.max()) if log_pow.size else 0.0
    pow_norm = (log_pow / pow_max) if pow_max > 1e-9 else np.ones_like(log_pow)

    # Floor penalty (GATE ONLY)
    period_floor  = float(BLS_PERIOD_MIN) * float(FLOOR_PERIOD_MULT)
    near_floor    = periods_arr < period_floor
    weak_power    = pow_norm < 0.2
    floor_penalty = np.where(near_floor, float(FLOOR_PENALTY_SCALE), 1.0)

    # Hybrid for diagnostics only
    hybrid = (ALPHA * cnn_term + (1.0 - ALPHA) * pow_norm) * floor_penalty

    # Combined objective — floor_penalty baked in so near-floor artifacts can't win selection
    eps = 1e-9
    score_cnn = np.clip(cnn_scores, 0.0, 1.0) + eps
    score_pow = np.clip(log_pow, 0.0, None) + eps
    combined  = (score_cnn ** float(GAMMA_CNN)) * (score_pow ** float(BETA_POW)) * floor_penalty

    # ---- Selection logic ----
    alias_switched = False

    high_cnn_mask = cnn_scores >= float(HIGH_CNN_OVERRIDE_THRESH)
    if high_cnn_mask.any():
        keep = np.where(high_cnn_mask)[0]
        best_idx = int(keep[np.argmax(cnn_scores[keep])])
        selection_mode = "high_cnn_override"
    else:
        cnn_gate_score = cnn_scores * floor_penalty
        best_gate = float(np.max(cnn_gate_score))

        keep = np.where(cnn_gate_score >= (best_gate - float(CNN_REL_TO_BEST)))[0]
        if keep.size == 0:
            keep = np.array([int(np.argmax(cnn_gate_score))], dtype=int)

        if float(CNN_MIN_KEEP) > 0:
            keep2 = keep[cnn_scores[keep] >= float(CNN_MIN_KEEP)]
            if keep2.size > 0:
                keep = keep2

        best_idx = int(keep[np.argmax(combined[keep])])
        selection_mode = "cnn_qualified_combined"

 
    # ---------------------------------------------------------------
    # Floor guard: if chosen is near-floor, allow plausible long pick
    # ---------------------------------------------------------------
    if FLOOR_GUARD_ENABLE:
        chosen_p = float(periods_arr[best_idx])
        if np.isfinite(chosen_p) and (chosen_p < float(period_floor)):
            long_idx = np.where(periods_arr >= float(FLOOR_GUARD_MIN_LONG_P))[0]
            if long_idx.size > 0:
                ok = []
                for ii in long_idx:
                    if cnn_scores[ii] < (cnn_scores[best_idx] - float(FLOOR_GUARD_CNN_DROP)):
                        continue
                    if bls_powers[ii] < float(FLOOR_GUARD_POW_FLOOR) * max(bls_powers[best_idx], 1e-12):
                        continue
                    if combined[ii] < float(FLOOR_GUARD_COMBINED_EPS) * max(combined[best_idx], 1e-12):
                        continue
                    ok.append(int(ii))

                if ok:
                    best_long = int(ok[np.argmax(combined[ok])])
                    if best_long != int(best_idx):
                        best_idx = best_long
                        selection_mode = selection_mode + "+floor_guard"

    # ---------------------------------------------------------------
    # Alias promote longer: prefer LONGER integer multiple when near-tied CNN
    # ---------------------------------------------------------------
    best_idx0 = int(best_idx)
    p0   = float(periods_arr[best_idx0])
    s0   = float(cnn_scores[best_idx0])
    pow0 = float(bls_powers[best_idx0])
    # Used by alias period cap and transit_ceil_guard below
    tspan_days = float(diag.get("tspan_days", np.nan))

    for ii in np.argsort(cnn_scores)[::-1]:
        pi  = float(periods_arr[ii])
        si  = float(cnn_scores[ii])
        pwi = float(bls_powers[ii])

        if pi <= p0:
            continue
        if si < (s0 - float(ALIAS_TIE_EPS)):
            break
        if pwi < float(ALIAS_POWER_FLOOR) * max(pow0, 1e-12):
            continue
        # Cap: never promote to a period with fewer than 3 observable transits.
        # Without this cap the loop cascades (A→B→C…) accumulating period errors
        # and landing on unphysical long periods that transit_ceil_guard must fix.
        if np.isfinite(tspan_days) and tspan_days > 0:
            if pi > tspan_days / 3.0:
                continue

        ratio = pi / max(p0, 1e-12)
        k = int(round(ratio))
        if 2 <= k <= int(ALIAS_MAX_K) and abs(ratio - k) < float(ALIAS_RATIO_TOL):
            # Guard: only promote if longer candidate is competitive under combined objective
            if combined[ii] < 0.98 * combined[best_idx0]:
                continue
            best_idx0 = int(ii)
            p0, s0, pow0 = pi, si, pwi
            alias_switched = True

    if alias_switched:
        best_idx = best_idx0
        selection_mode = selection_mode + "+alias_promote_longer"

    # ---------------------------------------------------------------
    # Short-alias guard: try stepping DOWN to a plausible divisor
    # ---------------------------------------------------------------
    if SHORT_ALIAS_ENABLE:
        cur_idx = int(best_idx)
        p_cur   = float(periods_arr[cur_idx])
        s_cur   = float(cnn_scores[cur_idx])
        pow_cur = float(bls_powers[cur_idx])

        best_short = cur_idx

        for ii in np.argsort(cnn_scores)[::-1]:
            pi  = float(periods_arr[ii])
            si  = float(cnn_scores[ii])
            pwi = float(bls_powers[ii])

            if not (np.isfinite(pi) and np.isfinite(p_cur)):
                continue
            if pi >= p_cur:
                continue
            if pi < float(SHORT_ALIAS_MIN_PERIOD):
                continue
            if si < (s_cur - float(SHORT_ALIAS_TIE_EPS)):
                continue
            if pwi < float(SHORT_ALIAS_POWER_FLOOR) * max(pow_cur, 1e-12):
                continue

            ratio = p_cur / max(pi, 1e-12)
            k = int(round(ratio))
            if 2 <= k <= int(SHORT_ALIAS_MAX_K) and abs(ratio - k) < float(SHORT_ALIAS_RATIO_TOL):
                best_short = int(ii)
                p_cur, s_cur, pow_cur = pi, si, pwi

        if best_short != int(best_idx):
            best_idx = int(best_short)
            selection_mode = selection_mode + "+short_alias_guard"

    # ---------------------------------------------------------------
    # 3-transit ceiling guard: reject periods that allow < 3 transits
    # in the observed baseline. Scientifically motivated — you can't
    # confirm a transit signature with fewer than 3 events.
    # ---------------------------------------------------------------
    TRANSIT_MIN_COUNT = 3
    # tspan_days already defined above (used also in alias period cap)
    if np.isfinite(tspan_days) and tspan_days > 0:
        p_ceil = tspan_days / float(TRANSIT_MIN_COUNT)
        chosen_p = float(periods_arr[best_idx])
        if np.isfinite(chosen_p) and chosen_p > p_ceil:
            # Fallen back to an unphysically long period — find best
            # surviving candidate that fits within the ceiling
            ok_mask = periods_arr <= p_ceil
            if ok_mask.any():
                # Among survivors, pick by combined objective
                ok_idxs = np.where(ok_mask)[0]
                fallback_idx = int(ok_idxs[np.argmax(combined[ok_idxs])])
                best_idx = fallback_idx
                selection_mode = selection_mode + "+transit_ceil_guard"

    # Winner (defined ONCE, always)
    best_cr = cand_rows[int(best_idx)]
    best_score  = float(best_cr["score_used"])
    best_period = float(best_cr["bls_period"])
    best_t0     = float(best_cr["bls_t0"])
    best_dur    = float(best_cr["bls_duration"])
    best_depth  = float(best_cr["bls_depth"])
    best_power  = float(best_cr["bls_power"])

    # ---------------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------------
    top5_idx = np.argsort(hybrid)[::-1][:5]
    top5_diag = [{
        "k":       int(cand_rows[i]["k"]),
        "period":  float(periods_arr[i]),
        "cnn":     float(cnn_scores[i]),
        "power":   float(bls_powers[i]),
        "hybrid":  float(hybrid[i]),
        "penalty": float(floor_penalty[i]),
    } for i in top5_idx]

    gate_idx = keep.tolist() if hasattr(keep, "tolist") else list(keep)

    diag.update({
        "bls_period":   best_period,
        "bls_t0":       best_t0,
        "bls_duration": best_dur,
        "bls_depth":    best_depth,
        "bls_power":    best_power,

        "polarity_flipped":   bool(best_cr["polarity_flipped"]),
        "score_dip":          float(best_cr["score_dip"]),
        "score_bump":         float(best_cr["score_bump"]),
        "score_used":         float(best_cr["score_used"]),
        "w_nan_frac":         float(best_cr["w_nan_frac"]),
        "w_std":              float(best_cr["w_std"]),
        "best_k":             int(best_cr["k"]),

        "hybrid_score_best":  float(hybrid[int(best_idx)]),
        "floor_penalty_best": float(floor_penalty[int(best_idx)]),
        "pow_norm_best":      float(pow_norm[int(best_idx)]),

        "selection_mode":     str(selection_mode),
        "alias_switched":     bool(alias_switched),
        "gate_idx":           gate_idx,
        "top5_candidates":    top5_diag,
    })

    phase = ((time_rs - best_t0 + 0.5 * best_period) % best_period) / best_period - 0.5
    diag["min_abs_phase"] = float(np.nanmin(np.abs(phase)))

    # ---------------------------------------------------------------
    # Injection test on the winning candidate
    # ---------------------------------------------------------------
    w_best_raw = fold_and_center_window(time_rs, dflux, best_period, best_t0, window_len=WINDOW_LEN).astype(np.float32)
    w_used_raw = -w_best_raw if diag.get("polarity_flipped", False) else w_best_raw

    def _inject_box_transit(w_in: np.ndarray, sigma_mult: float = 8.0, width_frac: float = 0.03) -> np.ndarray:
        x = np.linspace(-0.5, 0.5, len(w_in), endpoint=False, dtype=np.float32)
        mask = np.abs(x) <= float(width_frac)
        w2 = w_in.copy().astype(np.float32)
        s = float(np.nanstd(w2))
        if (not np.isfinite(s)) or s == 0:
            s = 1.0
        w2[mask] -= float(sigma_mult) * s
        return w2

    try:
        w_inj = zscore_window(_inject_box_transit(w_used_raw))
        w_inj = np.nan_to_num(w_inj, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        s_inj = float(model.predict(w_inj[None, :, None], verbose=0).ravel()[0])
        diag["score_injected"]       = float(s_inj)
        diag["score_delta_injected"] = float(s_inj - best_score)
    except Exception as e:
        diag["score_injected"]       = np.nan
        diag["score_delta_injected"] = np.nan
        diag["inject_error"]         = f"{type(e).__name__}: {e}"

    diag["n_cands_used"] = int(len(cand_rows))
    diag["score"]        = float(best_score)

    # ---------------------------------------------------------------
    # Morphology pre-filters on the winning candidate
    # Scores the folded phase curve for transit shape, secondary eclipse,
    # duty cycle, and ingress/egress asymmetry to flag likely EBs and
    # rotation signals before they reach the output catalog.
    # ---------------------------------------------------------------
    _MORPH_DEFAULTS = {
        "morph_v_score":       None,
        "morph_asymmetry":     None,
        "morph_secondary_dip": None,
        "morph_depth_cv":      None,
        "morph_duty_cycle":    None,
        "morph_n_flags":       -1,
        "morph_is_strong_fp":  False,
        "morph_flags":         "morph_unavailable",
    }
    if _MORPH_FILTERS_AVAILABLE:
        try:
            _mf = compute_morph_flags(
                time     = time_rs.astype(np.float64),
                flux     = dflux.astype(np.float64),
                period   = best_period,
                t0       = best_t0,
                duration = best_dur,
            )
            diag.update(morph_flags_to_dict(_mf))
        except Exception as _morph_exc:
            _MORPH_DEFAULTS["morph_flags"] = f"morph_error:{type(_morph_exc).__name__}"
            diag.update(_MORPH_DEFAULTS)
    else:
        diag.update(_MORPH_DEFAULTS)

    return float(best_score), diag, cand_rows


def score_target(model, payload: Tuple[np.ndarray, np.ndarray]) -> Tuple[float, Dict[str, Any]]:
    score, diag, _ = score_target_topk(model, payload)
    return float(score), diag


# =============================================================================
# Benchmark against confirmed planets (NASA Exoplanet Archive via astroquery)
# =============================================================================
def _require_astroquery_exoarchive():
    try:
        from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "astroquery (NASA Exoplanet Archive) import failed. "
            "Install/upgrade astroquery and its deps (and restart the kernel). "
            f"Original error: {type(e).__name__}: {e}"
        )


def fetch_confirmed_planets_from_exoarchive(limit: int = 2000) -> pd.DataFrame:
    """
    Fetch confirmed planets with TIC IDs from NASA Exoplanet Archive.

    Returns DataFrame with at least:
      pl_name, tic_id, pl_orbper, disc_facility
    """
    _require_astroquery_exoarchive()
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

    select_cols = ["pl_name", "tic_id", "pl_orbper", "disc_facility"]
    optional_cols = ["pl_trandep", "pl_rade", "sy_tmag", "hostname"]

    def _tab_to_df(tab) -> pd.DataFrame:
        try:
            return tab.to_pandas()
        except Exception:
            return pd.DataFrame(tab.as_array())

    def _try_query(table: str, select_list: List[str], use_where: bool) -> Optional[pd.DataFrame]:
        select_str = ",".join(select_list)

        where = "tic_id is not null AND pl_orbper is not null"
        if BENCHMARK_TESS_ONLY:
            where += " AND disc_facility like 'TESS%'"

        try:
            if use_where:
                tab = NasaExoplanetArchive.query_criteria(table=table, select=select_str, where=where)
            else:
                tab = NasaExoplanetArchive.query_criteria(table=table, select=select_str)
        except TypeError:
            return None
        except Exception as e:
            msg = str(e).lower()
            if ("unknown" in msg) or ("unrecognized" in msg) or ("invalid" in msg) or ("column" in msg):
                return None
            raise

        df = _tab_to_df(tab)
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=select_list)
        return df

    tables_to_try = ["pscomppars", "ps"]

    df = None
    for table in tables_to_try:
        df = _try_query(table, select_cols + optional_cols, use_where=True)
        if df is not None and len(df) > 0:
            break
        df = _try_query(table, select_cols, use_where=True)
        if df is not None and len(df) > 0:
            break

    if df is None or len(df) == 0:
        for table in tables_to_try:
            df = _try_query(table, select_cols + optional_cols, use_where=False)
            if df is not None and len(df) > 0:
                break
            df = _try_query(table, select_cols, use_where=False)
            if df is not None and len(df) > 0:
                break

    if df is None or len(df) == 0:
        raise RuntimeError("ExoArchive query_criteria returned no rows. Check network / service availability.")

    df = df.copy()

    if "tic_id" not in df.columns:
        raise RuntimeError(f"ExoArchive response missing 'tic_id'. Columns: {list(df.columns)}")

    tic_raw = df["tic_id"]
    tic_num = pd.to_numeric(tic_raw, errors="coerce")
    if tic_num.isna().all():
        s = tic_raw.astype(str)
        extracted = s.str.extract(r"(\d{5,12})", expand=False)
        tic_num = pd.to_numeric(extracted, errors="coerce")

    df["tic_id"] = tic_num.astype("Int64")

    if "pl_orbper" not in df.columns:
        raise RuntimeError(f"ExoArchive response missing 'pl_orbper'. Columns: {list(df.columns)}")

    df["pl_orbper"] = pd.to_numeric(df["pl_orbper"], errors="coerce")

    df = df[df["tic_id"].notna() & np.isfinite(df["pl_orbper"])].copy()

    if BENCHMARK_TESS_ONLY and "disc_facility" in df.columns:
        df["disc_facility"] = df["disc_facility"].astype(str)
        df = df[df["disc_facility"].str.upper().str.startswith("TESS")].copy()

    df = df.sort_values("pl_orbper", ascending=True)
    if limit is not None:
        df = df.head(int(limit)).copy()

    return df

def run_confirmed_benchmark(
    model,
    t_candidate: float,
    max_targets: int = 25,
    fail_fast: bool = True
) -> None:
    """
    Confirmed-planet benchmark runner.

    Reports:
      - BEST   (algorithm output): diag["bls_period"] etc.
      - ANY    (top-K retrieval): first harmonic/alias match in cand_rows (BLS rank order)
      - ORACLE (benchmark-only): among matched cand_rows, pick max CNN score_used (no mutation)

    Also outputs 'bench_promoted_long' analysis flag.
    """
    # ---- Planet list: local cache first, ExoArchive fallback ----
    # Cache file avoids a network call on every run and makes resume robust.
    # Delete tess_benchmark_planet_list.csv to force a fresh fetch.
    PLANET_LIST_CACHE = "tess_benchmark_planet_list.csv"
    if os.path.exists(PLANET_LIST_CACHE):
        print(f"[BENCH] Loading planet list from local cache: {PLANET_LIST_CACHE}")
        df = pd.read_csv(PLANET_LIST_CACHE)
        df["tic_id"] = pd.to_numeric(df["tic_id"], errors="coerce").astype("Int64")
        df["pl_orbper"] = pd.to_numeric(df["pl_orbper"], errors="coerce")
    else:
        print(f"[BENCH] No local cache found — fetching from NASA ExoArchive (requires network)...")
        df = fetch_confirmed_planets_from_exoarchive(limit=BENCHMARK_LIMIT)
        df.to_csv(PLANET_LIST_CACHE, index=False)
        print(f"[BENCH] Saved planet list to {PLANET_LIST_CACHE} ({len(df)} rows) — future runs use cache.")

    required = ["pl_name", "tic_id", "pl_orbper"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[BENCH] Missing required columns from ExoArchive: {missing}. Got: {list(df.columns)}")

    df = df.copy()
    df["tic_id"] = pd.to_numeric(df["tic_id"], errors="coerce").astype("Int64")
    df["pl_orbper"] = pd.to_numeric(df["pl_orbper"], errors="coerce")
    df = df[df["tic_id"].notna() & np.isfinite(df["pl_orbper"])].copy()

    if len(df) == 0:
        print("[BENCH] 0 rows after numeric filters. Tip: set BENCHMARK_TESS_ONLY=False.")
        return

    df["tic_str"] = df["tic_id"].apply(lambda x: f"TIC {int(x)}")

    if BENCHMARK_REQUIRE_2_TRANSITS:
        pmax = 0.5 * float(MAX_SPAN_DAYS)
        df = df[df["pl_orbper"] <= pmax].copy()

    if len(df) == 0:
        print("[BENCH] No rows after filters. Increase BENCHMARK_LIMIT or relax BENCHMARK_* filters.")
        return

    print(f"[BENCH] Confirmed planets to test: {len(df)}")

    if max_targets is not None and int(max_targets) > 0:
        df = df.iloc[:int(max_targets)].copy()
        print(f"[BENCH] Fail-fast mode: running first {len(df)} targets")

    prefer_spoc = ("SPOC" in [a.upper() for a in HARVEST_AUTHOR_WHITELIST]) if HARVEST_AUTHOR_WHITELIST else False

    # ---- Checkpoint / resume ----
    # Load already-completed targets from the output CSV so a crashed
    # run can be resumed by simply re-running the script.
    done_targets: set = set()
    rows: List[Dict[str, Any]] = []
    if os.path.exists(BENCHMARK_OUT_CSV):
        try:
            df_prev = pd.read_csv(BENCHMARK_OUT_CSV)
            done_targets = set(df_prev["target"].dropna().astype(str).values)
            rows = df_prev.to_dict("records")
            print(f"[BENCH] Resume: found {len(done_targets)} already-done targets in {BENCHMARK_OUT_CSV}")
        except Exception as e:
            print(f"[BENCH] Warning: could not load previous CSV for resume ({e}); starting fresh.")
            rows = []
            done_targets = set()

    n_total = len(df)
    n_remaining = len([r for r in df.itertuples() if str(getattr(r, "tic_str", "")) not in done_targets])
    print(f"[BENCH] Total in filtered list: {n_total}  |  Remaining (after resume): {n_remaining}")
    t_bench_start = time.time()

    def base_row(target: str, r, p_true: float, status: str) -> dict:
        return {
            "target": target,
            "pl_name": str(r.pl_name),
            "tic_id": int(r.tic_id),
            "p_true_days": float(p_true),
            "disc_facility": str(getattr(r, "disc_facility", "")),
            "download_status": status,
            "score": np.nan,

            # BEST (algorithm output)
            "p_pred_best_days": np.nan,
            "period_match_best": False,
            "best_match_k": np.nan,
            "best_match_kind": "",
            "best_match_rel_err": np.nan,
            "selection_mode": "",
            "alias_switched": False,
            "best_k": -1,
            "bls_power_best": np.nan,
            "bls_depth_best": np.nan,
            "tspan_days": np.nan,

            # ANY (top-K retrieval)
            "period_match_any": False,
            "any_match_k": np.nan,
            "any_match_kind": "",
            "any_match_rel_err": np.nan,
            "match_rank": np.nan,
            "p_pred_match_days": np.nan,

            # ORACLE (benchmark-only; no mutation)
            "oracle_best_k": np.nan,
            "oracle_p_pred_best_days": np.nan,
            "oracle_period_match_best": False,
            "oracle_match_k": np.nan,
            "oracle_match_kind": "",
            "oracle_match_rel_err": np.nan,

            # analysis-only
            "bench_promoted_long": False,
            "bench_promote_reason": "",
        }

    def _find_cand_by_k(cand_rows: List[Dict[str, Any]], k: int):
        for cr in cand_rows:
            if int(cr.get("k", -1)) == int(k):
                return cr
        return None

    try:
        for i, r in enumerate(df.itertuples(index=False), 1):
            target = str(r.tic_str)
            p_true = float(r.pl_orbper)

            # Resume: skip targets already in the output CSV
            if target in done_targets:
                continue

            payload, status = download_tess_lc_mainprocess(
                target,
                max_span_days=MAX_SPAN_DAYS,
                prefer_spoc=prefer_spoc,
            )

            if payload is None:
                rows.append(base_row(target, r, p_true, status))
                done_targets.add(target)
                # Flush incrementally every 10 targets so resume always has recent data
                if len(rows) % 10 == 0:
                    pd.DataFrame(rows).to_csv(BENCHMARK_OUT_CSV, index=False)
                elapsed = time.time() - t_bench_start
                n_done_this_run = len(done_targets) - (n_total - n_remaining)
                rate = n_done_this_run / max(elapsed, 1)
                eta_s = (n_remaining - n_done_this_run) / max(rate, 1e-9)
                if i % 10 == 0 or i == n_total:
                    print(f"[BENCH] {i}/{n_total} | done_this_run={n_done_this_run} "
                          f"| elapsed={elapsed/60:.1f}m | ETA={eta_s/60:.1f}m | status={status}")
                polite_sleep()
                continue

            try:
                score, diag, cand_rows = score_target_topk(model, payload)

                # -------------------------
                # BEST (algorithm output)
                # -------------------------
                p_pred_best = float(diag.get("bls_period", np.nan))
                best_ok, best_k, best_kind, best_err = period_match_harmonic(
                    p_pred_best, p_true,
                    frac_tol=BENCHMARK_PERIOD_FRAC_TOL,
                    max_harm=BENCHMARK_MAX_HARMONIC
                )

                # -------------------------
                # ANY (top-K retrieval): first match in BLS-rank order
                # -------------------------
                match_any = False
                match_rank = np.nan
                p_pred_match = np.nan
                any_k = np.nan
                any_kind = ""
                any_err = np.nan
                any_cr = None

                for cr in cand_rows:
                    p_k = float(cr.get("bls_period", np.nan))
                    ok, k, kind, err = period_match_harmonic(
                        p_k, p_true,
                        frac_tol=BENCHMARK_PERIOD_FRAC_TOL,
                        max_harm=BENCHMARK_MAX_HARMONIC
                    )
                    if ok:
                        match_any = True
                        match_rank = int(cr.get("k", -1))
                        p_pred_match = float(p_k)
                        any_k = float(k) if k is not None else np.nan
                        any_kind = str(kind) if kind is not None else ""
                        any_err = float(err)
                        any_cr = cr
                        break

                # -------------------------
                # ORACLE (benchmark-only): among matched, pick max CNN score_used
                # -------------------------
                oracle_best_k = np.nan
                oracle_p_pred_best = np.nan
                oracle_ok = False
                oracle_k = np.nan
                oracle_kind = ""
                oracle_err = np.nan
                oracle_cr = None

                oracle_best_cnn = float("-inf")
                for cr in cand_rows:
                    p_k = float(cr.get("bls_period", np.nan))
                    ok, k, kind, err = period_match_harmonic(
                        p_k, p_true,
                        frac_tol=BENCHMARK_PERIOD_FRAC_TOL,
                        max_harm=BENCHMARK_MAX_HARMONIC
                    )
                    if not ok:
                        continue
                    c = float(cr.get("score_used", np.nan))
                    if np.isfinite(c) and c > oracle_best_cnn:
                        oracle_best_cnn = c
                        oracle_cr = cr
                        oracle_k = float(k) if k is not None else np.nan
                        oracle_kind = str(kind) if kind is not None else ""
                        oracle_err = float(err)

                if oracle_cr is not None:
                    k_val = oracle_cr.get("k", np.nan)
                    oracle_best_k = int(k_val) if np.isfinite(k_val) else np.nan
                    oracle_p_pred_best = float(oracle_cr.get("bls_period", np.nan))
                    oracle_ok = True  # by construction (picked from matched set)

                # -------------------------
                # DEBUG: ANY-not-BEST (now includes cnn/power comparisons)
                # -------------------------
                if match_any and (not best_ok):
                    chosen_k = int(diag.get("best_k", -1))
                    chosen_cr = _find_cand_by_k(cand_rows, chosen_k)

                    def _fmt(cr):
                        if cr is None:
                            return "None"
                        return (f"k={int(cr.get('k',-1))} "
                                f"p={float(cr.get('bls_period',np.nan)):.6f} "
                                f"pow={float(cr.get('bls_power',np.nan)):.3g} "
                                f"cnn={float(cr.get('score_used',np.nan)):.3f}")

                    print(f"\n[ANY-not-BEST] target: {target} p_true: {p_true}")
                    print(" chosen:", _fmt(chosen_cr), " mode:", diag.get("selection_mode", ""))
                    print(" any   :", _fmt(any_cr), f" match_k={any_k} kind={any_kind} rel_err={any_err:.4g}")
                    print(" oracle:", _fmt(oracle_cr), f" match_k={oracle_k} kind={oracle_kind} rel_err={oracle_err:.4g}")

                    N = min(8, len(cand_rows))
                    for cr in cand_rows[:N]:
                        print(f"  k={int(cr.get('k',-1))}  "
                              f"p={float(cr.get('bls_period',np.nan)):.6f}  "
                              f"pow={float(cr.get('bls_power',np.nan)):.3g}  "
                              f"cnn={float(cr.get('score_used',np.nan)):.3f}")

                # -------------------------
                # Guarded "long alias plausible" flag (analysis only)
                # -------------------------
                bench_promoted_long = False
                bench_promote_reason = ""

                BENCH_PROMOTE_MIN_PERIOD   = 0.60
                BENCH_PROMOTE_CNN_MAX_DROP = 0.04
                BENCH_PROMOTE_POWER_FLOOR  = 0.30

                if oracle_cr is None:
                    bench_promote_reason = "no_match_in_topk"
                else:
                    p_best = float(diag.get("bls_period", np.nan))
                    p_new  = float(oracle_cr.get("bls_period", np.nan))

                    chosen_k = int(diag.get("best_k", -1))
                    chosen_cr = _find_cand_by_k(cand_rows, chosen_k)

                    if not (np.isfinite(p_best) and np.isfinite(p_new)):
                        bench_promote_reason = "nan_periods"
                    elif not ((p_new > p_best) and (p_new >= BENCH_PROMOTE_MIN_PERIOD)):
                        bench_promote_reason = "match_not_long_or_below_min_period"
                    else:
                        cnn_ok = True
                        pow_ok = True
                        if chosen_cr is not None:
                            cnn_best = float(chosen_cr.get("score_used", np.nan))
                            cnn_new  = float(oracle_cr.get("score_used", np.nan))
                            pow_best = float(chosen_cr.get("bls_power", np.nan))
                            pow_new  = float(oracle_cr.get("bls_power", np.nan))

                            if np.isfinite(cnn_best) and np.isfinite(cnn_new):
                                cnn_ok = (cnn_new >= cnn_best - BENCH_PROMOTE_CNN_MAX_DROP)
                            if np.isfinite(pow_best) and np.isfinite(pow_new):
                                pow_ok = (pow_new >= BENCH_PROMOTE_POWER_FLOOR * max(pow_best, 1e-12))

                        if cnn_ok and pow_ok:
                            bench_promoted_long = True
                            bench_promote_reason = "long_alias_plausible"
                        else:
                            bench_promote_reason = "long_alias_but_weak"

                # -------------------------
                # Write row
                # -------------------------
                row = base_row(target, r, p_true, "ok")
                row.update({
                    "score": float(score),

                    # BEST
                    "p_pred_best_days": float(p_pred_best) if np.isfinite(p_pred_best) else np.nan,
                    "period_match_best": bool(best_ok),
                    "best_match_k": float(best_k) if best_k is not None else np.nan,
                    "best_match_kind": str(best_kind) if best_kind is not None else "",
                    "best_match_rel_err": float(best_err),

                    "selection_mode": str(diag.get("selection_mode", "")),
                    "alias_switched": bool(diag.get("alias_switched", False)),
                    "best_k": int(diag.get("best_k", -1)) if np.isfinite(diag.get("best_k", np.nan)) else -1,
                    "bls_power_best": float(diag.get("bls_power", np.nan)),
                    "bls_depth_best": float(diag.get("bls_depth", np.nan)),
                    "tspan_days": float(diag.get("tspan_days", np.nan)),

                    # ANY
                    "period_match_any": bool(match_any),
                    "any_match_k": any_k,
                    "any_match_kind": str(any_kind),
                    "any_match_rel_err": float(any_err) if np.isfinite(any_err) else np.nan,
                    "match_rank": match_rank,
                    "p_pred_match_days": float(p_pred_match) if np.isfinite(p_pred_match) else np.nan,

                    # ORACLE
                    "oracle_best_k": oracle_best_k,
                    "oracle_p_pred_best_days": float(oracle_p_pred_best) if np.isfinite(oracle_p_pred_best) else np.nan,
                    "oracle_period_match_best": bool(oracle_ok),
                    "oracle_match_k": oracle_k,
                    "oracle_match_kind": str(oracle_kind),
                    "oracle_match_rel_err": float(oracle_err) if np.isfinite(oracle_err) else np.nan,

                    # analysis-only
                    "bench_promoted_long": bool(bench_promoted_long),
                    "bench_promote_reason": str(bench_promote_reason),
                })
                rows.append(row)
                done_targets.add(target)

            except Exception as e:
                row = base_row(target, r, p_true, "ok")
                row["error"] = f"{type(e).__name__}: {e}"
                rows.append(row)
                done_targets.add(target)

                if fail_fast:
                    import traceback
                    print("\n[FAIL-FAST] Target:", target, "p_true_days:", p_true)
                    print("[FAIL-FAST] download_status:", status)
                    traceback.print_exc()
                    raise

            # Incremental flush every 10 targets
            if len(rows) % 10 == 0:
                pd.DataFrame(rows).to_csv(BENCHMARK_OUT_CSV, index=False)

            elapsed = time.time() - t_bench_start
            n_done_this_run = len(done_targets) - (n_total - n_remaining)
            rate = max(n_done_this_run, 1) / max(elapsed, 1)
            eta_s = (n_remaining - n_done_this_run) / max(rate, 1e-9)
            if i % 10 == 0 or i == n_total:
                print(f"[BENCH] {i}/{n_total} | done_this_run={n_done_this_run} "
                      f"| elapsed={elapsed/60:.1f}m | ETA={eta_s/60:.1f}m")

            polite_sleep()

    finally:
        out = pd.DataFrame(rows)
        out.to_csv(BENCHMARK_OUT_CSV, index=False)
        print(f"✓ Wrote benchmark: {BENCHMARK_OUT_CSV} ({len(out)} rows)")

        # POST summary (same as you had; keep it)
        df_dbg = pd.read_csv(BENCHMARK_OUT_CSV)
        print("\n[POST] download_status counts:\n",
              df_dbg["download_status"].value_counts(dropna=False).head(20))

        ok_mask = df_dbg["download_status"].astype(str).str.startswith("ok")
        score_num = pd.to_numeric(df_dbg.get("score", np.nan), errors="coerce")
        finite_mask = np.isfinite(score_num)
        print("[POST] ok rows:", int(ok_mask.sum()),
              "finite-score rows:", int((ok_mask & finite_mask).sum()))

        ok_scored = df_dbg.loc[ok_mask & finite_mask].copy()
        if len(ok_scored) == 0:
            print("[BENCH] No successful scores; check downloads/model.")
            return

        detected = ok_scored[ok_scored["score"] >= float(t_candidate)]
        det_rate = len(detected) / len(ok_scored)

        pm_best = float(detected["period_match_best"].astype(bool).mean()) if len(detected) > 0 else 0.0
        pm_any  = float(detected["period_match_any"].astype(bool).mean()) if len(detected) > 0 else 0.0
        pm_orcl = float(detected["oracle_period_match_best"].astype(bool).mean()) if len(detected) > 0 else 0.0

        print("\n==============================")
        print("Confirmed-planet benchmark summary")
        print("==============================")
        print(f"Scored (download ok): {len(ok_scored)}")
        print(f"Detected (score >= {t_candidate:.3f}): {len(detected)}  ({det_rate:.3f})")
        print(f"Period-match BEST   among detected (algorithm): {pm_best:.3f}")
        print(f"Period-match ANY    among detected (top-K):     {pm_any:.3f}")
        print(f"Period-match ORACLE among detected (top-K):     {pm_orcl:.3f}")
        print("Score percentiles (scored):", np.nanpercentile(ok_scored["score"], [5, 50, 95]))

        if len(detected) > 0:
            print("Score percentiles (detected):", np.nanpercentile(detected["score"], [5, 50, 95]))
            mr = detected.loc[detected["period_match_any"].astype(bool), "match_rank"]
            if len(mr) > 0:
                print("Match-rank percentiles (among ANY-matched detections):", np.nanpercentile(mr, [5, 50, 95]))

            if "bench_promoted_long" in detected.columns:
                pr_ok = int(detected["bench_promoted_long"].astype(bool).sum())
                print(f"Guarded long-alias opportunities among detected: {pr_ok}/{len(detected)}")
# =============================================================================
# Sector harvesting via astroquery MAST Observations (stable)
# =============================================================================
def harvest_targets_for_sector(sector: int, done: set) -> List[str]:
    """
    FAST CAOM harvest: query server-side by sequence_number (sector).
    Returns ["TIC 123", ...] after filters + resume skip.
    """
    from astroquery.mast import Observations

    sector = int(sector)

    kwargs = dict(
        obs_collection="TESS",
        dataproduct_type="timeseries",
        sequence_number=sector,   # sector is sequence_number in CAOM
    )

    if HARVEST_AUTHOR_WHITELIST and len(HARVEST_AUTHOR_WHITELIST) == 1:
        kwargs["provenance_name"] = HARVEST_AUTHOR_WHITELIST[0]

    try:
        obs = Observations.query_criteria(**kwargs)
    except Exception as e:
        print(f"[HARVEST] sector={sector}: CAOM query failed: {type(e).__name__}: {e}")
        return []

    if obs is None or len(obs) == 0:
        print(f"[HARVEST] sector={sector}: no CAOM rows returned")
        return []

    try:
        df = obs.to_pandas()
    except Exception:
        df = pd.DataFrame(obs.as_array())

    if df is None or len(df) == 0:
        print(f"[HARVEST] sector={sector}: empty DataFrame")
        return []

    if HARVEST_DEBUG_SCHEMA:
        print(f"[HARVEST] sector={sector}: rows={len(df)} cols={list(df.columns)}")
        if "provenance_name" in df.columns:
            pv = df["provenance_name"].astype(str).str.upper().value_counts().head(10)
            print("[HARVEST] provenance sample:", dict(pv))
        if "t_exptime" in df.columns:
            vals = pd.to_numeric(df["t_exptime"], errors="coerce")
            finite = vals[np.isfinite(vals)]
            if len(finite) > 0:
                sample = sorted(set(int(round(x)) for x in finite.values))[:25]
                print("[HARVEST] t_exptime sample rounded seconds:", sample)

    if HARVEST_EXPTIME_SECONDS:
        exset = set(int(x) for x in HARVEST_EXPTIME_SECONDS)
        if "t_exptime" in df.columns:
            vals = pd.to_numeric(df["t_exptime"], errors="coerce")
            m = np.isfinite(vals) & vals.round().astype("Int64").isin(exset)
            df = df[m].copy()

    if len(df) == 0:
        print(f"[HARVEST] sector={sector}: 0 rows after filters")
        return []

    def extract_tic(s: str):
        if s is None:
            return None
        s0 = str(s)
        su = s0.upper()

        m = re.search(r"\bTIC\W*(\d{5,12})\b", su)
        if m:
            return int(m.group(1))
        m = re.search(r"\bTIC(\d{5,12})\b", su)
        if m:
            return int(m.group(1))

        runs = re.findall(r"(\d{9,16})", s0)
        if runs:
            runs = sorted(runs, key=len, reverse=True)
            for r in runs:
                tid = int(r.lstrip("0") or "0")
                if 100000 <= tid <= 2000000000:
                    return tid

        runs2 = re.findall(r"(\d{7,12})", s0)
        if runs2:
            for r in runs2:
                tid = int(r.lstrip("0") or "0")
                if 100000 <= tid <= 2000000000:
                    return tid
        return None

    tic_ids = []
    cols_to_try = [c for c in ["target_name", "obs_id", "obs_title", "dataURL"] if c in df.columns]
    for c in cols_to_try:
        for s in df[c].astype(str).values:
            tid = extract_tic(s)
            if tid is not None:
                tic_ids.append(int(tid))

    for c in ["targetid", "tic_id", "ticid", "objectid", "objID"]:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce")
            vals = vals[np.isfinite(vals)]
            tic_ids.extend([int(v) for v in vals.values if int(v) > 0])

    tic_ids = sorted(set(tic_ids))

    if len(tic_ids) == 0:
        print(f"[HARVEST] sector={sector}: found 0 TIC IDs after parsing.")
        for c in cols_to_try:
            print(f"[HARVEST] sample {c}:")
            for s in df[c].astype(str).head(5).tolist():
                print("   ", s[:200])
        return []

    if TARGETS_PER_SECTOR is not None:
        tic_ids = tic_ids[:int(TARGETS_PER_SECTOR)]

    targets = [f"TIC {tid}" for tid in tic_ids]

    before = len(targets)
    targets = [t for t in targets if t not in done]
    skipped = before - len(targets)

    print(f"[HARVEST] sector={sector}: TICs={len(tic_ids)} kept={len(targets)} resume_skip={skipped}")
    return targets


# =============================================================================
# Main
# =============================================================================
def main():
    print(f"[PID] main pid={os.getpid()}")

    t_candidate, t_likely, t_high = load_thresholds(THRESHOLDS_JSON)

    global WRITE_SHORTLIST_SCORE, WRITE_HIGH_SCORE
    WRITE_SHORTLIST_SCORE = float(t_candidate)
    WRITE_HIGH_SCORE = float(t_high)

    model = keras.models.load_model(MODEL_PATH)
    print(f"✓ Loaded model: {MODEL_PATH}")

    globals()["model"] = model

    import inspect
    print("score_target_topk:", inspect.getsourcefile(score_target_topk), inspect.getsourcelines(score_target_topk)[1])
    print("run_confirmed_benchmark:", inspect.getsourcefile(run_confirmed_benchmark), inspect.getsourcelines(run_confirmed_benchmark)[1])

    # --- Benchmark mode (Reviewer #1) ---
    if BENCHMARK_CONFIRMED:
        run_confirmed_benchmark(model, t_candidate, max_targets=None, fail_fast=False)
        return

    done = load_done_targets()

    t_wall0 = time.time()
    wall_limit = float(MAX_WALLTIME_HOURS) * 3600.0

    timeouts_in_row = 0
    candidates_high: List[Dict[str, Any]] = []
    candidates_short: List[Dict[str, Any]] = []
    all_scored: List[Dict[str, Any]] = []
    run_rows: List[Dict[str, Any]] = []

    if USE_TEST_TARGETS:
        print(f"✓ Using TEST_TARGETS: {len(TEST_TARGETS)}")
        sector = -1
        targets = TEST_TARGETS

        prefer_spoc = ("SPOC" in [a.upper() for a in HARVEST_AUTHOR_WHITELIST]) if HARVEST_AUTHOR_WHITELIST else False

        for i, target in enumerate(targets, 1):
            if (time.time() - t_wall0) >= wall_limit:
                print(f"[STOP] hit walltime limit ({MAX_WALLTIME_HOURS}h)")
                break

            payload, status = download_tess_lc_mainprocess(
                target,
                max_span_days=MAX_SPAN_DAYS,
                prefer_spoc=prefer_spoc,
            )
            print(f"[DL] {i}/{len(targets)} {target} -> {status}  payload={'yes' if payload is not None else 'no'}")

            if payload is None:
                append_checkpoint_row(CHECKPOINT_CSV, {
                    "sector": sector, "target": target, "score": np.nan,
                    "label": "download_fail", "download_status": status,
                })
                polite_sleep()
                continue

            try:
                score, diag = score_target(model, payload)
            except Exception as e:
                print(f"[SCORE ERROR] {target}: {type(e).__name__}: {e}")
                append_checkpoint_row(CHECKPOINT_CSV, {
                    "sector": sector, "target": target, "score": np.nan,
                    "label": "score_fail", "download_status": "ok",
                })
                polite_sleep()
                continue

            label = label_scores(np.array([score]), t_candidate, t_likely, t_high)[0]
            row = {
                "sector": sector, "target": target, "score": float(score),
                "label": label, "download_status": "ok",
                **{k: diag.get(k, np.nan) for k in diag.keys()}
            }
            # Morph fields (already in diag via score_target_topk, but
            # ensure correct types — None stays None, not np.nan)
            for _mkey in ("morph_v_score", "morph_asymmetry", "morph_secondary_dip",
                          "morph_depth_cv", "morph_duty_cycle", "morph_grazing_ratio"):
                if row.get(_mkey) is None:
                    row[_mkey] = None
            row["morph_grazing_flag"]  = bool(diag.get("morph_grazing_flag", False))
            _nf = diag.get("morph_n_flags")
            row["morph_n_flags"]      = int(_nf) if _nf is not None else -1
            row["morph_is_strong_fp"] = bool(diag.get("morph_is_strong_fp", False))
            row["morph_flags"]        = str(diag.get("morph_flags", "none"))

            append_checkpoint_row(CHECKPOINT_CSV, row)
            all_scored.append(row)
            run_rows.append(row)

            if score >= WRITE_SHORTLIST_SCORE:
                candidates_short.append(row)
            if score >= WRITE_HIGH_SCORE:
                candidates_high.append(row)

            polite_sleep()

    else:
        next_sector = load_sector_progress(SECTOR_PROGRESS_JSON, SECTOR_START)
        print(f"✓ Starting at sector {next_sector} (progress: {SECTOR_PROGRESS_JSON})")

        empty_sectors_in_row = 0

        while True:
            if (time.time() - t_wall0) >= wall_limit:
                print(f"[STOP] Hit walltime limit ({MAX_WALLTIME_HOURS}h). Saving progress & exiting.")
                save_sector_progress(SECTOR_PROGRESS_JSON, next_sector)
                break

            sec0 = int(next_sector)
            batch_sectors = list(range(sec0, min(SECTOR_END + 1, sec0 + SECTORS_PER_BATCH)))
            batch_targets: List[Tuple[int, str]] = []

            for sector in batch_sectors:
                print(f"\n[SECTOR] {sector}: harvesting...")
                sector_targets = harvest_targets_for_sector(sector, done)
                print(f"[SECTOR] {sector}: harvested {len(sector_targets)} targets (post-resume)")

                if len(sector_targets) == 0:
                    empty_sectors_in_row += 1
                    print(f"[SECTOR] {sector}: empty (empty_sectors_in_row={empty_sectors_in_row})")
                else:
                    empty_sectors_in_row = 0

                for t in sector_targets:
                    batch_targets.append((sector, t))

                next_sector = sector + 1
                save_sector_progress(SECTOR_PROGRESS_JSON, next_sector)

                if empty_sectors_in_row >= EMPTY_SECTOR_STOP:
                    print(f"[STOP] {EMPTY_SECTOR_STOP} empty sectors in a row -> stopping.")
                    batch_targets = []
                    break

            if not batch_targets:
                if empty_sectors_in_row >= EMPTY_SECTOR_STOP:
                    break
                continue

            random.shuffle(batch_targets)
            print(f"\n[BATCH] scoring {len(batch_targets)} targets from sectors {batch_sectors}...")

            prefer_spoc = ("SPOC" in [a.upper() for a in HARVEST_AUTHOR_WHITELIST]) if HARVEST_AUTHOR_WHITELIST else False

            for i, (sector, target) in enumerate(batch_targets, 1):
                if (time.time() - t_wall0) >= wall_limit:
                    print(f"[STOP] Hit walltime limit ({MAX_WALLTIME_HOURS}h) mid-batch. Exiting cleanly.")
                    save_sector_progress(SECTOR_PROGRESS_JSON, next_sector)
                    break

                payload, status = download_tess_lc_mainprocess(
                    target,
                    max_span_days=MAX_SPAN_DAYS,
                    prefer_spoc=prefer_spoc
                )
                print(f"[DL] S{sector} {i}/{len(batch_targets)} {target} -> {status}  payload={'yes' if payload is not None else 'no'}")

                if status == "ok":
                    timeouts_in_row = 0
                elif str(status).startswith(("timeout", "hardcap_timeout")):
                    timeouts_in_row += 1
                else:
                    timeouts_in_row = 0

                if timeouts_in_row >= MAX_CONSECUTIVE_TIMEOUTS:
                    print(f"[ABORT] {timeouts_in_row} timeouts in a row -> stopping run.")
                    break

                if payload is None:
                    append_checkpoint_row(CHECKPOINT_CSV, {
                        "sector": sector, "target": target, "score": np.nan,
                        "label": "download_fail", "download_status": status,
                    })
                    polite_sleep()
                    continue

                try:
                    score, diag = score_target(model, payload)
                except Exception as e:
                    print(f"[SCORE ERROR] {target}: {type(e).__name__}: {e}")
                    append_checkpoint_row(CHECKPOINT_CSV, {
                        "sector": sector, "target": target, "score": np.nan,
                        "label": "score_fail", "download_status": "ok",
                    })
                    polite_sleep()
                    continue

                label = label_scores(np.array([score]), t_candidate, t_likely, t_high)[0]

                row = {
                    "sector": sector,
                    "target": target,
                    "score": float(score),
                    "label": label,
                    "download_status": "ok",

                    "tspan_days": float(diag.get("tspan_days", np.nan)),
                    "median_dt_days": float(diag.get("median_dt_days", np.nan)),

                    "bls_period": float(diag.get("bls_period", np.nan)),
                    "bls_t0": float(diag.get("bls_t0", np.nan)),
                    "bls_duration": float(diag.get("bls_duration", np.nan)),
                    "bls_depth": float(diag.get("bls_depth", np.nan)),
                    "bls_power": float(diag.get("bls_power", np.nan)),

                    "polarity_flipped": diag.get("polarity_flipped", np.nan),
                    "score_dip": float(diag.get("score_dip", np.nan)),
                    "score_bump": float(diag.get("score_bump", np.nan)),
                    "score_used": float(diag.get("score_used", np.nan)),
                    "score_injected": float(diag.get("score_injected", np.nan)),
                    "score_delta_injected": float(diag.get("score_delta_injected", np.nan)),

                    "w_nan_frac": float(diag.get("w_nan_frac", np.nan)),
                    "w_std": float(diag.get("w_std", np.nan)),
                    "min_abs_phase": float(diag.get("min_abs_phase", np.nan)),

                    # Morphology filter scores
                    "morph_v_score":       diag.get("morph_v_score"),
                    "morph_asymmetry":     diag.get("morph_asymmetry"),
                    "morph_secondary_dip": diag.get("morph_secondary_dip"),
                    "morph_depth_cv":      diag.get("morph_depth_cv"),
                    "morph_duty_cycle":    diag.get("morph_duty_cycle"),
                    "morph_grazing_ratio": diag.get("morph_grazing_ratio"),
                    "morph_grazing_flag":  bool(diag.get("morph_grazing_flag", False)),
                    "morph_n_flags":       int(_nf) if (_nf := diag.get("morph_n_flags")) is not None else -1,
                    "morph_is_strong_fp":  bool(diag.get("morph_is_strong_fp", False)),
                    "morph_flags":         str(diag.get("morph_flags", "none")),
                }

                append_checkpoint_row(CHECKPOINT_CSV, row)
                all_scored.append(row)
                run_rows.append(row)
                done.add(str(target).strip())

                if score >= WRITE_SHORTLIST_SCORE:
                    candidates_short.append(row)
                if score >= WRITE_HIGH_SCORE:
                    candidates_high.append(row)

                if i % 25 == 0:
                    print(f"[S2] batch progress {i}/{len(batch_targets)}")

                polite_sleep()

    # =========================
    # Outputs
    # =========================
    if all_scored:
        all_scored_sorted = sorted(all_scored, key=lambda r: r.get("score", -np.inf), reverse=True)
        topN = all_scored_sorted[:min(len(all_scored_sorted), WRITE_TOP_N)]
        write_candidates(OUT_TOP_CSV, topN)
    else:
        print("No targets produced a score (unexpected if payload=yes).")

    if candidates_short:
        candidates_short = sorted(candidates_short, key=lambda r: r["score"], reverse=True)
        write_candidates(OUT_SHORTLIST_CSV, candidates_short)
    else:
        print(f"No candidates above WRITE_SHORTLIST_SCORE={WRITE_SHORTLIST_SCORE:.2f}.")

    if candidates_high:
        candidates_high = sorted(candidates_high, key=lambda r: r["score"], reverse=True)
        write_candidates(OUT_HIGH_CSV, candidates_high)
    else:
        print(f"No candidates above WRITE_HIGH_SCORE={WRITE_HIGH_SCORE:.2f}.")

    if run_rows:
        df_run = pd.DataFrame(run_rows).sort_values("score", ascending=False)
        df_sum = df_run[df_run["score"] >= float(WRITE_SHORTLIST_SCORE)].copy()
        df_sum.to_csv(RUN_SUMMARY_CSV, index=False)
        print(f"✓ Wrote run summary: {RUN_SUMMARY_CSV} ({len(df_sum)} candidate+ rows)")
    else:
        print("No rows scored this run, so no run summary written.")

    print("Done.")


if __name__ == "__main__":
    mp.freeze_support()
    main()