# -*- coding: utf-8 -*-
"""
TESS Phase 2 (REVISED, Option A): Real-time BLS-centering + periodic transit injection

Goal:
  Build Phase 2 training windows that MATCH Phase 5 inference behavior:
    - detrend (moving median) -> BLS on time axis (days)
    - fold-and-center window via interpolation on phase grid
    - per-window z-score
  Labels are clean because positives are synthetic injections into real light curves.

Input:
  - tess_phase1_dataset.pkl  (recommended) OR tess_phase1_dataset_combined.pkl

Output:
  - tess_phase2_windows.npz : X (N, WINDOW_LEN, 1), y (N,)
    plus injection metadata + diagnostics arrays

Notes:
  - Phase 1 pickle contains only flux arrays (no timestamps). We therefore
    build a synthetic time axis spanning BASELINE_DAYS.
  - This aligns training with Phase 5 Option A (BLS on "days", fold window on "days").
"""

import time
import pickle
import numpy as np
from typing import Optional, Tuple, Dict, Any, List

from astropy.timeseries import BoxLeastSquares

# ----------------------------
# Config
# ----------------------------
PHASE1_PICKLE = "tess_phase1_dataset.pkl"
OUT_NPZ = "tess_phase2_windows.npz"

TARGET_LENGTH = 2001
WINDOW_LEN = 301

# Synthetic baseline for Phase 1 arrays (typical one TESS sector ~27.4 days).
# If your Phase 1 set tends to include 2 sectors, set this to ~54.8.
BASELINE_DAYS = 27.4

# Detrend (must match Phase 5)
DETREND_WIN = 101  # must be odd
CLIP_SIGMA = 8.0
MIN_FINITE_FRAC = 0.98

# Dataset size / balancing
SEED = 42
AUG_PER_LC = 6          # windows per light curve
P_INJECT = 0.50         # fraction of windows labeled positive

# Injection (periodic)
PERIOD_MIN_DAYS = 0.5
PERIOD_MAX_DAYS = 30.0  # further clipped to <= 0.5 * BASELINE_DAYS at runtime

# duration as fraction of period (then clipped to sane bounds)
DUR_FRAC_MIN = 0.01
DUR_FRAC_MAX = 0.08
DUR_MIN_DAYS = 1.0 / 24.0   # 1 hour
DUR_MAX_DAYS = 0.5          # 12 hours cap

# depth in "sigma of local window" units
SIGMA_MULT_RANGE = (2.0, 10.0)

# BLS grid (real-time)
N_PERIODS = 2000
N_DUR = 25
BLS_OVERSAMPLE = 5
TOPK_BLS = 5  # we’ll take best of top-k by BLS power, but you can keep small for speed

# ----------------------------
# Helpers (match Phase 5)
# ----------------------------
def ensure_odd(k: int) -> int:
    k = int(k)
    return k if (k % 2 == 1) else (k + 1)

DETREND_WIN = ensure_odd(DETREND_WIN)

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
    if s == 0 or (not np.isfinite(s)):
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

# ----------------------------
# Phase-fold window (match Phase 5)
# ----------------------------
def fold_and_center_window(
    t: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    window_len: int = WINDOW_LEN
) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    flux = np.asarray(flux, dtype=float)

    m = np.isfinite(t) & np.isfinite(flux)
    t = t[m]
    flux = flux[m]
    if t.size < 10:
        return np.full(window_len, np.nan, dtype=np.float32)

    phase = ((t - t0 + 0.5 * period) % period) / period - 0.5  # [-0.5, 0.5)

    o = np.argsort(phase)
    phase = phase[o]
    flux = flux[o]

    phase_grid = np.linspace(-0.5, 0.5, window_len, endpoint=False, dtype=np.float64)
    w = np.interp(phase_grid, phase, flux).astype(np.float32)
    return w

# ----------------------------
# BLS on real time axis
# ----------------------------
def bls_top_candidates(
    t: np.ndarray,
    flux: np.ndarray,
    top_k: int = TOPK_BLS,
    period_min: float = PERIOD_MIN_DAYS,
    period_max: Optional[float] = None,
    n_periods: int = N_PERIODS,
    dur_min_days: float = 0.5/24.0,
    dur_max_days: float = 12.0/24.0,
    n_dur: int = N_DUR,
    oversample: int = BLS_OVERSAMPLE,
) -> List[Tuple[float, float, float, float, float]]:
    t = np.asarray(t, dtype=float)
    y = np.asarray(flux, dtype=float)

    m = np.isfinite(t) & np.isfinite(y)
    t = t[m]; y = y[m]
    if t.size < 200:
        return []

    o = np.argsort(t)
    t = t[o]; y = y[o]

    tspan = float(t[-1] - t[0])
    if not np.isfinite(tspan) or tspan <= 0:
        return []

    if period_max is None:
        period_max = 0.5 * tspan

    period_min = float(max(0.1, period_min))
    period_max = float(max(period_max, period_min * 2.0))

    periods = np.exp(np.linspace(np.log(period_min), np.log(period_max), int(n_periods))).astype(float)

    durations = np.geomspace(dur_min_days, dur_max_days, int(n_dur)).astype(float)
    durations = durations[durations < 0.8 * period_min]
    if durations.size < 3:
        return []

    bls = BoxLeastSquares(t, y)
    res = bls.power(periods, durations, oversample=int(oversample))

    power = np.asarray(res.power, dtype=float)
    if power.size == 0 or not np.any(np.isfinite(power)):
        return []

    idx = np.argsort(power)[::-1]
    out = []
    for k in idx[:max(1, int(top_k)) * 6]:
        out.append((float(res.period[k]), float(res.transit_time[k]),
                    float(res.duration[k]), float(res.depth[k]), float(res.power[k])))
        if len(out) >= top_k:
            break
    return out


# ----------------------------
# Periodic transit injection into FULL detrended light curve
# ----------------------------
def inject_periodic_box_transits(
    t: np.ndarray,
    flux: np.ndarray,
    period_days: float,
    t0_days: float,
    duration_days: float,
    sigma_mult: float,
) -> np.ndarray:
    """
    Inject periodic box-shaped dips into flux. Depth is defined in sigma units of flux.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(flux, dtype=float)

    y2 = y.copy()

    # Depth in units of sigma of the current series (already detrended and scaled ~1)
    s = float(np.nanstd(y2))
    if (not np.isfinite(s)) or s == 0:
        s = 1.0
    depth = float(sigma_mult) * s

    # Compute phase distance to nearest transit center
    # Transit occurs when |((t - t0 + 0.5P) mod P) - 0.5P| <= dur/2
    P = float(period_days)
    dt = ((t - t0_days + 0.5 * P) % P) - 0.5 * P
    in_transit = np.abs(dt) <= (0.5 * float(duration_days))

    y2[in_transit] -= depth
    return y2

# ----------------------------
# Load Phase 1
# ----------------------------
np.random.seed(SEED)

with open(PHASE1_PICKLE, "rb") as f:
    d = pickle.load(f)

# Use only nonplanets as "real backgrounds" (clean)
if "X_nonplanets" in d:
    X_bg = d["X_nonplanets"]
elif "X" in d and "y" in d:
    X = d["X"]
    y = d["y"]
    X_bg = X[y == 0]
else:
    raise ValueError("Phase 1 pickle missing expected keys: X_nonplanets or (X,y).")

print("Loaded Phase 1 backgrounds:")
print("  nonplanets:", len(X_bg))

# Synthetic time axis (days)
# Keep strictly monotonic and spanning BASELINE_DAYS
t_base = np.linspace(0.0, float(BASELINE_DAYS), int(TARGET_LENGTH), endpoint=False, dtype=np.float64)

# Period max clipped to ensure multiple transits
P_MAX = float(min(PERIOD_MAX_DAYS, 0.5 * BASELINE_DAYS))
if P_MAX <= PERIOD_MIN_DAYS:
    raise ValueError("Invalid period range: increase BASELINE_DAYS or reduce PERIOD_MIN_DAYS.")

# ----------------------------
# Build Phase 2 windows
# ----------------------------
X_out: List[np.ndarray] = []
y_out: List[int] = []

# metadata
inj_period = []
inj_t0 = []
inj_duration = []
inj_sigma_mult = []

# diagnostics
bls_period = []
bls_t0 = []
bls_duration = []
bls_depth = []
bls_power = []

# ----------------------------
# Option 3 (added): injection recovery diagnostic
# ----------------------------
inj_recover_frac_err = []  # |P_bls - P_inj| / P_inj for injected samples, else NaN

t_start = time.time()

for i in range(len(X_bg)):
    flux0 = np.asarray(X_bg[i], dtype=float)
    if flux0.size != TARGET_LENGTH:
        # pad/trim to TARGET_LENGTH if needed
        if flux0.size < TARGET_LENGTH:
            tmp = np.full(TARGET_LENGTH, np.nan, dtype=float)
            tmp[:flux0.size] = flux0
            flux0 = tmp
        else:
            flux0 = flux0[:TARGET_LENGTH]

    # Phase 5-ish detrend
    dflux = detrend_flux(flux0, win=DETREND_WIN)

    for _ in range(AUG_PER_LC):
        injected = (np.random.rand() < P_INJECT)

        if injected:
            # sample period log-uniform for diversity
            logP = np.random.uniform(np.log(PERIOD_MIN_DAYS), np.log(P_MAX))
            P = float(np.exp(logP))

            # sample duration proportional to period
            dur_frac = float(np.random.uniform(DUR_FRAC_MIN, DUR_FRAC_MAX))
            dur = float(np.clip(dur_frac * P, DUR_MIN_DAYS, DUR_MAX_DAYS))

            # random epoch within one period
            t0 = float(np.random.uniform(0.0, P))

            sm = float(np.random.uniform(*SIGMA_MULT_RANGE))

            dflux_inj = inject_periodic_box_transits(t_base, dflux, P, t0, dur, sm)

            inj_period.append(P)
            inj_t0.append(t0)
            inj_duration.append(dur)
            inj_sigma_mult.append(sm)
            y_out.append(1)
        else:
            dflux_inj = dflux
            inj_period.append(np.nan)
            inj_t0.append(np.nan)
            inj_duration.append(np.nan)
            inj_sigma_mult.append(np.nan)
            y_out.append(0)

        # Run BLS on (t_base, dflux_inj) just like Phase 5
        cands = bls_top_candidates(
            t_base,
            dflux_inj,
            top_k=TOPK_BLS,
            period_min=PERIOD_MIN_DAYS,
            period_max=P_MAX,
            n_periods=N_PERIODS,
            dur_min_days=DUR_MIN_DAYS,
            dur_max_days=DUR_MAX_DAYS,
            n_dur=N_DUR,
            oversample=BLS_OVERSAMPLE,
        )

        if not cands:
            # If BLS fails, skip this sample (rare)
            # Keep arrays aligned by popping metadata we just appended
            y_out.pop()
            inj_period.pop(); inj_t0.pop(); inj_duration.pop(); inj_sigma_mult.pop()
            continue

        # Take best power candidate (first in list by our selection)
        p_bls, t0_bls, dur_bls, dep_bls, pow_bls = cands[0]

        # ----------------------------
        # Option 3: record recovery error
        # ----------------------------
        if injected:
            inj_recover_frac_err.append(abs(float(p_bls) - float(P)) / float(P))
        else:
            inj_recover_frac_err.append(np.nan)

        # Build window by fold+interpolate (Phase 5)
        w_raw = fold_and_center_window(t_base, dflux_inj, float(p_bls), float(t0_bls), window_len=WINDOW_LEN)

        # z-score for CNN input (matches Phase 3 training input)
        w = zscore_window(w_raw)
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        if (len(w) != WINDOW_LEN) or (not np.all(np.isfinite(w))):
            # skip sample, keep metadata aligned
            y_out.pop()
            inj_period.pop(); inj_t0.pop(); inj_duration.pop(); inj_sigma_mult.pop()
            # IMPORTANT: also pop recovery diagnostic to stay aligned
            inj_recover_frac_err.pop()
            continue

        X_out.append(w)

        bls_period.append(float(p_bls))
        bls_t0.append(float(t0_bls))
        bls_duration.append(float(dur_bls))
        bls_depth.append(float(dep_bls))
        bls_power.append(float(pow_bls))

    if (i + 1) % 50 == 0:
        elapsed = time.time() - t_start
        print(f"Background LCs processed: {i+1}/{len(X_bg)}  samples={len(X_out)}  elapsed={elapsed:.1f}s")

X_out = np.asarray(X_out, dtype=np.float32)
y_out = np.asarray(y_out, dtype=np.int64)

# reshape for CNN
X_out = X_out[..., None]

print("\nPhase 2 (Option A periodic injection) complete:")
print("  X_out:", X_out.shape)
print("  y_out counts:", {0: int(np.sum(y_out == 0)), 1: int(np.sum(y_out == 1))})
print("  elapsed sec:", round(time.time() - t_start, 1))

np.savez_compressed(
    OUT_NPZ,
    X=X_out,
    y=y_out,

    # time axis metadata
    target_length=int(TARGET_LENGTH),
    window_len=int(WINDOW_LEN),
    baseline_days=float(BASELINE_DAYS),

    # detrend metadata
    detrend_win=int(DETREND_WIN),
    clip_sigma=float(CLIP_SIGMA),
    min_finite_frac=float(MIN_FINITE_FRAC),

    # injection metadata
    aug_per_lc=int(AUG_PER_LC),
    p_inject=float(P_INJECT),
    period_min_days=float(PERIOD_MIN_DAYS),
    period_max_days=float(P_MAX),
    dur_frac_min=float(DUR_FRAC_MIN),
    dur_frac_max=float(DUR_FRAC_MAX),
    sigma_mult_range=np.array(SIGMA_MULT_RANGE, dtype=np.float32),

    inj_period=np.array(inj_period, dtype=np.float32),
    inj_t0=np.array(inj_t0, dtype=np.float32),
    inj_duration=np.array(inj_duration, dtype=np.float32),
    inj_sigma_mult=np.array(inj_sigma_mult, dtype=np.float32),

    # Option 3 diagnostic
    inj_recover_frac_err=np.array(inj_recover_frac_err, dtype=np.float32),

    # BLS diagnostics (what centering used)
    bls_period=np.array(bls_period, dtype=np.float32),
    bls_t0=np.array(bls_t0, dtype=np.float32),
    bls_duration=np.array(bls_duration, dtype=np.float32),
    bls_depth=np.array(bls_depth, dtype=np.float32),
    bls_power=np.array(bls_power, dtype=np.float32),
)

print(f"✓ Saved Phase 2 windows: {OUT_NPZ}")
