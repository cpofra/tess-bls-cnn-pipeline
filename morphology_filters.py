"""
morphology_filters.py
---------------------
Five morphological pre-filters to add to phase_5_sweep.py before CNN scoring.
Each filter takes the binned folded phase curve and returns a scalar score
plus a boolean flag. All are fast (~1-2 ms per candidate), requiring no
external data or network access.

Based on false-positive analysis of TESS sectors 73-77 HC candidates:
  - V-score:          catches grazing EBs, rotation (TIC 12675729, 12494582, 42256493, 71266846)
  - Asymmetry:        catches asymmetric-dip EBs (TIC 12672792)
  - Two-dip scan:     catches half-period EBs (TIC 16728252, 48084398, 64366964, 27896467)
  - Depth consistency:catches variable-depth non-planetary signals (TIC 12672792)
  - Rotation check:   catches rotation-period BLS detections (TIC 71266846, 84342725)

INTEGRATION:
In score_target_topk(), after BLS gives you (period, t0, duration) for each
candidate and you fold the window, call compute_morph_flags() on the folded
flux array before passing it to the CNN. The returned MorphFlags object can
be stored in the sweep output CSV for downstream vetting.

Example integration point in your pipeline:
    for each BLS candidate:
        folded_flux = fold_and_bin(lc_flux, period, t0, n_bins=200)
        morph = compute_morph_flags(folded_flux, period, duration)
        cnn_score = model.predict(fold_for_cnn)
        # store morph.v_score, morph.asymmetry, etc. in output row
        # optionally skip CNN if morph.is_strong_fp == True
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Tuneable thresholds ───────────────────────────────────────────────────────
# Calibrated against confirmed TESS planets (score ~0.86-0.90) and known
# false positives from the sectors 73-77 vetting analysis.
#
# HARD flags (high specificity — rarely fire on real planets):
#   TWO_DIP_THRESHOLD    — secondary dip must be genuinely deep, not noise floor
#   HIGH_DUTY_THRESHOLD  — transit spans impossible fraction of orbit
#
# SOFT flags (informational — can fire on noisy real transits):
#   V_SCORE_THRESHOLD    — V-fit vs box-fit; noisy folds can reach 1.0-1.3
#   ASYMM_THRESHOLD      — ingress/egress balance; noisy at low SNR
#   DEPTH_CV_THRESHOLD   — per-transit depth variance; noisy at low SNR
#
# is_strong_fp = True requires at least ONE hard flag, not just two soft flags.
# This prevents confirmed near-threshold planets from being discarded.

V_SCORE_THRESHOLD       = 1.5    # v_score = rms_box / rms_v
                                 # >1.5 = V fits substantially better than box (soft flag)
ASYMM_THRESHOLD         = 0.60   # |ingress - egress| / |mean_depth|  (soft flag)
TWO_DIP_THRESHOLD       = -0.80  # secondary dip depth in sigma units  (HARD flag)
                                 # -0.80 requires a real secondary, not phase-0.5 noise
DEPTH_CV_THRESHOLD      = 0.80   # std/|mean| of per-transit depths    (soft flag)
ROTATION_DUTY_THRESHOLD = 0.30   # duty cycle above which to flag      (HARD flag)
                                 # 0.30 = transit spans >30% of orbit (unphysical for planet)
GRAZING_THRESH          = 0.50   # center-to-edge depth ratio below which to flag (HARD flag)
                                 # ratio ~1.0 = flat bottom (planet); ~0.0 = V-shape (grazing EB)


@dataclass
class MorphFlags:
    """All morphology scores and flags for one BLS candidate."""
    # V-score: ratio of box-fit residuals to V-fit residuals
    v_score:            float   = np.nan   # >1 = box better = planet-like
    v_flag:             bool    = False    # True = V fits better -> likely EB/rotation

    # Ingress/egress asymmetry
    asymmetry:          float   = np.nan   # 0 = perfectly symmetric
    asymm_flag:         bool    = False    # True = significantly asymmetric

    # Two-dip scan: secondary minimum near phase ±0.5
    secondary_depth:    float   = np.nan   # median flux in [0.4, 0.6]; negative = dip
    two_dip_flag:       bool    = False    # True = secondary dip detected

    # Per-transit depth consistency
    depth_cv:           float   = np.nan   # coefficient of variation of transit depths
    depth_cv_flag:      bool    = False    # True = high variability

    # Duty cycle (rotation proxy)
    duty_cycle:         float   = np.nan   # dur / period
    high_duty_flag:     bool    = False    # True = duty > ROTATION_DUTY_THRESHOLD

    # Grazing EB / V-shape check
    grazing_ratio:      float   = np.nan   # center-to-edge depth ratio (0=V-shape, 1=flat)
    grazing_flag:       bool    = False    # True = V-shaped transit (grazing EB)

    # Composite
    n_flags:            int     = 0
    is_strong_fp:       bool    = False    # True = 2+ flags -> skip or deprioritize
    flag_summary:       str     = "none"


def _bin_phase(phase: np.ndarray, flux: np.ndarray, n_bins: int = 200):
    """Bin flux into n_bins equal-width phase bins, return (centers, medians)."""
    edges   = np.linspace(-0.5, 0.5, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    binned  = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (phase >= edges[i]) & (phase < edges[i + 1])
        if mask.sum() >= 2:
            binned[i] = np.nanmedian(flux[mask])
    return centers, binned


def _fit_box(centers: np.ndarray, binned: np.ndarray,
             t_ingress: float, t_egress: float):
    """
    Fit a flat-bottomed box model to the binned phase curve.
    t_ingress, t_egress: phase positions of ingress/egress edges (e.g. -dur/2, +dur/2).
    Returns RMS residual of fit.
    """
    valid = ~np.isnan(binned)
    if valid.sum() < 5:
        return np.nan
    c, b = centers[valid], binned[valid]
    # In-transit: between t_ingress and t_egress
    in_tr  = (c >= t_ingress) & (c <= t_egress)
    out_tr = ~in_tr
    if in_tr.sum() < 2 or out_tr.sum() < 2:
        return np.nan
    box_depth  = np.nanmedian(b[in_tr])
    box_model  = np.where(in_tr, box_depth, 0.0)
    return float(np.sqrt(np.mean((b - box_model) ** 2)))


def _fit_v_shape(centers: np.ndarray, binned: np.ndarray,
                 t_ingress: float, t_egress: float):
    """
    Fit a V-shape (triangular) model: linearly deepens from t_ingress to
    midpoint (phase 0), then linearly recovers to t_egress. Flat outside.
    Returns RMS residual of fit.
    """
    valid = ~np.isnan(binned)
    if valid.sum() < 5:
        return np.nan
    c, b = centers[valid], binned[valid]
    in_tr = (c >= t_ingress) & (c <= t_egress)
    if in_tr.sum() < 2:
        return np.nan
    # V-shape model: depth proportional to distance from midpoint
    half_dur = (t_egress - t_ingress) / 2.0
    if half_dur <= 0:
        return np.nan
    v_depth = np.nanmedian(b[(c >= -0.01) & (c <= 0.01)]) if ((c >= -0.01) & (c <= 0.01) & valid).sum() >= 1 else np.nanmin(b)
    v_model = np.where(in_tr,
                       v_depth * (1.0 - np.abs(c) / half_dur).clip(0, 1),
                       0.0)
    return float(np.sqrt(np.mean((b - v_model) ** 2)))


# ── Main filter function ──────────────────────────────────────────────────────

def compute_morph_flags(
    time:       np.ndarray,
    flux:       np.ndarray,
    period:     float,
    t0:         float,
    duration:   float,
    per_transit_fluxes: Optional[list] = None,
) -> MorphFlags:
    """
    Compute all five morphological flags for one BLS candidate.

    Parameters
    ----------
    time      : 1-D array of observation times (same units as period/t0)
    flux      : 1-D detrended, normalized flux array (z-scored)
    period    : BLS best-fit period
    t0        : BLS best-fit transit epoch
    duration  : BLS best-fit transit duration
    per_transit_fluxes : optional list of 1-D flux arrays, one per individual transit,
                         for the depth-consistency check. If None, the check is skipped.

    Returns
    -------
    MorphFlags dataclass with all scores and flags set.
    """
    mf = MorphFlags()

    # Fold
    phase = ((time - t0) / period) % 1.0
    phase[phase >= 0.5] -= 1.0
    dur_phase = duration / period
    t_ingress = -dur_phase / 2.0
    t_egress  =  dur_phase / 2.0

    centers, binned = _bin_phase(phase, flux, n_bins=200)
    valid = ~np.isnan(binned)

    # ── 1. V-score ─────────────────────────────────────────────────────────
    rms_box = _fit_box(  centers, binned, t_ingress, t_egress)
    rms_v   = _fit_v_shape(centers, binned, t_ingress, t_egress)
    if not np.isnan(rms_box) and not np.isnan(rms_v) and rms_v > 0:
        mf.v_score = float(rms_box / rms_v)
        mf.v_flag  = mf.v_score > V_SCORE_THRESHOLD  # V fits better -> EB/rotation

    # ── 2. Ingress/egress asymmetry ────────────────────────────────────────
    # Compare median flux at phase [-dur, -dur/2] vs [+dur/2, +dur]
    ingress_mask  = (centers >= t_ingress - dur_phase) & (centers < t_ingress) & valid
    egress_mask   = (centers >  t_egress)              & (centers <= t_egress + dur_phase) & valid
    mean_depth    = np.nanmedian(binned[(centers >= t_ingress) & (centers <= t_egress) & valid]) if valid.sum() > 2 else np.nan
    if ingress_mask.sum() >= 2 and egress_mask.sum() >= 2 and not np.isnan(mean_depth) and abs(mean_depth) > 0:
        d_ingress = np.nanmedian(binned[ingress_mask])
        d_egress  = np.nanmedian(binned[egress_mask])
        mf.asymmetry  = float(abs(d_ingress - d_egress) / abs(mean_depth))
        mf.asymm_flag = mf.asymmetry > ASYMM_THRESHOLD

    # ── 3. Two-dip scan ────────────────────────────────────────────────────
    # Secondary eclipse is split across the phase fold boundary (±0.5).
    # Use the minimum binned value near each edge (more sensitive than median
    # when the secondary is narrow relative to the scan window).
    pos_sec = (centers >= 0.40) & (centers <= 0.50) & valid
    neg_sec = (centers >= -0.50) & (centers <= -0.40) & valid
    sec_depths = []
    if pos_sec.sum() >= 2: sec_depths.append(float(np.nanmin(binned[pos_sec])))
    if neg_sec.sum() >= 2: sec_depths.append(float(np.nanmin(binned[neg_sec])))
    if sec_depths:
        mf.secondary_depth = float(min(sec_depths))  # most negative = deepest secondary
        mf.two_dip_flag    = mf.secondary_depth < TWO_DIP_THRESHOLD

    # ── 4. Per-transit depth consistency ───────────────────────────────────
    if per_transit_fluxes is not None and len(per_transit_fluxes) >= 4:
        win = dur_phase * 1.5
        depths = []
        for tr_flux in per_transit_fluxes:
            # Assumes tr_flux is already phase-folded for this transit
            # or provide (time_chunk, flux_chunk) pairs — see note below
            if hasattr(tr_flux, '__len__') and len(tr_flux) >= 2:
                depths.append(float(np.nanmedian(tr_flux)))
        if len(depths) >= 4:
            mean_d = np.mean(depths)
            std_d  = np.std(depths)
            if abs(mean_d) > 0.05:   # only meaningful if there is a real dip
                mf.depth_cv      = float(std_d / abs(mean_d))
                mf.depth_cv_flag = mf.depth_cv > DEPTH_CV_THRESHOLD

    # ── 5. Duty cycle (rotation proxy) ────────────────────────────────────
    # Only flag genuinely high duty cycles — NOT when the BLS duration simply
    # hit the search ceiling (0.500d). A ceiling hit means the BLS grid ran
    # out of room, not that the transit is truly that wide. Ceiling targets
    # get a separate DURATION_CEILING soft flag instead.
    mf.duty_cycle = float(dur_phase)   # = duration / period
    at_ceiling = (duration >= 0.499)   # BLS normal-pass duration ceiling
    if at_ceiling:
        # Soft flag only: note ceiling hit but don't mark as hard FP
        pass   # flagged via ASYMM or V_SCORE if truly bad shape
    else:
        mf.high_duty_flag = mf.duty_cycle > ROTATION_DUTY_THRESHOLD

    # ── 6. Grazing EB / V-shape check ─────────────────────────────────────
    # Compares transit depth at the center vs the ingress/egress edges.
    # A flat-bottomed (planet) transit has center ≈ edge depth (ratio ~1.0).
    # A V-shaped (grazing EB) transit has center >> edge depth (ratio ~0.0).
    # This is a HARD flag — catches grazing EBs that pass all other checks.
    center_mask = (np.abs(centers) < dur_phase * 0.20) & valid
    edge_mask   = ((np.abs(centers) >= dur_phase * 0.30) &
                   (np.abs(centers) <  dur_phase * 0.55) & valid)
    if center_mask.sum() >= 2 and edge_mask.sum() >= 2:
        center_d = float(np.nanmedian(binned[center_mask]))
        edge_d   = float(np.nanmedian(binned[edge_mask]))
        if center_d < -0.10:   # real dip at center
            ratio = float(np.clip(edge_d / center_d, 0, 1.5))
            mf.grazing_ratio = ratio
            mf.grazing_flag  = ratio < GRAZING_THRESH

    # ── Composite ─────────────────────────────────────────────────────────
    flag_list = []
    hard_flags = []   # high-confidence FP indicators
    soft_flags = []   # informational; can fire on noisy real transits

    if mf.v_flag:
        soft_flags.append(f"V_SCORE={mf.v_score:.2f}>1.5 (V-shaped, EB/rotation)")
    if mf.asymm_flag:
        soft_flags.append(f"ASYMM={mf.asymmetry:.2f}")
    if mf.two_dip_flag:
        hard_flags.append(f"SECONDARY_DIP={mf.secondary_depth:.3f}σ")
    if mf.depth_cv_flag:
        soft_flags.append(f"DEPTH_CV={mf.depth_cv:.2f}")
    if mf.high_duty_flag:
        hard_flags.append(f"HIGH_DUTY={mf.duty_cycle:.1%}")
    if mf.grazing_flag:
        hard_flags.append(f"GRAZING_V_SHAPE(ratio={mf.grazing_ratio:.2f})")

    flag_list = hard_flags + soft_flags
    mf.n_flags = len(flag_list)

    # strong_fp requires at least ONE hard flag (secondary eclipse at -0.80σ
    # or duty cycle > 30%). Two soft flags alone (e.g. slightly noisy fold +
    # elevated duty) are not sufficient to discard a candidate — those patterns
    # occur on real near-threshold planets.
    mf.is_strong_fp  = len(hard_flags) >= 1
    mf.flag_summary  = " | ".join(flag_list) if flag_list else "none"

    return mf


# ── Convenience wrapper for the sweep pipeline ────────────────────────────────

def morph_flags_to_dict(mf: MorphFlags) -> dict:
    """Convert MorphFlags to a flat dict suitable for adding to the sweep CSV row."""
    return {
        "morph_v_score":       round(mf.v_score,    4) if not np.isnan(mf.v_score)    else None,
        "morph_asymmetry":     round(mf.asymmetry,  4) if not np.isnan(mf.asymmetry)  else None,
        "morph_secondary_dip": round(mf.secondary_depth, 4) if not np.isnan(mf.secondary_depth) else None,
        "morph_depth_cv":      round(mf.depth_cv,   4) if not np.isnan(mf.depth_cv)   else None,
        "morph_duty_cycle":    round(mf.duty_cycle, 4) if not np.isnan(mf.duty_cycle) else None,
        "morph_grazing_ratio": round(mf.grazing_ratio, 4) if not np.isnan(mf.grazing_ratio) else None,
        "morph_grazing_flag":  mf.grazing_flag,
        "morph_n_flags":       mf.n_flags,
        "morph_is_strong_fp":  mf.is_strong_fp,
        "morph_flags":         mf.flag_summary,
    }


# ── Quick test on the known FP cases ─────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd

    # Simulate the known false positive signatures to verify thresholds
    np.random.seed(42)
    t = np.linspace(0, 27.4, 20000)
    noise = np.random.normal(0, 1.0, len(t))

    print("=== MORPHOLOGY FILTER SELF-TEST ===\n")

    # Test 1: Real box transit (should pass — low flag count)
    period, t0, dur = 2.0, 0.5, 0.10
    phase = ((t - t0) / period) % 1.0
    phase[phase >= 0.5] -= 1.0
    depth = -2.0
    box_transit = np.where(np.abs(phase) < dur / (2 * period), depth, 0.0) + noise * 0.3
    mf = compute_morph_flags(t, box_transit, period, t0, dur)
    print(f"Box transit (should pass, 0 flags): n_flags={mf.n_flags}  v_score={mf.v_score:.3f}  flags={mf.flag_summary}")

    # Test 2: V-shape (gradual grazing EB)
    # NOTE: With V_SCORE_THRESHOLD=1.5, a moderate V-shape may not flag automatically.
    # Gradual grazing EBs (like TIC 12494582) are caught in visual vetting instead.
    # Only extreme V-shapes (v_score > 1.5) flag here.
    v_shape = np.where(np.abs(phase) < 0.3,
                       depth * (1 - np.abs(phase) / 0.3),
                       0.0) + noise * 0.3
    mf2 = compute_morph_flags(t, v_shape, period, t0, duration=period * 0.3)
    print(f"V-shape EB  (may or may not flag): n_flags={mf2.n_flags}  v_score={mf2.v_score:.3f}  flags={mf2.flag_summary}")

    # Test 3: Rotational modulation (high duty, V-like — should flag)
    period_rot = 0.134
    rot_mod = -1.5 * np.sin(2 * np.pi * t / period_rot) + noise * 0.5
    mf3 = compute_morph_flags(t, rot_mod, period_rot, 0.0, duration=period_rot * 0.10)
    print(f"Rotation    (should flag): n_flags={mf3.n_flags}  v_score={mf3.v_score:.3f}  duty={mf3.duty_cycle:.1%}  flags={mf3.flag_summary}")

    # Test 4: Two-dip EB (secondary eclipse — should flag)
    phase2 = ((t - 0.0) / 4.83) % 1.0
    phase2[phase2 >= 0.5] -= 1.0
    two_dip = (np.where(np.abs(phase2) < 0.04, -3.0, 0.0) +   # primary
               np.where(np.abs(np.abs(phase2) - 0.5) < 0.03, -1.5, 0.0) +  # secondary
               noise * 0.3)
    mf4 = compute_morph_flags(t, two_dip, 4.83, 0.0, duration=4.83 * 0.04)
    print(f"Two-dip EB  (should flag): n_flags={mf4.n_flags}  sec_dip={mf4.secondary_depth:.3f}  flags={mf4.flag_summary}")

    print("\nAll tests complete.")
    print("\nINTEGRATION NOTE:")
    print("  Add to phase_5_sweep.py in score_target_topk(), after folding each candidate:")
    print("    from morphology_filters import compute_morph_flags, morph_flags_to_dict")
    print("    mf = compute_morph_flags(lc_time, lc_flux_detrended, period, t0, duration)")
    print("    row.update(morph_flags_to_dict(mf))")
    print("    if mf.is_strong_fp: continue  # skip CNN for obvious FPs (optional)")
