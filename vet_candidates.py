#!/usr/bin/env python3
"""
vet_candidates.py
-----------------
Downloads TESS light curves for the 20 high-confidence candidates and produces
a vetting plot for each, saved as PNG files in ./vetting_plots/.

Each plot shows:
  - Top left:    Full detrended light curve with transit markers
  - Top right:   Folded phase curve (all transits stacked), binned
  - Bottom left: Odd transits folded (transit numbers 1, 3, 5, ...)
  - Bottom right: Even transits folded (transit numbers 2, 4, 6, ...)
    => Odd/even depth difference flags eclipsing binaries

  - Phase 0.5 panel: Flux at secondary eclipse phase (flags EBs)

  Summary line printed to console: score, depth, odd/even ratio, phase-0.5 level.

Usage:
    pip install lightkurve matplotlib numpy scipy
    python vet_candidates.py

Output:
    vetting_plots/TIC_XXXXXXXX_P_X.XXXX.png   (one per candidate)
    vetting_summary.csv                         (odd/even depth, phase-0.5, flags)
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

try:
    import lightkurve as lk
except ImportError:
    raise ImportError("Run: pip install lightkurve")

# ── Command-line argument support ─────────────────────────────────────────────
# Allows loading candidates from analyze_batch.py output:
#   python vet_candidates.py --csv tess_run_summary_priority.csv --top 20
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--csv",  default=None, help="Priority CSV from analyze_batch.py")
_parser.add_argument("--top",  type=int, default=None, help="Limit to top N by score")
_parser.add_argument("-h", "--help", action="store_true")
_args, _ = _parser.parse_known_args()

if _args.help:
    print(__doc__)
    sys.exit(0)

def _load_from_csv(path, top_n=None):
    """Load candidate list from a CSV produced by analyze_batch.py."""
    df = pd.read_csv(path)
    required = {"target", "bls_period", "bls_t0", "bls_duration", "score", "sector"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    if top_n:
        df = df.sort_values("score", ascending=False).head(top_n)
    cands = []
    for _, row in df.iterrows():
        tic_str = str(row["target"])
        tic_num = int(tic_str.replace("TIC ", "").strip())
        cands.append({
            "tic":    tic_num,
            "period": float(row["bls_period"]),
            "t0":     float(row["bls_t0"]),
            "dur":    float(row["bls_duration"]),
            "score":  float(row["score"]),
            "sector": int(row["sector"]),
        })
    return cands

# ── Candidates ────────────────────────────────────────────────────────────────
CANDIDATES = CANDIDATES = [
    {"tic": 672624,   "period": 0.5480, "t0": 0.196, "dur": 0.042, "score": 0.8958, "sector": 5},
    {"tic": 7211118,  "period": 0.7235, "t0": 0.479, "dur": 0.133, "score": 0.8918, "sector": 7},
    {"tic": 11198532, "period": 0.8069, "t0": 0.088, "dur": 0.225, "score": 0.8887, "sector": 6},
    {"tic": 11197950, "period": 0.3066, "t0": 0.132, "dur": 0.042, "score": 0.8856, "sector": 6},
    {"tic": 7724421,  "period": 0.3212, "t0": 0.386, "dur": 0.042, "score": 0.8836, "sector": 6},
    {"tic": 6518047,  "period": 0.9268, "t0": 0.446, "dur": 0.133, "score": 0.8842, "sector": 7},
    {"tic": 4254645,  "period": 2.0019, "t0": 0.652, "dur": 0.296, "score": 0.8916, "sector": 6},
    {"tic": 1196529,  "period": 3.7074, "t0": 0.600, "dur": 0.296, "score": 0.8827, "sector": 5},
    {"tic": 7020254,  "period": 4.0592, "t0": 0.650, "dur": 0.100, "score": 0.8821, "sector": 7},
]

OUT_DIR = "vetting_plots"
os.makedirs(OUT_DIR, exist_ok=True)

PHASE_BINS   = 200   # bins for folded phase curve
ODD_EVEN_WIN = 1.5   # fold window = ODD_EVEN_WIN × duration on each side of transit


# ── Helpers ───────────────────────────────────────────────────────────────────

def detrend(time, flux, kernel_frac=0.05):
    """Moving-median detrend, same as pipeline."""
    n = max(11, int(len(time) * kernel_frac) | 1)   # ensure odd
    from scipy.ndimage import median_filter
    trend = median_filter(flux, size=n, mode="reflect")
    residual = flux - trend
    std = np.nanstd(residual)
    if std > 0:
        residual /= std
    return residual


def sigma_clip(flux, nsigma=5):
    med = np.nanmedian(flux)
    std = np.nanstd(flux)
    mask = np.abs(flux - med) < nsigma * std
    return mask


def fold(time, flux, period, t0):
    """Return phase in [-0.5, 0.5)."""
    phase = ((time - t0) / period) % 1.0
    phase[phase >= 0.5] -= 1.0
    return phase


def bin_phase(phase, flux, n_bins=PHASE_BINS):
    edges = np.linspace(-0.5, 0.5, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    binned = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (phase >= edges[i]) & (phase < edges[i + 1])
        if mask.sum() >= 2:
            binned[i] = np.nanmedian(flux[mask])
    return centers, binned


def transit_numbers(time, t0, period):
    """Return integer transit number for each data point."""
    return np.floor((time - t0) / period + 0.5).astype(int)


def in_transit_window(phase, dur_phase, window_mult=ODD_EVEN_WIN):
    """Boolean mask: within window_mult × half-duration of phase 0."""
    return np.abs(phase) < window_mult * dur_phase / 2


def odd_even_depths(time, flux, phase, t0, period, dur):
    """
    Returns (odd_depth, even_depth, odd_n, even_n).
    depth = median in-transit flux (negative = dip).
    """
    tnum = transit_numbers(time, t0, period)
    dur_phase = dur / period
    win = in_transit_window(phase, dur_phase)

    depths = {}
    for n in np.unique(tnum):
        mask = win & (tnum == n)
        if mask.sum() >= 3:
            depths[n] = np.nanmedian(flux[mask])

    odd_vals  = [d for n, d in depths.items() if n % 2 != 0]
    even_vals = [d for n, d in depths.items() if n % 2 == 0]

    odd_d  = np.nanmedian(odd_vals)  if odd_vals  else np.nan
    even_d = np.nanmedian(even_vals) if even_vals else np.nan
    return odd_d, even_d, len(odd_vals), len(even_vals)


def phase05_level(flux, phase, dur_phase, hw=0.05):
    """Median flux near phase 0.5 (secondary eclipse window)."""
    mask = (np.abs(phase - 0.5) < hw) | (np.abs(phase + 0.5) < hw)
    if mask.sum() < 3:
        return np.nan
    return np.nanmedian(flux[mask])


# ── Per-target plot ───────────────────────────────────────────────────────────

def vet_target(cand):
    tic    = cand["tic"]
    period = cand["period"]
    t0_btjd= cand["t0"]
    dur    = cand["dur"]
    score  = cand["score"]
    sector = cand["sector"]

    tag = f"TIC {tic}  P={period:.4f}d  score={score:.4f}  sector {sector}"
    print(f"\n{'='*60}\n{tag}")

    # ── Download ──────────────────────────────────────────────────────────────
    try:
        sr = lk.search_lightcurve(f"TIC {tic}", mission="TESS", sector=sector,
                                   author="SPOC", exptime=120)
        if len(sr) == 0:
            sr = lk.search_lightcurve(f"TIC {tic}", mission="TESS", sector=sector)
        if len(sr) == 0:
            print(f"  No light curve found — skipping.")
            return None
        lc = sr[0].download()
        lc = lc.remove_nans().remove_outliers(sigma=8)
    except Exception as e:
        print(f"  Download error: {e}")
        return None

    # ── Prep arrays ───────────────────────────────────────────────────────────
    # Cast everything to plain float64 — lightkurve sometimes returns
    # masked arrays or Quantity objects that scipy median_filter rejects
    time     = np.array(lc.time.value,     dtype=np.float64)
    flux_raw = np.array(lc.flux.value,     dtype=np.float64)
    flux_err = np.array(lc.flux_err.value, dtype=np.float64) if hasattr(lc, "flux_err") else None

    # Replace any remaining NaNs/infs before detrending
    finite = np.isfinite(flux_raw)
    time, flux_raw = time[finite], flux_raw[finite]
    if flux_err is not None:
        flux_err = flux_err[finite]

    # Detrend
    flux = detrend(time, flux_raw)
    mask = sigma_clip(flux, nsigma=7)
    time, flux = time[mask], flux[mask]

    # Fold — t0 in the sweep CSV is stored as days since the start of the
    # light curve (the pipeline normalizes time to time - time.min() before
    # running BLS, so t0 is relative, NOT absolute BTJD).
    # Correct epoch: t0_full = time.min() + t0_bls
    t0_full = time.min() + t0_btjd
    phase = fold(time, flux, period, t0_full)
    dur_phase = dur / period

    # Odd/even
    odd_d, even_d, n_odd, n_even = odd_even_depths(time, flux, phase, t0_full, period, dur)
    p05 = phase05_level(flux, phase, dur_phase)

    # ── Duty cycle ────────────────────────────────────────────────────────────
    # For a genuine transit: duty = dur/period should be <~15%.
    # Large duty cycle (>20%) means the "transit" spans a big fraction of the
    # orbit — inconsistent with a planet, consistent with stellar rotation or
    # an EB at half the true period.
    duty_cycle = dur / period  # fractional (0–1)

    # ── Dip centroid offset ───────────────────────────────────────────────────
    # Where is the actual deepest point of the folded light curve relative to
    # phase 0?  A real transit should be centred at phase 0 (BLS epoch).
    # Large offset (>5% of the period) suggests the epoch is wrong, or that
    # the dip is not a transit at all (e.g. rotational modulation).
    bc_all, bb_all = bin_phase(phase, flux, n_bins=PHASE_BINS)
    valid_bins = ~np.isnan(bb_all)
    if valid_bins.sum() > 5:
        dip_center = bc_all[valid_bins][np.argmin(bb_all[valid_bins])]
    else:
        dip_center = np.nan

    # ── Flag logic ────────────────────────────────────────────────────────────
    flags = []

    # 1. Odd/even depth ratio
    if not np.isnan(odd_d) and not np.isnan(even_d) and abs(odd_d) > 0:
        ratio = abs(even_d / odd_d) if abs(odd_d) > 0 else np.nan
        if not np.isnan(ratio) and (ratio < 0.6 or ratio > 1.7):
            flags.append(f"ODD/EVEN DEPTH RATIO={ratio:.2f} ⚠ possible EB")
    else:
        ratio = np.nan

    # 2. Secondary eclipse (phase 0.5 dip)
    if not np.isnan(p05) and p05 < -0.3:
        flags.append(f"SECONDARY DIP at phase 0.5 ({p05:.3f}σ) ⚠ possible EB")

    # 3. Duration at BLS search ceiling
    if dur >= 0.499:
        flags.append("DURATION AT SEARCH CEILING")

    # 4. Duty cycle — transit spans too large a fraction of the orbit
    DUTY_WARN  = 0.15   # >15%: warn
    DUTY_FATAL = 0.25   # >25%: almost certainly not a planet
    if duty_cycle > DUTY_FATAL:
        flags.append(
            f"HIGH DUTY CYCLE={duty_cycle:.1%} ⚠ likely stellar rotation or EB "
            f"(transit duration is {duty_cycle:.0%} of orbit)"
        )
    elif duty_cycle > DUTY_WARN:
        flags.append(
            f"ELEVATED DUTY CYCLE={duty_cycle:.1%} — verify transit shape"
        )

    # 5. Dip centroid far from phase 0
    # BLS epoch estimation has intrinsic uncertainty of ~0.05–0.10 phase units
    # for shallow signals; only flag offsets > 0.15 as genuinely suspicious.
    # A real transit should be within ±0.15 of phase 0.
    DIP_OFFSET_WARN = 0.15
    if not np.isnan(dip_center) and abs(dip_center) > DIP_OFFSET_WARN:
        flags.append(
            f"DIP CENTROID OFFSET={dip_center:+.3f} phase units "
            f"⚠ possible non-transit morphology (rotation / EB / artifact)"
        )

    n_transits = len(np.unique(transit_numbers(time, t0_full, period)))
    print(f"  transits in sector: ~{n_transits}")
    print(f"  duty cycle: {duty_cycle:.1%}  (dur={dur:.4f}d / period={period:.4f}d)")
    print(f"  dip centroid: phase {dip_center:+.3f}")
    print(f"  odd depth: {odd_d:.3f}σ (n={n_odd})  even depth: {even_d:.3f}σ (n={n_even})")
    print(f"  phase-0.5 level: {p05:.3f}σ")
    if flags:
        for f in flags: print(f"  FLAG: {f}")
    else:
        print("  No red flags.")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(tag + ("\n⚠ " + " | ".join(flags) if flags else "  ✓ No red flags"),
                 fontsize=11, fontweight="bold",
                 color="firebrick" if flags else "darkgreen")

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # — Panel 1: full light curve ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.scatter(time, flux, s=1, c="steelblue", alpha=0.5, rasterized=True)
    ax1.axhline(0, color="k", lw=0.5, ls="--")
    # mark transit times
    t_first = t0_full - period * int((t0_full - time.min()) / period + 1)
    t_transits = np.arange(t_first, time.max() + period, period)
    for tt in t_transits:
        if time.min() <= tt <= time.max():
            ax1.axvline(tt, color="tomato", alpha=0.4, lw=0.8)
    ax1.set_xlabel("BTJD", fontsize=9)
    ax1.set_ylabel("Detrended flux (σ)", fontsize=9)
    ax1.set_title("Full light curve (red ticks = predicted transits)", fontsize=9)
    ax1.set_ylim(np.nanpercentile(flux, 0.5), np.nanpercentile(flux, 99.5))

    # — Panel 2: full folded phase curve ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ph_c, ph_b = bin_phase(phase, flux)
    ax2.scatter(phase, flux, s=1, c="steelblue", alpha=0.3, rasterized=True)
    ax2.plot(ph_c, ph_b, "r-", lw=1.5, label="binned median")
    ax2.axvline(0, color="k", lw=0.8, ls="--")
    ax2.set_xlim(-0.5, 0.5)
    ax2.set_ylim(np.nanpercentile(flux, 1), np.nanpercentile(flux, 99))
    ax2.set_xlabel("Phase", fontsize=9)
    ax2.set_ylabel("Flux (σ)", fontsize=9)
    ax2.set_title("All transits folded", fontsize=9)
    ax2.legend(fontsize=8)

    # — Panel 3: zoom on transit ──────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    zoom = max(dur_phase * 4, 0.08)
    zmask = np.abs(phase) < zoom
    if zmask.sum() > 5:
        ax3.scatter(phase[zmask], flux[zmask], s=4, c="steelblue", alpha=0.6)
        ph_z, ph_bz = bin_phase(phase[zmask], flux[zmask], n_bins=60)
        valid = ~np.isnan(ph_bz)
        if valid.sum() > 3:
            ax3.plot(ph_z[valid], ph_bz[valid], "r-", lw=1.5)
    ax3.axvline(0, color="k", lw=0.8, ls="--")
    ax3.axvline(-dur_phase/2, color="orange", lw=0.8, ls=":", alpha=0.7)
    ax3.axvline( dur_phase/2, color="orange", lw=0.8, ls=":", alpha=0.7)
    ax3.set_xlim(-zoom, zoom)
    ax3.set_xlabel("Phase", fontsize=9)
    ax3.set_ylabel("Flux (σ)", fontsize=9)
    ax3.set_title(f"Zoom on transit  (orange = BLS duration ±{dur:.3f}d)", fontsize=9)

    # — Panel 4: odd transits ─────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    tnum = transit_numbers(time, t0_full, period)
    odd_mask  = np.isin(tnum, [n for n in np.unique(tnum) if n % 2 != 0])
    even_mask = np.isin(tnum, [n for n in np.unique(tnum) if n % 2 == 0])
    win_ph = min(zoom * 1.5, 0.4)

    for ax_oe, oe_mask, label, col, depth_val, n_val in [
        (ax4, odd_mask,  "Odd transits",  "mediumblue", odd_d,  n_odd),
        (fig.add_subplot(gs[2, 1]), even_mask, "Even transits", "darkorange", even_d, n_even),
    ]:
        sub_ph = phase[oe_mask]
        sub_fl = flux[oe_mask]
        sub_win = np.abs(sub_ph) < win_ph
        if sub_win.sum() > 5:
            ax_oe.scatter(sub_ph[sub_win], sub_fl[sub_win], s=3, c=col, alpha=0.5)
            bc, bb = bin_phase(sub_ph[sub_win], sub_fl[sub_win], n_bins=50)
            valid = ~np.isnan(bb)
            if valid.sum() > 3:
                ax_oe.plot(bc[valid], bb[valid], "-", color=col, lw=1.5)
        depth_str = f"{depth_val:.3f}σ" if not np.isnan(depth_val) else "n/a"
        ax_oe.axvline(0, color="k", lw=0.8, ls="--")
        ax_oe.set_xlim(-win_ph, win_ph)
        ax_oe.set_xlabel("Phase", fontsize=9)
        ax_oe.set_ylabel("Flux (σ)", fontsize=9)
        ax_oe.set_title(f"{label}  (depth≈{depth_str}, n={n_val})", fontsize=9)

    fname = os.path.join(OUT_DIR, f"TIC_{tic}_P_{period:.4f}.png")
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")

    return {
        "tic_id":          tic,
        "sector":          sector,
        "score":           score,
        "bls_period":      period,
        "bls_dur":         dur,
        "duty_cycle_pct":  round(duty_cycle * 100, 2),
        "dip_centroid":    round(dip_center, 4) if not np.isnan(dip_center) else None,
        "n_transits":      n_transits,
        "odd_depth":       round(odd_d,  4) if not np.isnan(odd_d)  else None,
        "even_depth":      round(even_d, 4) if not np.isnan(even_d) else None,
        "odd_even_ratio":  round(ratio,  3) if not np.isnan(ratio)  else None,
        "phase05_level":   round(p05,    4) if not np.isnan(p05)    else None,
        "flags":           " | ".join(flags) if flags else "none",
        "plot_file":       fname,
    }


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Override CANDIDATES from CSV if --csv flag provided
    if _args.csv:
        print(f"Loading candidates from: {_args.csv}")
        CANDIDATES = _load_from_csv(_args.csv, top_n=_args.top)
        print(f"  Loaded {len(CANDIDATES)} candidates")
    elif _args.top and _args.top < len(CANDIDATES):
        CANDIDATES = sorted(CANDIDATES, key=lambda x: x["score"], reverse=True)[:_args.top]

    print(f"Vetting {len(CANDIDATES)} high-confidence candidates")
    print(f"Output directory: {os.path.abspath(OUT_DIR)}\n")

    rows = []
    for c in CANDIDATES:
        result = vet_target(c)
        if result:
            rows.append(result)

    if rows:
        summary = pd.DataFrame(rows)
        summary.to_csv("vetting_summary.csv", index=False)
        print("\n" + "=" * 60)
        print("VETTING SUMMARY")
        print("=" * 60)
        print(summary[["tic_id", "bls_period", "duty_cycle_pct", "dip_centroid",
                        "odd_even_ratio", "phase05_level", "flags"]].to_string(index=False))

        # Classify
        clean   = summary[summary["flags"] == "none"]
        flagged = summary[summary["flags"] != "none"]
        print(f"\nClean (no flags):  {len(clean)}")
        print(f"Flagged:           {len(flagged)}")
        if len(clean) > 0:
            print("\nCleanest candidates:")
            print(clean[["tic_id", "bls_period", "score", "duty_cycle_pct",
                          "dip_centroid", "odd_even_ratio", "phase05_level"]].to_string(index=False))
        print("\nSaved: vetting_summary.csv")
        print("Plots saved in:", os.path.abspath(OUT_DIR))
