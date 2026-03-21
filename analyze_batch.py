#!/usr/bin/env python3
"""
analyze_batch.py
----------------
Run after each batch of sectors to classify candidates and identify
the most promising targets for visual vetting.

Usage:
    python analyze_batch.py tess_run_summary.csv
    python analyze_batch.py tess_run_summary_batch1.csv --sector-label "Sectors 1-13"

Output:
    - Console summary table
    - <input_stem>_analysis.csv  (full classification)
    - <input_stem>_priority.csv  (Tier A & B only, for vet_candidates.py)
"""

import sys
import argparse
import numpy as np
import pandas as pd


# ── Tier classification logic ─────────────────────────────────────────────────
def classify_candidate(row):
    """
    Returns (tier, reasons) based on automated vetting checks.

    Tier A  — all checks pass, high priority for visual vetting
    Tier A* — elevated duty cycle only, otherwise clean
    Tier B  — one minor warning flag
    Tier C  — strong false-positive indicator (EB or rotation)
    Tier D  — ambiguous, low priority
    """
    score         = float(row.get("score", 0) or 0)
    polarity      = bool(row.get("polarity_flipped", True))
    duty          = float(row.get("morph_duty_cycle") or 0)
    v_score       = row.get("morph_v_score")
    sec_dip       = float(row.get("morph_secondary_dip") or 0)
    oe_ratio      = row.get("odd_even_ratio")   # from vetting CSV if present
    n_flags       = int(row.get("morph_n_flags") or 0)
    morph_flags   = str(row.get("morph_flags") or "")
    bls_dur       = float(row.get("bls_duration") or 0)
    bls_period    = float(row.get("bls_period") or 1)

    reasons = []

    # Disqualifying: bump polarity
    if polarity:
        return "bump", ["polarity=bump (EB/flare)"]

    # Strong FP indicators
    if sec_dip < -0.30:
        reasons.append(f"secondary dip {sec_dip:.2f}σ")
    if duty >= 0.25:
        reasons.append(f"high duty cycle {duty:.1%}")
    if v_score is not None and float(v_score) > 1.2:
        reasons.append(f"V-shaped fold (v_score={float(v_score):.2f})")
    if "SECONDARY_DIP" in morph_flags:
        if f"secondary dip" not in " ".join(reasons):
            reasons.append("secondary dip (morph flag)")
    if bls_dur >= 0.499:
        reasons.append("duration at search ceiling")

    if len([r for r in reasons if "secondary" in r or "V-shaped" in r
            or "high duty" in r]) >= 1:
        return "C", reasons

    # Moderate warnings
    warns = []
    if 0.15 <= duty < 0.25:
        warns.append(f"elevated duty {duty:.1%}")
    if n_flags == 1 and "HIGH_DUTY" not in morph_flags:
        warns.append(f"1 morph flag: {morph_flags[:50]}")
    if bls_dur >= 0.499 and not reasons:
        warns.append("duration at ceiling")

    if len(reasons) > 0:
        return "C", reasons
    if len(warns) >= 2:
        return "D", warns
    if len(warns) == 1 and "elevated duty" in warns[0] and 0.15 <= duty < 0.20:
        return "A*", warns
    if len(warns) == 1:
        return "B", warns

    return "A", ["clean"]


# ── Multi-sector consistency check ───────────────────────────────────────────
def find_multi_sector(df, period_tol=0.001):
    """
    Find targets appearing in multiple sectors with consistent BLS periods.
    These are the strongest planet candidates without a telescope.
    """
    dip_clean = df[(df["polarity_flipped"] == False) &
                   (df.get("morph_n_flags", pd.Series([-1]*len(df))).fillna(-1) <= 0)].copy()

    if "target" not in dip_clean.columns:
        return pd.DataFrame()

    multi = dip_clean.groupby("target").filter(lambda x: len(x) >= 2)
    if len(multi) == 0:
        return pd.DataFrame()

    consistent = []
    for tic, grp in multi.groupby("target"):
        periods = grp["bls_period"].dropna()
        if len(periods) < 2:
            continue
        cv = periods.std() / periods.mean() if periods.mean() > 0 else 1.0
        if cv < period_tol:
            best = grp.sort_values("score", ascending=False).iloc[0]
            consistent.append({
                "target":       tic,
                "n_sectors":    len(grp),
                "sectors":      sorted(grp["sector"].tolist()),
                "period_mean":  round(float(periods.mean()), 6),
                "period_cv":    round(float(cv), 6),
                "score_max":    round(float(grp["score"].max()), 4),
                "score_min":    round(float(grp["score"].min()), 4),
                "morph_n_flags":int(grp["morph_n_flags"].max() if "morph_n_flags" in grp else -1),
            })

    return pd.DataFrame(consistent).sort_values("score_max", ascending=False)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Analyze TESS sweep batch output")
    parser.add_argument("csv", help="Path to tess_run_summary.csv or similar")
    parser.add_argument("--sector-label", default="", help="Label for this batch")
    parser.add_argument("--score-threshold", type=float, default=0.86)
    parser.add_argument("--hc-threshold",    type=float, default=0.895)
    args = parser.parse_args()

    if not __import__("os").path.exists(args.csv):
        print(f"ERROR: file not found: {args.csv}")
        sys.exit(1)

    df = pd.read_csv(args.csv)
    label = args.sector_label or args.csv
    print(f"\n{'='*70}")
    print(f"BATCH ANALYSIS: {label}")
    print(f"{'='*70}")
    print(f"Total rows in file: {len(df)}")

    # ── Basic counts ──────────────────────────────────────────────────────────
    ok = df[df.get("download_status", pd.Series(["ok"]*len(df))) == "ok"].copy() \
        if "download_status" in df.columns else df.copy()

    candidates = ok[ok["score"] >= args.score_threshold].copy()
    dip = candidates[candidates.get("polarity_flipped", pd.Series([False]*len(candidates))) == False].copy() \
        if "polarity_flipped" in candidates.columns else candidates.copy()
    hc  = dip[dip["score"] >= args.hc_threshold].copy()

    bump_n = len(candidates) - len(dip) if "polarity_flipped" in candidates.columns else 0

    print(f"\nCandidates (score >= {args.score_threshold}): {len(candidates)}")
    print(f"  Dip-polarity (transit-like):  {len(dip)} ({len(dip)/max(len(candidates),1)*100:.1f}%)")
    print(f"  Bump-polarity (EB/flare):     {bump_n} ({bump_n/max(len(candidates),1)*100:.1f}%)")
    print(f"  High-confidence (>={args.hc_threshold}): {len(hc)}")

    # ── Morph filter summary ──────────────────────────────────────────────────
    if "morph_n_flags" in dip.columns:
        clean_morph = dip[dip["morph_n_flags"] == 0]
        flagged_morph = dip[dip["morph_n_flags"] > 0]
        strong_fp = dip[dip["morph_is_strong_fp"] == True] if "morph_is_strong_fp" in dip.columns else pd.DataFrame()
        print(f"\nMorphology filter (dip candidates):")
        print(f"  Clean (0 flags):   {len(clean_morph)} ({len(clean_morph)/max(len(dip),1)*100:.1f}%)")
        print(f"  Flagged (>0):      {len(flagged_morph)} ({len(flagged_morph)/max(len(dip),1)*100:.1f}%)")
        print(f"  Strong FP (>=2):   {len(strong_fp)}")

        # HC breakdown
        if len(hc) > 0 and "morph_n_flags" in hc.columns:
            hc_clean = hc[hc["morph_n_flags"] == 0]
            print(f"\nHC after morph filter:")
            print(f"  HC clean (0 flags): {len(hc_clean)}/{len(hc)}")

    # ── Period distribution ───────────────────────────────────────────────────
    if "bls_period" in dip.columns:
        bins = [0, 0.5, 1, 2, 5, 14]
        labels_b = ["<0.5d (extreme USP)", "0.5-1d (USP)", "1-2d", "2-5d", "5-13.7d"]
        dip_periods = pd.to_numeric(dip["bls_period"], errors="coerce")
        cut = pd.cut(dip_periods, bins=bins, labels=labels_b)
        print(f"\nPeriod distribution (dip candidates):")
        for lbl, cnt in cut.value_counts().reindex(labels_b).items():
            print(f"  {lbl:<22}: {cnt:>4}")

    # ── Tier classification ───────────────────────────────────────────────────
    dip = dip.copy()
    dip["tier"]         = ""
    dip["tier_reasons"] = ""
    for idx, row in dip.iterrows():
        tier, reasons = classify_candidate(row)
        dip.at[idx, "tier"] = tier
        dip.at[idx, "tier_reasons"] = "; ".join(reasons)

    tier_counts = dip["tier"].value_counts()
    print(f"\nTier classification (dip candidates):")
    for tier in ["A", "A*", "B", "C", "D", "bump"]:
        n = tier_counts.get(tier, 0)
        desc = {"A":"clean (visual vetting priority)", "A*":"elevated duty only",
                "B":"one warning flag", "C":"strong FP (discard)",
                "D":"ambiguous (low priority)", "bump":"bump polarity"}.get(tier, "")
        if n > 0:
            print(f"  Tier {tier}: {n:>4}  — {desc}")

    # ── Priority targets for visual vetting ───────────────────────────────────
    priority = dip[dip["tier"].isin(["A", "A*", "B"])].sort_values("score", ascending=False)
    print(f"\nPriority targets for visual vetting (Tier A/A*/B): {len(priority)}")
    if len(priority) > 0:
        cols_show = [c for c in ["target", "sector", "score", "bls_period",
                                  "morph_n_flags", "morph_duty_cycle", "tier",
                                  "tier_reasons"] if c in priority.columns]
        print(priority[cols_show].head(20).to_string(index=False))

    # ── Multi-sector consistency ──────────────────────────────────────────────
    multi = find_multi_sector(dip)
    if len(multi) > 0:
        print(f"\n{'='*50}")
        print(f"MULTI-SECTOR CONSISTENT CANDIDATES: {len(multi)}")
        print("(Same TIC, same period, in 2+ sectors — strongest available evidence)")
        print(multi.to_string(index=False))
    else:
        print(f"\nNo multi-sector consistent candidates in this batch.")

    # ── Sector breakdown ──────────────────────────────────────────────────────
    if "sector" in df.columns:
        print(f"\nCandidates per sector:")
        sec_counts = candidates.groupby("sector").agg(
            n_all=("score", "count"),
            n_dip=("polarity_flipped", lambda x: (x == False).sum()) if "polarity_flipped" in candidates.columns else ("score", "count"),
            score_max=("score", "max"),
        ).reset_index()
        print(sec_counts.to_string(index=False))

    # ── Save outputs ──────────────────────────────────────────────────────────
    stem = args.csv.replace(".csv", "")
    analysis_path  = f"{stem}_analysis.csv"
    priority_path  = f"{stem}_priority.csv"

    dip.to_csv(analysis_path, index=False)
    if len(priority) > 0:
        priority.to_csv(priority_path, index=False)
        print(f"\n✓ Saved analysis:  {analysis_path}")
        print(f"✓ Saved priority:  {priority_path}")
        print(f"\nNext step: upload {priority_path} to vet_candidates.py for visual vetting.")
    else:
        print(f"\n✓ Saved analysis:  {analysis_path}")
        print(f"\nNo priority targets this batch — continue to next sector batch.")

    # ── Quick instructions for vet_candidates.py ──────────────────────────────
    if len(priority) > 0:
        print(f"\n{'='*50}")
        print("TO VET PRIORITY TARGETS:")
        print("  1. Open vet_candidates.py")
        print("  2. Replace CANDIDATES list with:")
        print()
        for _, r in priority.head(10).iterrows():
            tic_num = str(r.get("target", "")).replace("TIC ", "")
            p = r.get("bls_period", 0)
            t0 = r.get("bls_t0", 0)
            dur = r.get("bls_duration", 0)
            score = r.get("score", 0)
            sector = r.get("sector", 0)
            print(f'    {{"tic": {tic_num}, "period": {p:.4f}, "t0": {t0:.4f}, '
                  f'"dur": {dur:.4f}, "score": {score:.4f}, "sector": {int(sector)}}},')
        print("  3. Run: %runfile vet_candidates.py --wdir")


if __name__ == "__main__":
    # If no args given, try the default run summary CSV
    if len(sys.argv) == 1:
        default = "tess_run_summary.csv"
        if __import__("os").path.exists(default):
            sys.argv.append(default)
        else:
            print("Usage: python analyze_batch.py <csv_file>")
            print("       python analyze_batch.py tess_run_summary.csv --sector-label 'Sectors 1-13'")
            sys.exit(0)
    main()
