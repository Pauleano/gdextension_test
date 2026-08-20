#!/usr/bin/env python
"""Plot the corner log written by tcp_receiver.py's 'l' key.

    py -3 tools/plot_reproj_log.py tools/reproj_logs/reproj_log_20260819_143512.csv

What the log holds: per frame and marker, corner 0 of the DETECTED quad (green in the overlay)
and of the REPROJECTED one (red), plus how far the head had turned and moved since the world
poses were latched. The reprojection is deliberately cross-frame, so the red-green gap is only
meaningful relative to those baselines -- see parse_markers() in tcp_receiver.py.

The point of plotting it rather than watching the overlay: the overlay shows the gap, but only
the log shows whether the gap is PROPORTIONATE to the head motion that produced it. Figure 2 is
the one that answers that, by converting both corner tracks into angles through the same
intrinsics the device used and comparing them against the reported head rotation. If the marker
sweeps far more degrees across the image than the head reportedly turned, the fault is upstream
of the reprojection -- in the head pose itself -- and no calibration value will move it.
"""

import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# The drawing vocabulary shared with plot_marker_pos.py -- palette, the NaN-gapping, the robust
# clipping and the dots-are-samples convention. Shared rather than copied so the two scripts cannot
# drift into making different promises about the same kind of panel; see plotlib.py. (They had:
# robust_ylim/mark_offscale/trace grew up in the other script and this one never got them, so a
# single bad solve could set the scale of a panel here while the same spike was clipped and counted
# over there.)
from plotlib import (C_ACCENT, C_AXES, C_INK, C_MUTED, gapped, load, mark_offscale, robust_ylim,
                     sample_note, style, trace)

# MUST match camera_intrinsics in src/OpenCVProcessor.h -- and, if the scene overrides that property,
# the value in project/aruco_markers.tscn, which is what the device actually ran with.
FX = 436.90348444
FY = 436.86219469
CX = 321.49573022
CY = 239.71397166

# The columns a corner log must have. Checked on load so pointing this at a marker_pos CSV -- the
# other log in the same tools/ tree, with a similar name -- says so by name instead of failing as a
# KeyError inside a figure.
REQUIRED_COLS = ("frame", "id", "dist", "green_x", "green_y", "red_x", "red_y",
                 "baseline_deg", "baseline_m")

# Semantic, not decorative: these ARE the overlay's colours, so the plot reads against what was on
# screen. Red/green is the classic colour-vision-deficient pair, so both tracks additionally carry
# a distinct line style and a direct label -- identity is never colour alone. Local rather than in
# plotlib: they mean "detected vs reprojected", which is a distinction only this script draws.
C_GREEN = "#1b9e4b"
C_RED = "#d1341f"
# Per-marker identity in the cross-id figure. Deliberately NOT the green/red pair used for the two
# overlays -- that pair already means "detected vs reprojected" everywhere else in this file, and
# reusing it for marker ids would make two different distinctions share one colour language. Blue /
# orange separates under every common colour-vision deficiency, and marker shape carries the same
# information again so identity never rests on colour alone.
C_IDS = ["#2f6fd0", "#d95f02"]
M_IDS = ["o", "^"]


def split_holds(baseline_deg):
    """Index ranges between latch resets.

    A re-latch drops the baseline back to zero, so a sharp DECREASE marks the boundary. Splitting
    on that rather than on a fixed 5s stride keeps the segmentation honest when frames were
    dropped -- the log is not evenly sampled, the receiver drops a frame whenever the previous one
    is still draining.
    """
    cuts = [0]
    for i in range(1, len(baseline_deg)):
        if baseline_deg[i] < baseline_deg[i - 1] - 1.0:
            cuts.append(i)
    cuts.append(len(baseline_deg))
    return [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b - a >= 3]


def excursion_deg(x, y):
    """Angle between each frame's bearing and the FIRST frame's, in degrees.

    Both image axes, not just the column. Comparing an x-only excursion against baseline_deg -- a
    full 3D rotation magnitude -- silently understates any motion with pitch or roll in it, which
    made the ratio below read far under 1.0 on captures that were not pure yaw. It is still only a
    sanity check and not an identity: a head rotation about the marker's own bearing moves the
    marker not at all, so the ratio has meaning only while the marker stays well off that axis.
    """
    b = np.stack([(x - CX) / FX, (y - CY) / FY, np.ones_like(x)], axis=1)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip(b @ b[0], -1.0, 1.0)))


def mark_holds(ax, frame, holds):
    for a, _b in holds[1:]:
        ax.axvline(frame[a], color=C_MUTED, linewidth=0.8, alpha=0.5, zorder=0)


def figure_tracks(d, holds, mid):
    """Four stacked panels on one shared frame axis.

    Deliberately four panels and not two with twin axes: gap, angle, degrees and metres have
    unrelated scales, and overlaying them on a second y-axis invents visual crossings that carry
    no meaning.

    Which panels get robust_ylim is a per-panel decision and not a default, because the outliers
    mean different things in each. In the corner tracks a single bad solve is noise and must not be
    allowed to set the scale. In the two baseline panels the PEAK is the measurement -- how far the
    head got before the latch reset -- so clipping the top of the ramp would remove the number being
    read. The gap panel is already on a log scale, which absorbs the spikes without hiding them.
    """
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True, layout="constrained",
                             gridspec_kw={"height_ratios": [1.3, 1.3, 1, 1]})
    fig.suptitle("Reprojection gap vs. head motion  |  marker id %d" % mid,
                 fontsize=13, color=C_INK, x=0.06, ha="left")

    ax = axes[0]
    trace(ax, d["frame"], d["dist"], C_ACCENT)
    ax.set_yscale("log")
    ax.set_ylabel("red-green gap (px, log)")
    ax.set_title("Grey rules = latch reset. Within a hold the gap should stay flat and small.",
                 fontsize=9, color=C_MUTED, loc="left", pad=6)

    ax = axes[1]
    fg, vg = trace(ax, d["frame"], d["green_x"], C_GREEN, "-")
    fr, vr = trace(ax, d["frame"], d["red_x"], C_RED, "--")
    # Both tracks share one scale by definition -- the whole point is how far apart they are -- so
    # the limits come from the pair, not from either one.
    lim = robust_ylim(ax, np.concatenate([vg, vr]))
    mark_offscale(ax, fg, vg, lim)
    mark_offscale(ax, fr, vr, lim)
    ax.plot([], [], color=C_GREEN, linestyle="-", label="detected (green)")
    ax.plot([], [], color=C_RED, linestyle="--", label="reprojected (red)")
    ax.set_ylabel("corner 0, x (px)")
    ax.legend(frameon=False, fontsize=8, labelcolor=C_INK, loc="upper right")
    ax.set_title("If these diverge, the two are not describing the same head motion.",
                 fontsize=9, color=C_MUTED, loc="left", pad=6)

    ax = axes[2]
    trace(ax, d["frame"], d["baseline_deg"], C_ACCENT)
    ax.set_ylabel("head turn since latch (deg)")

    ax = axes[3]
    trace(ax, d["frame"], d["baseline_m"] * 1000.0, C_ACCENT)
    ax.set_ylabel("head move since latch (mm)")
    ax.set_xlabel("frame")
    # Bottom right, which is the one corner no panel here occupies: the gap panel fills its top, the
    # corner tracks arch through their middle and the two baselines ramp along the diagonal.
    sample_note(ax, "lower right")

    for ax in axes:
        style(ax)
        mark_holds(ax, d["frame"], holds)
    return fig


def figure_consistency(d, holds, mid):
    """Per hold: how far each track swept in ANGLE, against the reported head turn.

    This is the diagnostic. Red is produced from the head pose, so red's sweep should track the
    reported turn. Green is what the camera actually saw. Green and red parting company means the
    head pose does not describe the motion the camera underwent.
    """
    ga, ra, ba, labels = [], [], [], []
    for a, b in holds:
        ga.append(np.nanmax(excursion_deg(d["green_x"][a:b], d["green_y"][a:b])))
        ra.append(np.nanmax(excursion_deg(d["red_x"][a:b], d["red_y"][a:b])))
        ba.append(np.nanmax(d["baseline_deg"][a:b]))
        labels.append("%d-%d" % (d["frame"][a], d["frame"][b - 1]))

    ga, ra, ba = np.array(ga), np.array(ra), np.array(ba)
    y = np.arange(len(labels))
    h = 0.26

    fig, ax = plt.subplots(figsize=(11, 1.1 + 0.62 * len(labels)))
    # 2px surface gap between adjacent bars comes from the height/offset pair, not from edges.
    ax.barh(y + h, ga, height=h * 0.92, color=C_GREEN, label="marker swept (green)")
    ax.barh(y, ra, height=h * 0.92, color=C_RED, label="reprojection swept (red)")
    ax.barh(y - h, ba, height=h * 0.92, color=C_ACCENT, label="head turn reported")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("angular excursion since latch (deg)")
    ax.set_title("Per hold: what the camera saw vs. what the head pose claims  |  marker id %d"
                 % mid, fontsize=12, color=C_INK, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=C_INK, loc="lower right")
    style(ax)

    # Direct labels only on the ratio -- the number the whole figure exists to show. Labelling
    # every bar would triple the ink for values the axis already carries.
    for i, (g, b) in enumerate(zip(ga, ba)):
        if b > 0.5:
            ax.text(max(g, b) + 0.8, y[i], "%.1fx" % (g / b), va="center", fontsize=8,
                    color=C_INK)
    fig.tight_layout()
    return fig, ga, ra, ba


def figure_diagnostics(d, ids):
    """The two questions the per-id figures cannot answer.

    LEFT -- gap against head DISPLACEMENT. Turning the head exposes lens_translation; only strafing
    exposes a wrong marker size or focal length, because those put the marker at the wrong depth and
    it takes a viewpoint change for parallax to reveal that. A run whose displacement never grew has
    simply not tested that family, however small its gap looked.

    RIGHT -- the same frame's gap for one marker against the other. Two markers are different
    physical objects with independent detections and independently latched poses, so a per-marker
    fault (wrong size, planar-pose ambiguity, a bad latch) lands OFF the diagonal. Only a fault in
    what they share -- the head pose, the lens pose -- can push both at once, which puts points ON
    it. That distinction is not visible in either marker's own time series.
    """
    has_axes = all(k in d for k in ("dx_m", "dy_m", "dz_m"))
    fig, axes = plt.subplots(1, 3 if has_axes else 2,
                             figsize=(18 if has_axes else 13, 5.4), layout="constrained")

    ax = axes[0]
    for slot, mid in enumerate(ids):
        sel = d["id"].astype(int) == mid
        ax.scatter(d["baseline_m"][sel], np.maximum(d["dist"][sel], 0.05), s=9, alpha=0.28,
                   color=C_IDS[slot % 2], marker=M_IDS[slot % 2], linewidths=0,
                   label="marker %d" % mid)
    # Binned medians on top: the cloud shows spread, the line shows the trend that a residual scale
    # error would produce. Drawn in ink rather than a series colour -- it is a summary of both.
    edges = np.array([0.0, 0.05, 0.10, 0.20, 0.35, 0.60, 10.0])
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (d["baseline_m"] >= lo) & (d["baseline_m"] < hi)
        if sel.sum() > 20:
            xs.append(np.median(d["baseline_m"][sel]))
            ys.append(np.median(d["dist"][sel]))
    if xs:
        ax.plot(xs, ys, color=C_INK, linewidth=2.0, marker="s", markersize=5, zorder=5,
                label="median per bin")
    ax.set_yscale("log")
    ax.set_xlabel("head displacement since latch (m)")
    ax.set_ylabel("red-green gap (px, log)")
    ax.set_title("Does the gap grow with STRAFING?\nrising median = residual depth/scale error",
                 fontsize=10, color=C_INK, loc="left", pad=8)
    ax.legend(frameon=False, fontsize=8, labelcolor=C_INK, loc="upper left")
    style(ax)

    if has_axes:
        # WHICH axis drives the gap. The displacement arrives split into the latched head's own frame,
        # and the three directions do genuinely different things: sideways and vertical move the
        # viewpoint ACROSS the marker and so generate parallax -- that is what a wrong depth (marker
        # size, focal length) shows up in -- while forward/back only changes the range. A gap that
        # rides on the lateral curve and ignores the forward one is a depth error; one that tracks
        # range instead points at something scaling with distance.
        ax = axes[1]
        lateral = np.hypot(d["dx_m"], d["dy_m"])
        series = [("sideways + vertical (parallax)", lateral, "-"),
                  ("forward / back (range)", np.abs(d["dz_m"]), "--"),
                  ("total displacement", d["baseline_m"], ":")]
        # Quantile edges, not equal width: displacement is heavily bottom-loaded (most frames sit
        # early in a hold), so equal-width bins would crowd thousands of samples into the first and
        # leave a handful in the last -- exactly where the signal is. The bin COUNT scales with the
        # sample count so every bin keeps ~80 rows, which is far more than a median needs; a fixed
        # count was coarse on long logs, but the cap stays LOW on purpose: the motion has plateaus, so
        # fine bins slice through a cluster and the medians then order by which hold they fell in
        # rather than by displacement, which reads as a zigzag that means nothing. This only affects
        # what the curves LOOK like -- the slopes in the title come from a fit over every row.
        n_bins = int(np.clip(len(d["dist"]) // 120, 5, 9))
        for slot, (label, mag, ls) in enumerate(series):
            edges = np.unique(np.quantile(mag, np.linspace(0.0, 1.0, n_bins + 1)))
            xs, ys = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                sel = (mag >= lo) & (mag < hi)
                if sel.sum() > 15:
                    xs.append(np.median(mag[sel]))
                    ys.append(np.median(d["dist"][sel]))
            if xs:
                ax.plot(xs, ys, color=C_AXES[slot], linewidth=2.0, linestyle=ls, marker="o",
                        markersize=5, label=label)
        # The curves alone CANNOT attribute cause, and reading them that way is a trap: each is
        # plotted against its own axis, and in any real sweep the axes are correlated (you cannot
        # strafe without also drifting forward). An axis that only ever covered 0.2m looks steeper
        # than one that covered 0.9m even when it drives nothing at all. So the attribution is done
        # by a joint least-squares fit instead, whose slopes are px per metre and therefore directly
        # comparable across axes -- and the axis correlation is printed beside them, because a high
        # one means the split is weakly identified however clean the fit looks.
        fwd = np.abs(d["dz_m"])
        A = np.column_stack([np.ones_like(lateral), lateral, fwd])
        coef, *_ = np.linalg.lstsq(A, d["dist"], rcond=None)
        r_axes = float(np.corrcoef(lateral, fwd)[0, 1])
        ax.set_yscale("log")
        ax.set_xlabel("displacement along that axis (m)")
        ax.set_ylabel("red-green gap, median (px, log)")
        ax.set_title("WHICH axis drives it?\nfit: %+.1f px/m lateral, %+.1f px/m forward "
                     "(axes correlated r=%+.2f)" % (coef[1], coef[2], r_axes),
                     fontsize=10, color=C_INK, loc="left", pad=8)
        ax.legend(frameon=False, fontsize=8, labelcolor=C_INK, loc="upper left")
        style(ax)
        print("\naxis fit: gap = %+.2f %+.2f*lateral %+.2f*forward  (px, m) | corr(lateral,forward)"
              " = %+.2f" % (coef[0], coef[1], coef[2], r_axes))
        if abs(r_axes) > 0.8:
            print("  axes strongly correlated -- the per-axis split is NOT identifiable from this")
            print("  run. Record one sweep that is mostly sideways and one mostly toward/away.")

    ax = axes[-1]
    if len(ids) >= 2:
        a_id, b_id = ids[0], ids[1]
        ga = {int(f): g for f, i, g in zip(d["frame"], d["id"], d["dist"]) if int(i) == a_id}
        gb = {int(f): g for f, i, g in zip(d["frame"], d["id"], d["dist"]) if int(i) == b_id}
        common = sorted(set(ga) & set(gb))
        a = np.array([ga[k] for k in common])
        b = np.array([gb[k] for k in common])
    if len(ids) >= 2 and len(common) > 0:
        lim = max(a.max(), b.max()) * 1.3
        ax.plot([0.05, lim], [0.05, lim], color=C_MUTED, linewidth=1.2, linestyle="--",
                zorder=1, label="equal (shared cause)")
        ax.scatter(np.maximum(a, 0.05), np.maximum(b, 0.05), s=10, alpha=0.3,
                   color=C_ACCENT, linewidths=0, zorder=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.05, lim)
        ax.set_ylim(0.05, lim)
        ax.set_xlabel("marker %d gap (px, log)" % a_id)
        ax.set_ylabel("marker %d gap (px, log)" % b_id)
        r = np.corrcoef(a, b)[0, 1]
        both = int(((a > 10) & (b > 10)).sum())
        only_a = int(((a > 10) & (b <= 10)).sum())
        only_b = int(((b > 10) & (a <= 10)).sum())
        ax.set_title("Do BOTH markers spike together?\ncorrelation %+.2f  |  >10px: both %d, "
                     "only id%d %d, only id%d %d"
                     % (r, both, a_id, only_a, b_id, only_b),
                     fontsize=10, color=C_INK, loc="left", pad=8)
        ax.legend(frameon=False, fontsize=8, labelcolor=C_INK, loc="upper left")
        print("\ncross-marker: correlation %+.2f | gap>10px  both=%d  only id%d=%d  only id%d=%d"
              % (r, both, a_id, only_a, b_id, only_b))
        print("  on the diagonal = a fault in the SHARED path (head pose / lens pose);")
        print("  off it = per-marker (size, planar-pose ambiguity, a poisoned latch).")
    else:
        # Two ids that never share a frame are as useless here as one id: the comparison is
        # per-frame by construction, because only then do both markers see the same head pose.
        ax.text(0.5, 0.5, "needs two marker ids visible in the SAME frames", ha="center",
                va="center", color=C_MUTED, transform=ax.transAxes)
    style(ax)
    return fig


def out_path(src, suffix):
    """PNG name derived from the CSV name, written beside it -- i.e. in tools/reproj_logs/.

    Fixed output names would overwrite the previous log's figures the moment a second log is
    plotted -- and these logs exist precisely to be compared against each other (one capture per
    configuration). Deriving the name from the input keeps every run's figures alongside its data,
    and since tcp_receiver.py now stamps the CSV with its capture time, the figures inherit that
    stamp and stay identifiable without opening them.
    """
    stem = os.path.splitext(os.path.basename(src))[0]
    return os.path.join(os.path.dirname(src) or ".", "%s_%s.png" % (stem, suffix))


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    d = load(sys.argv[1], REQUIRED_COLS)
    ids = sorted(set(d["id"].astype(int)))

    fd = figure_diagnostics(d, ids)
    p_diag = out_path(sys.argv[1], "diagnostics")
    fd.savefig(p_diag, dpi=130)
    print("written: %s" % p_diag)

    for mid in ids:
        sel = d["id"].astype(int) == mid
        one = {k: v[sel] for k, v in d.items()}
        holds = split_holds(one["baseline_deg"])
        if not holds:
            print("marker %d: no usable holds" % mid)
            continue

        f1 = figure_tracks(one, holds, mid)
        f2, ga, ra, ba = figure_consistency(one, holds, mid)

        print("\nmarker %d: %d frames, %d holds" % (mid, sel.sum(), len(holds)))
        print("  gap px      p50=%6.1f  p90=%6.1f  max=%6.1f"
              % (np.percentile(one["dist"], 50), np.percentile(one["dist"], 90),
                 one["dist"].max()))
        print("  per hold, angular excursion (deg):")
        print("    %-14s %8s %8s %8s %8s" % ("frames", "green", "red", "head", "green/head"))
        for (a, b), g, r, h in zip(holds, ga, ra, ba):
            ratio = ("%.1f" % (g / h)) if h > 0.5 else "-"
            print("    %-14s %8.1f %8.1f %8.1f %8s"
                  % ("%d-%d" % (one["frame"][a], one["frame"][b - 1]), g, r, h, ratio))
        # green/red is the diagnostic; the /head ratios are context only. How far EITHER track
        # swept relative to the head's rotation magnitude depends on what kind of motion it was --
        # a turn about the marker's own bearing moves it not at all, while strafing moves it
        # without any rotation -- so that ratio legitimately ranges over a factor of several
        # between healthy captures. The two tracks parting company from EACH OTHER is what a bad
        # latch or a bad calibration looks like, and that is what 'gap px' above measures.
        good = ba > 0.5
        if good.any():
            print("  green/red  median %.2f  <- THE number: ~1.0 means the two describe the same"
                  " motion" % np.median(ga[good] / np.maximum(ra[good], 1e-6)))
            print("  vs head    red %.2f / green %.2f  (context only -- motion-dependent, compare"
                  " these two to each other, not to 1.0)"
                  % (np.median(ra[good] / ba[good]), np.median(ga[good] / ba[good])))

        p1 = out_path(sys.argv[1], "tracks_id%d" % mid)
        p2 = out_path(sys.argv[1], "consistency_id%d" % mid)
        f1.savefig(p1, dpi=130)
        f2.savefig(p2, dpi=130)
        print("  written: %s , %s" % (p1, p2))

    # Skipped under a headless backend, where it only warns. The PNGs above are written either way,
    # so a non-interactive run is still useful.
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
