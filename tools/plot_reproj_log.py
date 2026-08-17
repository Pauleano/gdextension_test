#!/usr/bin/env python
"""Plot the corner log written by tcp_receiver.py's 'l' key.

    py -3 tools/plot_reproj_log.py tools/images/reproj_log.csv

What the log holds: per frame and marker, corner 0 of the DETECTED quad (green in the overlay)
and of the REPROJECTED one (red), plus how far the head had turned and moved since the world
poses were latched. The reprojection is deliberately cross-frame, so the red-green gap is only
meaningful relative to those baselines -- see recv_markers() in tcp_receiver.py.

The point of plotting it rather than watching the overlay: the overlay shows the gap, but only
the log shows whether the gap is PROPORTIONATE to the head motion that produced it. Figure 2 is
the one that answers that, by converting both corner tracks into angles through the same
intrinsics the device used and comparing them against the reported head rotation. If the marker
sweeps far more degrees across the image than the head reportedly turned, the fault is upstream
of the reprojection -- in the head pose itself -- and no calibration value will move it.
"""

import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# MUST match camera_intrinsics in src/OpenCVProcessor.h.
FX = 435.37335635
FY = 435.96983202
CX = 320.84589009
CY = 241.55014114

# Semantic, not decorative: these ARE the overlay's colours, so the plot reads against what was on
# screen. Red/green is the classic colour-vision-deficient pair, so both tracks additionally carry
# a distinct line style and a direct label -- identity is never colour alone.
C_GREEN = "#1b9e4b"
C_RED = "#d1341f"
C_ACCENT = "#2f6fd0"
C_GRID = "#d8d8d8"
C_INK = "#222222"
C_MUTED = "#777777"


def load(path):
    """CSV -> dict of column name to array, with rows sorted by frame."""
    raw = np.genfromtxt(path, delimiter=",", names=True)
    if raw.size == 0:
        raise SystemExit("empty log")
    cols = {n: np.atleast_1d(raw[n]).astype(float) for n in raw.dtype.names}
    order = np.argsort(cols["frame"], kind="stable")
    return {k: v[order] for k, v in cols.items()}


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


def gapped(frame, value):
    """Insert NaN across dropped frames so the line breaks instead of interpolating over a gap.

    Without this a 10-frame dropout is drawn as a straight segment, which looks exactly like a
    slow steady drift and is the one artefact that would mislead the reading.
    """
    f = np.arange(int(frame[0]), int(frame[-1]) + 1, dtype=float)
    v = np.full(f.shape, np.nan)
    v[(frame - frame[0]).astype(int)] = value
    return f, v


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


def style(ax):
    ax.grid(True, color=C_GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_GRID)
    ax.tick_params(colors=C_MUTED, labelsize=8)
    ax.yaxis.label.set_color(C_INK)
    ax.xaxis.label.set_color(C_INK)


def mark_holds(ax, frame, holds):
    for a, _b in holds[1:]:
        ax.axvline(frame[a], color=C_MUTED, linewidth=0.8, alpha=0.5, zorder=0)


def figure_tracks(d, holds, mid):
    """Four stacked panels on one shared frame axis.

    Deliberately four panels and not two with twin axes: gap, angle, degrees and metres have
    unrelated scales, and overlaying them on a second y-axis invents visual crossings that carry
    no meaning.
    """
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True, layout="constrained",
                             gridspec_kw={"height_ratios": [1.3, 1.3, 1, 1]})
    fig.suptitle("Reprojection gap vs. head motion  |  marker id %d" % mid,
                 fontsize=13, color=C_INK, x=0.06, ha="left")

    f, v = gapped(d["frame"], d["dist"])
    ax = axes[0]
    ax.plot(f, v, color=C_ACCENT, linewidth=2.0)
    ax.set_yscale("log")
    ax.set_ylabel("red-green gap (px, log)")
    ax.set_title("Grey rules = latch reset. Within a hold the gap should stay flat and small.",
                 fontsize=9, color=C_MUTED, loc="left", pad=6)

    ax = axes[1]
    fg, vg = gapped(d["frame"], d["green_x"])
    fr, vr = gapped(d["frame"], d["red_x"])
    ax.plot(fg, vg, color=C_GREEN, linewidth=2.0, linestyle="-", label="detected (green)")
    ax.plot(fr, vr, color=C_RED, linewidth=2.0, linestyle="--", label="reprojected (red)")
    ax.set_ylabel("corner 0, x (px)")
    ax.legend(frameon=False, fontsize=8, labelcolor=C_INK, loc="upper right")
    ax.set_title("If these diverge, the two are not describing the same head motion.",
                 fontsize=9, color=C_MUTED, loc="left", pad=6)

    ax = axes[2]
    f, v = gapped(d["frame"], d["baseline_deg"])
    ax.plot(f, v, color=C_ACCENT, linewidth=2.0)
    ax.set_ylabel("head turn since latch (deg)")

    ax = axes[3]
    f, v = gapped(d["frame"], d["baseline_m"] * 1000.0)
    ax.plot(f, v, color=C_ACCENT, linewidth=2.0)
    ax.set_ylabel("head move since latch (mm)")
    ax.set_xlabel("frame")

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


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    d = load(sys.argv[1])

    for mid in sorted(set(d["id"].astype(int))):
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

        f1.savefig("tools/images/reproj_tracks_id%d.png" % mid, dpi=130)
        f2.savefig("tools/images/reproj_consistency_id%d.png" % mid, dpi=130)
        print("  written: tools/images/reproj_{tracks,consistency}_id%d.png" % mid)

    # Skipped under a headless backend, where it only warns. The PNGs above are written either way,
    # so a non-interactive run is still useful.
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
