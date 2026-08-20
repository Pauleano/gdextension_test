"""Shared drawing vocabulary for the plot scripts (plot_reproj_log.py, plot_marker_pos.py).

Not a general plotting library -- it is exactly the pieces both scripts had, or should have had.
They were copied between the two, and the copies had begun to drift: plot_marker_pos.py grew
robust_ylim/mark_offscale/trace and plot_reproj_log.py never got them, so the two ended up making
DIFFERENT promises about the same kind of picture. That is the real cost of the duplication, more
than the line count: a reader who has learned to read one figure has not learned to read the other.

What lives here is what has to look and behave identically across both. What does NOT: the
semantic colours (green/red for detected-vs-reprojected, the per-id pair), which mean something in
one script and nothing in the other, and out_path(), which genuinely differs -- one writes beside
its CSV, the other into a subfolder because a two-marker recording makes eight figures.
"""

import numpy as np

# The palette. Identity never rests on colour alone anywhere in these scripts, so the axis colours
# come with matching line styles: blue / orange / purple stays separable under deuteranopia and
# protanopia, and the styles survive greyscale printing.
C_AXES = ["#2f6fd0", "#d95f02", "#7570b3"]
S_AXES = ["-", "--", "-."]
C_ACCENT = "#2f6fd0"
C_GRID = "#d8d8d8"
C_INK = "#222222"
C_MUTED = "#777777"


def load(path, required=()):
    """CSV -> dict of column name to array, with rows sorted by frame.

    `required` is checked here rather than at the first use, so a CSV of the wrong KIND (the two
    scripts read two different logs from the same tools/ folder, with similar names) says so by
    name instead of failing later as a KeyError inside a figure.
    """
    raw = np.genfromtxt(path, delimiter=",", names=True)
    if raw.size == 0:
        raise SystemExit("empty log")
    cols = {n: np.atleast_1d(raw[n]).astype(float) for n in raw.dtype.names}
    missing = set(required) - set(cols)
    if missing:
        raise SystemExit("wrong log kind for this script, missing columns: %s" % sorted(missing))
    order = np.argsort(cols["frame"], kind="stable")
    return {k: v[order] for k, v in cols.items()}


def gapped(frame, value):
    """Insert NaN across frames with no row so the line breaks instead of interpolating.

    Both logs are sampled per STREAMED frame and both drop frames (the receiver drops one whenever
    the previous is still draining, and a marker out of view leaves no row at all). Without this a
    10-frame dropout is drawn as a straight segment, which looks exactly like a slow steady drift
    -- the one artefact that would invert the reading of either plot.
    """
    f = np.arange(int(frame[0]), int(frame[-1]) + 1, dtype=float)
    v = np.full(f.shape, np.nan)
    v[(frame - frame[0]).astype(int)] = value
    return f, v


def robust_ylim(ax, v, floor_zero=False, lo=0.5, hi=99.5, pad=0.08):
    """Scale a panel to the BULK of its data and return the limits, instead of autoscaling.

    Single-frame pose blowups run to a metre (or hundreds of pixels), while the variation these
    plots exist to show is millimetres and single pixels. Autoscaled, three bad solves out of 1400
    flatten every good frame onto the zero line and the panel shows nothing but the spikes -- which
    are the least interesting thing in it, being a known and unfixable property of one small planar
    marker.

    A percentile range rather than a fixed clip, so it adapts to logs whose real spread is large (a
    step of half a metre must still fit, and does: at 0.5/99.5 anything short of a quarter of the
    samples survives the cut). mark_offscale() is the other half of the contract -- a clip that hid
    its excluded samples would be a lie, so they are drawn on the frame edge.

    NOT for every panel. A track whose PEAK is the number being read (a baseline that ramps up, a
    cumulative angle) must not have its top 0.5% clipped away; use it where outliers are noise, not
    where they are the measurement.
    """
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return None
    a, b = np.percentile(finite, [lo, hi])
    if b - a < 1e-9:                      # degenerate: a genuinely constant panel
        a, b = float(finite.min()), float(finite.max())
    if b - a < 1e-9:
        a, b = a - 1.0, b + 1.0
    margin = (b - a) * pad
    a, b = (0.0 if floor_zero else a - margin), b + margin
    ax.set_ylim(a, b)
    return a, b


def mark_offscale(ax, f, v, limits):
    """Draw whatever robust_ylim() cut out as markers on the frame edge, and count it.

    The point is that a clipped sample stays VISIBLE as an event at its own frame -- you can see
    that three spikes happened and when -- without being allowed to set the scale for the other
    1373.
    """
    if limits is None:
        return
    lo, hi = limits
    over = np.isfinite(v) & (v > hi)
    under = np.isfinite(v) & (v < lo)
    for mask, edge, marker in ((over, hi, "^"), (under, lo, "v")):
        if mask.any():
            ax.plot(f[mask], np.full(int(mask.sum()), edge), marker, color=C_MUTED,
                    markersize=4, linestyle="none", clip_on=False, zorder=5)
    n = int(over.sum()) + int(under.sum())
    if n:
        ax.text(0.995, 0.06, "%d off-scale" % n, transform=ax.transAxes, ha="right",
                va="bottom", fontsize=8, color=C_MUTED)


def trace(ax, frame, value, color, linestyle="-"):
    """Draw one track as a faint connecting line PLUS a dot at every recorded sample.

    The distinction is not decoration. Between two consecutive rows nothing was measured -- the
    line there is drawing, not data -- and at this sampling rate a short dropout renders as a
    straight segment that is indistinguishable from a smooth move. The dots say where the log
    actually has values, so wherever they thin out you can see that the line is an assumption.

    Returns the gapped (frame, value) pair, which is what robust_ylim and mark_offscale work on:
    they must see the same NaN-broken series that was drawn, not the dense sample array.
    """
    f, v = gapped(frame, value)
    ax.plot(f, v, color=color, linestyle=linestyle, linewidth=0.9, alpha=0.4, zorder=1)
    ax.plot(frame, value, linestyle="none", marker=".", markersize=2.5, color=color, alpha=0.95,
            zorder=2)
    return f, v


# Where sample_note() may sit. Four corners rather than free coordinates: the note has to dodge
# whatever the data is doing, and which corner is free differs per figure (a deviation panel is busy
# in the middle, a ramp is busy on the diagonal) -- but nothing needs finer placement than that.
_NOTE_CORNERS = {
    "upper left": (0.005, 0.94, "left", "top"),
    "upper right": (0.995, 0.94, "right", "top"),
    "lower left": (0.005, 0.06, "left", "bottom"),
    "lower right": (0.995, 0.06, "right", "bottom"),
}


def sample_note(ax, corner="upper left"):
    """The one-line legend for trace(): says that the dots are the data and the line is not.

    Worth stating on the figure rather than only in the docstring, because it is the difference
    between reading a dropout as missing data and reading it as a slow move -- and the reader who
    needs that most is the one who did not open the script.
    """
    x, y, ha, va = _NOTE_CORNERS[corner]
    ax.text(x, y, "dots = recorded samples; the line between them is drawn, not measured",
            transform=ax.transAxes, ha=ha, va=va, fontsize=8, color=C_MUTED)


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
