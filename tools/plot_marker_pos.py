#!/usr/bin/env python
"""Plot the marker-position log written while TcpDebugStream.record_global_marker_pos was on.

    py -3 tools/plot_marker_pos.py tools/marker_poses/marker_pos_20260819_143512.csv
    py -3 tools/plot_marker_pos.py tools/marker_poses/marker_pos_20260819_143512.csv 907:1930

The optional trailing frame range selects one continuous stretch, and exists because a recording
that contains a STEP cannot be read for noise:
the step sets every scale, and the millimetre structure the log is for collapses into two dots.
Find the step in the 'discontinuities' listing, then re-run on either side of it. The range lands in
the output filenames, so the whole-log figures survive alongside the per-segment ones.

What the log holds: one row per marker per STREAMED frame, `frame,id,x,y,z[,qx,qy,qz,qw]` = the
marker's POSE in WORLD space, i.e. what came out of solvePnP after head_pose * lens_pose was baked
onto it. The quaternion columns are present only in logs captured after _put_world_block gained them;
see ON ROTATION at the bottom of this docstring.

Every panel draws a DOT at each recorded sample over a faint connecting line. The line between two
dots is drawing, not data -- at this sampling rate a short dropout otherwise renders as a smooth
straight segment, which is exactly the artefact that would be read as a slow steady move.

WHAT THIS MEASURES. The marker does not move. So the world position is supposed to be a constant,
and every millimetre of variation in these curves is error -- there is no signal here to separate
from the noise, the whole trace IS the noise. That is what makes the log worth plotting rather
than watching: a constant is the easiest thing in the world to eyeball.

Which is why the panels show DEVIATION FROM THE MEDIAN in millimetres rather than the raw metres.
The raw numbers sit around 0.43, 1.18, -0.52 m, and the interesting variation is three or four
orders of magnitude below that -- plotted absolute, every panel is a flat line and the log looks
perfect. The median absolute position is printed in each panel's label instead, where it belongs.

HOW TO READ THE SHAPE, because the two failure modes look nothing alike:

  SCATTER about a fixed centre = solvePnP noise on a single small planar marker. Irreducible by
      calibration; the fix is temporal averaging or a multi-marker board. Expect a couple of mm.
  RAMP or STEP = the estimate is WANDERING, and that is a pose-chain problem. If it tracks your
      head motion it is the lens pose or the capture-time head lookup (the same error the red
      reprojection overlay lands in pixels); if it wanders with the head still, it is neither.
      'net drift' in the printout separates these two without needing the plot at all.

BEFORE READING ANY OF THAT, check the premise: all of it assumes the marker STAYED PUT for the
whole recording. Nothing in the log can tell a moved marker from a wandering estimate -- both are
a ramp -- so a drift of centimetres is far more likely to mean the marker (or the sheet it is on)
was nudged than to mean the calibration failed by that much. A static-marker recording is the only
kind this plot can interpret, and it is worth being sure the capture was one.

Figure 2 is the error CLOUD seen down two axes, and its shape carries the diagnosis the tracks
cannot: an isotropic blob is plain solvePnP noise, while a cloud elongated along the camera
bearing is a DEPTH error -- the classic single-marker failure, where the pose slides toward and
away from the camera far more freely than it slides sideways. A wrong aruco_patch_sizes entry or a
wrong focal length both look exactly like that, and both are cheap to check once you know to.

ON THE TIME AXIS: `frame` is tcp_receiver.py's counter of RECEIVED frames, not wall-clock time and
not a detection count. The stream is throttled and drops frames whenever the receiver falls behind
(see TcpDebugStream), so the spacing is only roughly uniform -- treat it as ordering, not duration.
Rows are absent for frames in which the marker was not detected, and gapped() below breaks the
line across those rather than drawing through them: a 10-frame dropout interpolated into a
straight segment is indistinguishable from a slow steady drift, which is the one artefact that
would invert the reading.

ON ROTATION: logs written after _put_world_block gained its quaternion carry qx,qy,qz,qw beside the
position, and get a third figure -- the same four-panel treatment, in degrees about the three world
axes, taken relative to the median orientation. Older five-column logs still plot; they simply say
that rotation is absent instead of quietly dropping a figure.

Read the rotation panels against a floor, not against zero: solvePnP's orientation on a single
small planar marker scatters ~2.2deg median / 5.2deg p90 however good the calibration is (see the
note at lens_rotation_raw in src/OpenCVProcessor.h). Scatter inside that band is the marker being
small and flat and no lens-pose work will shrink it. A STEP or a RAMP is the thing worth having --
noise does not produce either, and both mean the estimate settled somewhere new.
"""

import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# The drawing vocabulary shared with plot_reproj_log.py -- palette, axis styles, the NaN-gapping,
# the robust clipping and the dots-are-samples convention. Shared rather than copied so the two
# scripts cannot drift into making different promises about the same kind of panel; see plotlib.py.
from plotlib import (C_ACCENT, C_AXES, C_INK, C_MUTED, S_AXES, load, mark_offscale,
                     robust_ylim, sample_note, style, trace)

# Every track panel in this file carries the same x axis, and the parenthetical is the whole
# warning: this is a counter of received frames, so it orders the samples but does not time them.
X_LABEL = "frame  (receiver's streamed-frame counter, not wall time)"
# The columns a marker-position log must have for anything here to mean something.
REQUIRED_COLS = ("frame", "id", "x", "y", "z")

# The three world axes, in the order they sit in the CSV. Godot is Y-up, so y is HEIGHT -- worth
# naming in the labels, because a reader reaching for "which one is depth" will otherwise guess z
# and be wrong: these are world axes, not camera axes, and the camera bearing is some diagonal
# through all three.
AXES = [("x", "x  (world right)"), ("y", "y  (world up)"), ("z", "z  (world forward/back)")]

# The rotation panels. Named by the world axis turned about, with the aviation word beside it only
# as a reading aid -- these are WORLD axes, so "yaw" here means about world up, not about the
# marker's own normal.
ROT_AXES = [("rx", "about x  (pitch)"), ("ry", "about y  (yaw)"), ("rz", "about z  (roll)")]
ROT_COLS = ("qx", "qy", "qz", "qw")

# Frames at each end averaged into the "net drift" figure. Long enough that solvePnP scatter
# averages down (~30 samples cuts it by ~5x), short enough to still be one end of the recording.
DRIFT_WINDOW = 30


####################################################################################################
# --- Panel builders -------------------------------------------------------------------------------
#
# The two four-panel figures below (position and orientation) and the seven-panel quaternion/euler
# one are the same picture three times over -- per-axis tracks stacked on a shared frame axis, each
# clipped to its own bulk, plus in two cases a total-magnitude panel with p50/p90 rules. Written out
# three times, any improvement to the layout has to be made three times and will eventually be made
# once. So the PANEL is the shared unit, not the figure: each figure still composes its own panels
# in its own order with its own annotations, which is where they genuinely differ.

def axis_panel(ax, frame, values, color, linestyle, label, unit=None, zero_line=True):
    """One stacked track panel: the trace, its clipping, and the label.

    zero_line is for the deviation figures, where zero is the reference the whole panel is measured
    against; the raw-quaternion panels have no such datum and would just gain a rule through nothing.
    """
    f, v = trace(ax, frame, values, color, linestyle)
    if zero_line:
        ax.axhline(0.0, color=C_MUTED, linewidth=0.8, alpha=0.6)
    mark_offscale(ax, f, v, robust_ylim(ax, v))
    ax.set_ylabel("%s\n[%s]" % (label, unit) if unit else label, fontsize=9)
    style(ax)


def total_panel(ax, frame, values, label, rule_fmt):
    """The magnitude panel that closes the deviation figures: total distance / total angle.

    Its own panel rather than a fourth track, because it is the one number that cannot go negative
    and the one the p50/p90 rules belong to -- those are the summary the printout quotes, drawn
    where the reader can see which samples sit above them.
    """
    f, v = trace(ax, frame, values, C_INK)
    lim = robust_ylim(ax, v, floor_zero=True)
    for q, name in ((50, "p50"), (90, "p90")):
        y = np.percentile(values, q)
        ax.axhline(y, color=C_ACCENT, linewidth=0.8, linestyle=":", alpha=0.8)
        ax.text(f[0], y, rule_fmt % (name, y), fontsize=8, color=C_ACCENT, va="bottom")
    mark_offscale(ax, f, v, lim)
    ax.set_ylabel(label, fontsize=9)
    ax.set_xlabel(X_LABEL, fontsize=9)
    style(ax)


def corner_note(ax, text):
    """Figure-level annotation in the top right, where it cannot collide with sample_note()."""
    ax.text(0.995, 0.94, text, transform=ax.transAxes, ha="right", va="top", fontsize=8,
            color=C_MUTED)


####################################################################################################

def deviations_mm(one):
    """Per-axis deviation from the median, in mm, plus the 3D distance from the median point.

    MEDIAN rather than mean, throughout: a handful of frames where the detector latched a bad
    corner produce positions metres away, and a mean centred on those would shift the whole
    baseline and report the error as a constant offset instead of as the few outliers it is.
    """
    centre = np.array([np.median(one["x"]), np.median(one["y"]), np.median(one["z"])])
    dev = np.stack([one["x"], one["y"], one["z"]], axis=1) - centre
    return centre, dev * 1000.0, np.linalg.norm(dev, axis=1) * 1000.0


def unit_quats(one):
    """The log's four quaternion columns as an (N, 4) array, renormalised.

    Renormalised because they arrive as float32 off the wire and every consumer here assumes unit
    length -- rotvec_deg reads w as cos(angle/2), quat_to_matrix has no normalising step of its own.
    One helper rather than the four verbatim copies of this pair of lines this file used to carry:
    they cannot drift now, and the epsilon guard is stated once.
    """
    q = np.stack([one[c] for c in ROT_COLS], axis=1)
    return q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)


def _quat_hemisphere(q):
    """Flip every sample onto one hemisphere.

    q and -q are the SAME rotation. A log that happens to straddle the sign boundary would
    otherwise get a component-wise median sitting halfway between two IDENTICAL orientations,
    which is not an orientation at all -- and every deviation would then be measured from it.
    """
    sign = np.sign(q @ q[0])
    sign[sign == 0.0] = 1.0
    return q * sign[:, None]


def quat_median(q):
    """Robust central orientation: component-wise median on one hemisphere, renormalised.

    Median rather than mean for the same reason as everywhere else in this file -- the occasional
    frame where solvePnP picks the wrong branch of the planar ambiguity is off by tens of degrees,
    and an averaged reference would carry a piece of that into every other frame's deviation.
    """
    m = np.median(_quat_hemisphere(q), axis=0)
    n = float(np.linalg.norm(m))
    return m / n if n > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])


def quat_mul(a, b):
    """Hamilton product in the (x, y, z, w) component order -- Godot's, so the CSV columns go
    straight in as they came out of Basis.get_rotation_quaternion()."""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz], axis=-1)


def quat_conj(q):
    out = np.array(q, dtype=float, copy=True)
    out[..., :3] *= -1.0
    return out


def rotvec_deg(q):
    """Quaternion -> rotation vector (axis * angle) in degrees, along the shortest arc.

    A rotation VECTOR rather than euler angles, deliberately. Euler needs a convention stated to be
    read at all, wraps at +-180deg, and degenerates at gimbal lock -- three ways for a plot to show
    a jump that never happened. Taken relative to the median orientation these angles are a couple
    of degrees, nowhere near any of those failure modes, and each component reads simply as "how far
    about that world axis".
    """
    q = np.atleast_2d(np.asarray(q, dtype=float))
    # q and -q are the same rotation; picking w >= 0 picks the <=180deg way round.
    q = q * np.where(q[:, 3:4] < 0.0, -1.0, 1.0)
    w = np.clip(q[:, 3], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    sin_half = np.sqrt(np.maximum(1.0 - w * w, 0.0))
    axis = np.zeros((len(q), 3))
    ok = sin_half > 1e-9          # at sin_half == 0 the rotation IS identity; axis stays zero
    axis[ok] = q[ok, :3] / sin_half[ok, None]
    return np.degrees(axis * angle[:, None])


def rotation_deviation_deg(one):
    """(median orientation, per-frame rotation vector from it in deg, total angle in deg)."""
    q = unit_quats(one)
    med = quat_median(q)
    rel = quat_mul(quat_conj(med)[None, :], q)
    rv = rotvec_deg(rel)
    return med, rv, np.linalg.norm(rv, axis=1)


def quat_to_matrix(q):
    """(N, 4) quaternions in (x, y, z, w) -> (N, 3, 3) rotation matrices."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1),
    ], axis=1)


def quat_to_euler_yxz_deg(q):
    """Quaternion -> euler angles in degrees, in Godot's DEFAULT order (EULER_ORDER_YXZ).

    The order is the whole reason this function exists rather than a one-liner: euler angles are
    meaningless without one, and the useful choice is the one the engine that produced the pose
    uses, so these numbers are directly comparable with Node3D.rotation_degrees in the (remote)
    inspector. A different order applied to the same rotation gives three different numbers, all
    correct, none comparable with anything.

    The gimbal-lock branches are transcribed from Basis::get_euler rather than left to fall over:
    at x = +-90deg the y and z axes coincide, y absorbs the whole remaining rotation and z is
    pinned to zero. Vectorised through np.where, so the degenerate case costs nothing and cannot
    produce a NaN that would silently break the line.
    """
    m = quat_to_matrix(q)
    m12 = np.clip(m[:, 1, 2], -1.0, 1.0)
    locked = np.abs(m12) > 1.0 - 1e-7

    x = np.where(locked, np.copysign(np.pi / 2.0, -m12), np.arcsin(-m12))
    y = np.where(locked,
                 np.copysign(1.0, -m12) * np.arctan2(m[:, 0, 1], m[:, 0, 0]),
                 np.arctan2(m[:, 0, 2], m[:, 2, 2]))
    z = np.where(locked, 0.0, np.arctan2(m[:, 1, 0], m[:, 1, 1]))

    # Unwrapped before the degree conversion. A marker sitting near the +-180deg seam otherwise
    # draws a full-scale vertical line every time the angle crosses it -- an artefact of the
    # representation that looks exactly like the pose flipping, which is the one thing this plot
    # must not invent. Unwrapping moves the values off the canonical range on purpose; read them
    # as a continuous track, not as canonical euler angles.
    return np.degrees(np.unwrap(np.stack([x, y, z], axis=1), axis=0))


def net_drift_mm(dev_mm):
    """Displacement from the first DRIFT_WINDOW frames to the last window's, per axis.

    This is the number that separates the two failure modes without reading the plot: pure
    solvePnP scatter averages toward zero over a window, so a recording that is only noisy has a
    net drift near zero however wide its per-frame spread. A drift comparable to the spread means
    the estimate actually WENT somewhere, which noise does not do.

    MEDIAN of each window, not the mean. Single-frame pose blowups of a metre or more are a normal
    feature of these logs -- solvePnP on one small planar marker occasionally picks the wrong
    branch of the two-fold ambiguity -- and one of those inside a 30-frame window shifts a mean by
    tens of mm, which is the whole scale of the number being reported.
    Returns None when the log is too short for two non-overlapping windows.
    """
    if len(dev_mm) < 2 * DRIFT_WINDOW:
        return None
    return (np.median(dev_mm[-DRIFT_WINDOW:], axis=0)
            - np.median(dev_mm[:DRIFT_WINDOW], axis=0))


def jumps_mm(one, dev_mm, threshold=50.0):
    """Frame-to-frame displacements above `threshold`, as (frame_before, frame_after, gap, d, dxyz).

    Reported separately from the spread statistics because a STEP and a SCATTER of the same
    magnitude mean entirely different things, and averaged together they hide each other. Two
    shapes turn up here and they are worth telling apart at a glance:

      A jump that REVERSES on the next frame, with gap 1, is a single bad solve -- the planar
      ambiguity firing once. Harmless, and no calibration change will remove it; reject it
      downstream by continuity if it matters.
      A jump that does NOT come back, especially across a multi-frame gap where the marker was out
      of view, is the estimate arriving at a NEW resting place. That is not solvePnP noise: either
      the marker moved while it was hidden, or the world frame moved under it (an OpenXR recentre
      or a tracking-loss relocalisation shifts the play space, which moves every world coordinate
      at once while leaving floor height alone). The log cannot distinguish those two on its own --
      a second, stationary marker in the same recording can, since a world-frame shift moves both
      and a nudged marker moves one.
    """
    out = []
    for i in range(1, len(dev_mm)):
        d = dev_mm[i] - dev_mm[i - 1]
        dist = float(np.linalg.norm(d))
        if dist > threshold:
            out.append((int(one["frame"][i - 1]), int(one["frame"][i]),
                        int(one["frame"][i] - one["frame"][i - 1]), dist, d))
    return out


def figure_tracks(one, mid, centre, dev_mm, dist_mm):
    """Three axis panels plus the total-distance panel, on one shared frame axis.

    Four separate panels rather than one with three lines: the axes routinely differ in spread by
    several times, and sharing a y-scale would flatten the quiet ones into the axis line. Sharing
    the x only is what keeps a step visible as SIMULTANEOUS across the three.

    Takes the deviations already computed rather than recomputing them: the caller prints
    statistics from the same arrays, and a second computation is a second median that could
    silently differ from the one quoted in the text.
    """
    fig, axes = plt.subplots(4, 1, figsize=(13, 9.5), sharex=True, layout="constrained")
    fig.suptitle("marker %d - world position over time (deviation from median)" % mid,
                 color=C_INK, fontsize=12)

    for i, (_key, label) in enumerate(AXES):
        axis_panel(axes[i], one["frame"], dev_mm[:, i], C_AXES[i], S_AXES[i], label, "mm")
        # The absolute position lives in the corner, not on the axis: it is context, and putting it
        # on the scale would cost the millimetre resolution this whole plot is for.
        corner_note(axes[i], "median %+.4f m" % centre[i])

    sample_note(axes[0])
    total_panel(axes[3], one["frame"], dist_mm, "|deviation|\n[mm]", " %s %.1f mm")
    return fig


def figure_rotation(one, mid, med, rv_deg, ang_deg):
    """The orientation counterpart of figure_tracks: three rotation panels plus the total angle.

    Deviation from the MEDIAN orientation, not absolute angles, for exactly the reason the position
    panels use deviation from the median position -- the absolute orientation is some arbitrary
    attitude in world axes, and plotting it would put the couple of degrees of variation that
    matter three orders of magnitude below the axis range.

    Read it against the known noise floor rather than against zero: solvePnP on a single small
    planar marker scatters ~2.2deg median / 5.2deg p90 no matter how good the calibration is, so a
    band of that width is the marker being small and flat and is not going to be improved by any
    lens-pose work. What IS worth looking at here is the same thing as in the position panels --
    a STEP or a RAMP, which noise does not produce, and which means the estimate settled somewhere
    new rather than jittering about one place.
    """
    fig, axes = plt.subplots(4, 1, figsize=(13, 9.5), sharex=True, layout="constrained")
    fig.suptitle("marker %d - world orientation over time (deviation from median)" % mid,
                 color=C_INK, fontsize=12)

    for i, (_key, label) in enumerate(ROT_AXES):
        axis_panel(axes[i], one["frame"], rv_deg[:, i], C_AXES[i], S_AXES[i], label, "deg")

    # The reference orientation itself, printed once rather than per panel: it is the same
    # quaternion for all three and belongs to the figure, not to any one axis.
    corner_note(axes[0], "median quat  (%+.4f, %+.4f, %+.4f, %+.4f)" % tuple(med))
    sample_note(axes[1])
    total_panel(axes[3], one["frame"], ang_deg, "total angle\n[deg]", " %s %.2f deg")
    return fig


def figure_quat_euler(one, mid):
    """The raw quaternion and the euler angles derived from it, on ONE shared frame axis.

    Together rather than in two figures, because the point is the correspondence: every feature in
    the euler tracks has to be present in the quaternion above it, and anything that appears in
    only one of them is an artefact of the euler conversion rather than something the marker did.
    That is not a hypothetical -- the two conventions fail in different places, and having them
    stacked is what makes the difference legible instead of a matter of trust.

    ABSOLUTE values here, unlike figure_rotation's deviation-from-median. This is the "what did the
    device actually report" view; the panels still resolve millimetre-scale detail because
    robust_ylim zooms each one to its own data rather than to the [-1, 1] a quaternion could span.

    The quaternion is hemisphere-aligned first. Without that the panels are shot through with
    full-scale sign flips that are not rotations at all -- q and -q are the same orientation, and
    the device has no reason to prefer one, so the raw stream genuinely does alternate.
    """
    q = _quat_hemisphere(unit_quats(one))
    eul = quat_to_euler_yxz_deg(q)

    fig, axes = plt.subplots(7, 1, figsize=(13, 14), sharex=True, layout="constrained")
    fig.suptitle("marker %d - orientation as reported (quaternion) and as converted (euler YXZ)"
                 % mid, color=C_INK, fontsize=12)

    for i, name in enumerate(ROT_COLS):
        # Fourth colour would be a new hue for no new meaning; w is the scalar part and reads as
        # the odd one out, so it carries the ink colour instead.
        # No zero rule: these are absolute components, and zero is not a datum any of them is
        # measured from (unlike the deviation figures, where it is the reference).
        axis_panel(axes[i], one["frame"], q[:, i], C_AXES[i] if i < 3 else C_INK,
                   S_AXES[i] if i < 3 else "-", name, zero_line=False)

    corner_note(axes[0], "hemisphere-aligned (q and -q are the same rotation)")

    for i, (_key, label) in enumerate(ROT_AXES):
        axis_panel(axes[4 + i], one["frame"], eul[:, i], C_AXES[i], S_AXES[i],
                   "euler %s" % label, "deg", zero_line=False)

    corner_note(axes[4], "Godot order YXZ, unwrapped - comparable with rotation_degrees")
    sample_note(axes[1])
    axes[6].set_xlabel(X_LABEL, fontsize=9)
    return fig


def figure_cloud(one, mid, dev_mm):
    """The error cloud down two orthogonal views, both on a shared equal-aspect mm scale.

    Equal aspect and a shared range are the entire point -- the shape is the diagnosis, and any
    cloud can be made to look round or stretched by choosing axis limits. Round = plain solvePnP
    scatter. Elongated = the estimate slides freely along one direction, and if that direction is
    the camera bearing it is a DEPTH error (wrong marker size or wrong focal length), not noise.
    """
    # Percentile, not max, for the same reason the track panels clip: the SHAPE of the cloud is the
    # entire diagnosis here, and a handful of metre-scale bad solves would shrink every real
    # structure in it to a couple of pixels around the origin.
    lim = float(np.percentile(np.abs(dev_mm), 99.5)) * 1.15 + 1e-6
    outside = int(np.count_nonzero(np.any(np.abs(dev_mm) > lim, axis=1)))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), layout="constrained")
    title = "marker %d - error cloud (deviation from median)" % mid
    if outside:
        # Never silently: a clipped scatter that does not say so reads as a complete picture.
        title += "   [%d of %d samples outside the frame]" % (outside, len(dev_mm))
    fig.suptitle(title, color=C_INK, fontsize=12)

    for ax, (i, j, xl, yl, title) in zip(axes, [
            (0, 2, "x  (world right) [mm]", "z  (world forward/back) [mm]", "top-down"),
            (0, 1, "x  (world right) [mm]", "y  (world up) [mm]", "from the front")]):
        # Coloured by frame so the cloud carries its own time axis: a drift reads as a colour
        # gradient across the blob, which is exactly the case a static scatter plot would hide.
        sc = ax.scatter(dev_mm[:, i], dev_mm[:, j], c=one["frame"], cmap="viridis",
                        s=7, alpha=0.75, linewidths=0)
        ax.axhline(0.0, color=C_MUTED, linewidth=0.8, alpha=0.5)
        ax.axvline(0.0, color=C_MUTED, linewidth=0.8, alpha=0.5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(xl, fontsize=9)
        ax.set_ylabel(yl, fontsize=9)
        ax.set_title(title, fontsize=10, color=C_INK)
        style(ax)
    fig.colorbar(sc, ax=axes, label="frame", shrink=0.85)
    return fig


def out_path(src, suffix):
    """PNG name derived from the CSV name, written into a marker_pose_plots/ subfolder beside it.

    Same naming rule as plot_reproj_log.py: these logs exist to be compared against each other, one
    capture per configuration, so a fixed output name would overwrite the run being compared to.
    The stem carries tcp_receiver.py's capture timestamp, so each figure says which recording it
    came from.

    The subfolder is where this one differs, and the reason is arithmetic: a single recording with
    two markers in it produces up to EIGHT figures, so after a couple of sessions the recordings --
    the things you actually pick from -- are a handful of CSVs scattered through a hundred PNGs.
    Derived from the source directory rather than hardcoded, so a CSV copied somewhere else still
    keeps its figures with it.
    """
    stem = os.path.splitext(os.path.basename(src))[0]
    out_dir = os.path.join(os.path.dirname(src) or ".", "marker_pose_plots")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, "%s_%s.png" % (stem, suffix))


def parse_range(arg):
    """"first:last" -> inclusive bounds. Either side may be omitted ("1968:" runs to the end)."""
    lo, sep, hi = arg.partition(":")
    if not sep:
        raise SystemExit('frame range must look like 907:1930 (got "%s")' % arg)
    return (int(lo) if lo else -(2 ** 62), int(hi) if hi else 2 ** 62)


####################################################################################################
# --- The printout -----------------------------------------------------------------------------------
#
# One function per SECTION of it, in the order they print. They were one 160-line loop body, which
# made the sections legible only by their own comments; each is a self-contained reading of the same
# arrays, so each gets a name. They print rather than return: the printout is the product here,
# read alongside the figures, and threading it back through a caller would only add a format step.

def report_position(one, mid, sel_count, centre, dev_mm, dist_mm):
    """Rows, span, the median position, and the per-axis spread about it."""
    span = int(one["frame"][-1] - one["frame"][0]) + 1
    print("\nmarker %d: %d rows over %d frames (%d with no detection)"
          % (mid, sel_count, span, span - sel_count))
    print("  median position   x=%+.4f  y=%+.4f  z=%+.4f  m" % tuple(centre))
    print("  per-axis deviation from that median, in mm:")
    # rsd = 1.4826 * median absolute deviation: the same quantity as a standard deviation for
    # normally distributed data, but computed from medians so a handful of metre-scale bad
    # solves cannot set it. A plain sd here read 82 mm on a stretch whose p90 was 2.4 mm --
    # three samples out of a thousand, describing nothing anyone wanted to know.
    # max|d| and range are deliberately NOT robust: their job is to report the worst single
    # frame, which is exactly what rsd is built to ignore. Read them as a pair.
    print("    %-6s %8s %8s %8s %8s" % ("axis", "rsd", "p90|d|", "max|d|", "range"))
    for i, (key, _label) in enumerate(AXES):
        col = dev_mm[:, i]
        rsd = 1.4826 * np.median(np.abs(col - np.median(col)))
        print("    %-6s %8.2f %8.2f %8.2f %8.2f"
              % (key, rsd, np.percentile(np.abs(col), 90),
                 np.abs(col).max(), col.max() - col.min()))
    print("  3D distance from median   p50=%.2f  p90=%.2f  max=%.2f  mm"
          % (np.percentile(dist_mm, 50), np.percentile(dist_mm, 90), dist_mm.max()))


def report_cloud_shape(one, dev_mm, dist_mm, has_rot):
    """Elongation, principal axis, and whether that axis is the viewing direction.

    The cloud figure shows the shape; this puts a number on it, because "looks stretched" is not a
    threshold anyone can act on. Near 1:1 is isotropic scatter -- the pose is equally constrained in
    every direction and there is nothing directional to explain. A large ratio means the estimate
    slides freely along ONE direction, and if that direction is the line from the camera to the
    marker it is DEPTH: the inherently soft direction of any single planar marker, and the one a
    wrong aruco_patch_sizes entry or a wrong focal length stretches further still.
    """
    # Computed on the inner 99% because the quantity of interest is the shape of the BULK. Left
    # whole, three metre-scale blowups dominate the covariance completely and the "principal axis"
    # reported is just the direction those three happened to point.
    keep = dist_mm < np.percentile(dist_mm, 99.0)
    if int(keep.sum()) < 8:
        return
    evals, evecs = np.linalg.eigh(np.cov(dev_mm[keep].T))
    sigma = np.sqrt(np.maximum(evals[::-1], 0.0))
    axis = evecs[:, -1]
    print("  cloud shape   elongation %.1f:1   sigma %.2f / %.2f / %.2f mm on its own axes"
          % (sigma[0] / max(sigma[2], 1e-9), sigma[0], sigma[1], sigma[2]))
    print("                principal axis (x %+.2f, y %+.2f, z %+.2f) in world axes"
          % (axis[0], axis[1], axis[2]))
    if not has_rot:
        return
    # Whether that axis IS the viewing direction is the question, and this log holds no camera
    # position -- so the marker's own normal stands in for it. The proxy is good exactly as far as
    # the marker was viewed face-on and degrades with obliquity, which is why the angle is printed
    # rather than a verdict dressed up as one.
    normal = quat_to_matrix(quat_median(unit_quats(one))[None, :])[0][:, 2]
    # Both are undirected lines -- an elongation axis has no sign and neither does a normal for
    # this purpose -- so fold the angle onto 0-90deg.
    ang = float(np.degrees(np.arccos(np.clip(abs(float(axis @ normal)), 0.0, 1.0))))
    print("                %.0f deg off the marker normal  (%s)"
          % (ang, "along it -> consistent with a DEPTH error" if ang < 30.0
             else "across it -> not a depth error"))


def report_drift(dev_mm):
    """Net drift between the two ends of the recording, with how to read it against the spread."""
    drift = net_drift_mm(dev_mm)
    if drift is None:
        print("  net drift   (log shorter than %d frames, not computed)" % (2 * DRIFT_WINDOW))
        return
    # THE number for telling the two failure modes apart. Scatter averages toward zero over a
    # window, so a log that is merely noisy drifts far less than it spreads; a net drift comparable
    # to the p90 spread means the estimate actually went somewhere, and that is a pose-chain
    # question (lens pose, capture-time head lookup) rather than a solvePnP one.
    print("  net drift   first %d frames -> last %d:  x=%+.2f  y=%+.2f  z=%+.2f  "
          "|d|=%.2f mm" % (DRIFT_WINDOW, DRIFT_WINDOW, drift[0], drift[1], drift[2],
                           float(np.linalg.norm(drift))))
    print("              compare against p90 above: >= it means WANDER, << it means"
          " scatter about a fixed point")


def report_jumps(one, dev_mm):
    """Frame-to-frame steps above the threshold.

    Printed after the spread, and read BEFORE it: one unreturned step puts most of the log on the
    far side of the median, which inflates every sd and p90 above into a description of the step
    rather than of the noise. If a step shows up here, the spread numbers are answering the wrong
    question and the per-side spread is the one worth having.
    """
    js = jumps_mm(one, dev_mm)
    if not js:
        print("  discontinuities   none above 50 mm")
        return
    print("  discontinuities   %d above 50 mm (frame gap in brackets):" % len(js))
    for f0, f1, gap, dist, d in js[:12]:
        print("    %5d -> %-5d [%2d]  %8.1f mm   dx=%+8.1f dy=%+8.1f dz=%+8.1f"
              % (f0, f1, gap, dist, d[0], d[1], d[2]))
    if len(js) > 12:
        print("    ... %d more" % (len(js) - 12))


def report_rotation(one, med, rv_deg, ang_deg):
    """The orientation counterpart of report_position, in degrees about the three world axes."""
    print("  median orientation  quat (%+.5f, %+.5f, %+.5f, %+.5f)" % tuple(med))
    print("  per-axis deviation from that median, in deg:")
    print("    %-6s %8s %8s %8s" % ("axis", "rsd", "p90|d|", "max|d|"))
    for i, (key, _label) in enumerate(ROT_AXES):
        col = rv_deg[:, i]
        rsd = 1.4826 * np.median(np.abs(col - np.median(col)))
        print("    %-6s %8.3f %8.3f %8.3f"
              % (key, rsd, np.percentile(np.abs(col), 90), np.abs(col).max()))
    # Against the ~2.2deg median / 5.2deg p90 that OpenCVProcessor.h records as solvePnP's own
    # floor on one small planar marker: at or under it there is nothing to chase.
    print("  total angle from median   p50=%.2f  p90=%.2f  max=%.2f  deg"
          % (np.percentile(ang_deg, 50), np.percentile(ang_deg, 90), ang_deg.max()))
    eul = quat_to_euler_yxz_deg(_quat_hemisphere(unit_quats(one)))
    print("  median euler YXZ (deg, Godot order)   x=%+.2f  y=%+.2f  z=%+.2f"
          % tuple(np.median(eul, axis=0)))


####################################################################################################

def write_figures(src, figures, mid, tag):
    """Save each (figure, suffix) pair under the CSV's own stem and return the paths.

    One loop over a list rather than a hand-written save block per group: which figures exist
    depends on whether the log has rotation columns, and that is a question about the LIST, not
    about how a figure gets written.
    """
    written = []
    for fig, suffix in figures:
        path = out_path(src, "%s_id%d%s" % (suffix, mid, tag))
        fig.savefig(path, dpi=130)
        written.append(path)
    return written


def process_marker(one, mid, sel_count, has_rot, src, tag):
    """The whole treatment of one marker id: the printout, then its figures.

    Everything is derived ONCE here and handed to both consumers -- the report functions quote the
    median the figures are centred on, so computing it twice would let the text and the picture
    disagree about the same number.
    """
    centre, dev_mm, dist_mm = deviations_mm(one)
    report_position(one, mid, sel_count, centre, dev_mm, dist_mm)
    report_cloud_shape(one, dev_mm, dist_mm, has_rot)
    report_drift(dev_mm)
    report_jumps(one, dev_mm)

    figures = [(figure_tracks(one, mid, centre, dev_mm, dist_mm), "tracks"),
               (figure_cloud(one, mid, dev_mm), "cloud")]

    # Logs captured before the quaternion columns existed still plot -- position only, and said so
    # out loud rather than by an absent figure nobody would notice was missing.
    if not has_rot:
        print("  rotation: not in this log (pre-quaternion capture), position figures only")
    else:
        med, rv_deg, ang_deg = rotation_deviation_deg(one)
        report_rotation(one, med, rv_deg, ang_deg)
        figures.append((figure_rotation(one, mid, med, rv_deg, ang_deg), "rotation"))
        figures.append((figure_quat_euler(one, mid), "quat_euler"))

    print("  written: %s" % " , ".join(write_figures(src, figures, mid, tag)))


def main():
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        return 1
    d = load(sys.argv[1], REQUIRED_COLS)

    # Applied before anything is computed, so the median, the spread and the clipping limits all
    # describe the selected stretch only. Filtering after the fact would leave the figures scaled
    # to data no longer in them.
    tag = ""
    if len(sys.argv) == 3:
        lo, hi = parse_range(sys.argv[2])
        keep = (d["frame"] >= lo) & (d["frame"] <= hi)
        if not keep.any():
            raise SystemExit("no rows in frame range %s" % sys.argv[2])
        d = {k: v[keep] for k, v in d.items()}
        tag = "_f%d-%d" % (int(d["frame"][0]), int(d["frame"][-1]))
        print("frame range %s: %d of %d rows" % (sys.argv[2], keep.sum(), keep.size))

    # Rotation columns are optional: logs captured before _put_world_block carried a quaternion have
    # five columns, and those must keep plotting rather than becoming unreadable.
    has_rot = all(c in d for c in ROT_COLS)
    ids = sorted(set(d["id"].astype(int)))

    for mid in ids:
        sel = d["id"].astype(int) == mid
        one = {k: v[sel] for k, v in d.items()}
        if sel.sum() < 2:
            print("marker %d: %d row(s), nothing to plot" % (mid, sel.sum()))
            continue

        process_marker(one, mid, int(sel.sum()), has_rot, sys.argv[1], tag)

    # Skipped under a headless backend, where it only warns. The PNGs are written either way.
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
