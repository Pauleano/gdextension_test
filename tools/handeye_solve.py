#!/usr/bin/env python

# Solve the camera->view transform (T_view_cam) from samples written by open_cv_processor.gd's
# handeye_capture. Only numpy is needed -- no cv2, no scipy.
#
# Why this exists: _lens_pose has to be the passthrough camera expressed in the OpenXR VIEW
# frame, but the only value any API hands out is Camera2's ACAMERA_LENS_POSE_*, which is
# referenced to the GYROSCOPE (LENS_POSE_REFERENCE == 1). The rotation between the IMU and the
# runtime's view space is not exposed by either side, so it cannot be looked up -- it has to be
# measured. That is what this does.
#
# The constraint: for a marker that does not move, its world pose
#     W = H_i @ L @ M_i
# is the SAME for every observation i, where H is the head pose (world) and M the marker pose
# (raw camera space). Eliminating W between two observations gives
#     A @ L = L @ B,     A = H_j^-1 @ H_i,   B = M_j @ M_i^-1
# i.e. the classic AX = XB hand-eye problem, solved here by Park & Martin's closed form:
# rotation from the matrix logs, then translation by linear least squares.
#
# Usage:
#     py -3 tools/handeye_solve.py tools/handeye_samples/handeye_samples_20260819_143512.jsonl
#
# Pull the file off the headset first, into tools/handeye_samples/. Godot's user:// lands in
# INTERNAL app storage on this device (/data/data/<pkg>/files/), which `adb pull` cannot reach --
# go through run-as, which works because the build is debuggable. Use exec-out rather than shell so
# the bytes come back without line-ending mangling.
#
# Two steps, because the device stamps each capture run with its start time (see HANDEYE_PREFIX in
# project/detection_diagnostics.gd) and the name is therefore not predictable from here. It is also
# printed to logcat when the capture opens and again when it closes, so `adb logcat` answers the
# same question if the headset is still running:
#     adb exec-out "run-as de.unigreifswald.opencvaruco ls files/"
#     adb exec-out "run-as de.unigreifswald.opencvaruco cat files/handeye_samples_20260819_143512.jsonl" > tools/handeye_samples/handeye_samples_20260819_143512.jsonl

import json
import sys
from itertools import combinations

import numpy as np

# Pairs whose relative rotation is tiny carry almost no information about the rotational part of
# L, and dividing by a near-zero angle amplifies their noise into the fit. 5deg is well clear of
# the 2deg movement gate the capture itself applies.
MIN_PAIR_ROT_DEG = 5.0
# All-pairs is O(n^2), so large captures get a random SUBSET -- emphatically not the first N in
# lexicographic order, which would be every pair involving the earliest ~50 samples and would
# throw away the rest of the run.
MAX_PAIRS = 20000
PAIR_SEED = 12345
# Outlier rejection. A handful of detections per run come back badly wrong (motion blur, a
# glancing view, a half-occluded marker), and least squares has no defence: one sample off by
# 250mm outweighs a hundred good ones. Samples are scored by how far H*L*M lands from the
# consensus world pose and the tail is dropped, then the solve is repeated on the survivors.
OUTLIER_MAD = 6.0        # keep within this many median-absolute-deviations
OUTLIER_FLOOR_MM = 15.0  # ...but never reject inside this radius, however tight the run
ROBUST_ROUNDS = 3


def load(path):
    """Read the JSONL capture into {marker_id: (heads, markers)} as arrays of 4x4 matrices."""
    by_id = {}
    with open(path) as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # A force-stop mid-write can truncate the final line; everything before it is fine.
                print("skipping malformed line %d (truncated capture?)" % line_no)
                continue
            h = to_4x4(rec["head"])
            m = to_4x4(rec["marker_cam"])
            by_id.setdefault(rec["id"], ([], []))
            by_id[rec["id"]][0].append(h)
            by_id[rec["id"]][1].append(m)
    return {k: (np.array(v[0]), np.array(v[1])) for k, v in by_id.items()}


def to_4x4(flat12):
    """The capture writes a row-major 3x4 [R | t]; make it homogeneous."""
    t = np.eye(4)
    t[:3, :4] = np.array(flat12, dtype=np.float64).reshape(3, 4)
    return t


def log_rot(r):
    """Rotation matrix -> rotation vector (axis * angle). The inverse of Rodrigues."""
    cos = (np.trace(r) - 1.0) / 2.0
    cos = min(1.0, max(-1.0, cos))
    theta = np.arccos(cos)
    if theta < 1e-9:
        return np.zeros(3)
    if theta > np.pi - 1e-6:
        # Near 180deg the skew part vanishes and the formula below is useless; recover the axis
        # from the diagonal of (R + I) instead, whose columns are all parallel to it.
        w = np.sqrt(np.maximum((np.diag(r) + 1.0) / 2.0, 0.0))
        axis = w / (np.linalg.norm(w) + 1e-12)
        return axis * theta
    v = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    return v * (theta / (2.0 * np.sin(theta)))


def mat_to_quat(r):
    """Rotation matrix -> (x, y, z, w), the component order Godot's Quaternion() takes."""
    tr = np.trace(r)
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w, x, y, z = (r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w, x, y, z = (r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w, x, y, z = (r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s
    return np.array([x, y, z, w])


def quat_mul(a, b):
    """Hamilton product in (x, y, z, w) order, matching Godot's Quaternion operator*."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def build_pairs(heads, markers):
    """A = H_j^-1 H_i and B = M_j M_i^-1 for every usefully-separated pair of observations."""
    n = len(heads)
    idx = list(combinations(range(n), 2))
    if len(idx) > MAX_PAIRS:
        # Uniform subset over the WHOLE run. Truncating the lexicographic order instead would
        # silently restrict the fit to pairs involving the first ~sqrt(2*MAX_PAIRS) samples.
        rng = np.random.default_rng(PAIR_SEED)
        idx = [idx[k] for k in rng.choice(len(idx), MAX_PAIRS, replace=False)]
    a_list, b_list = [], []
    for i, j in idx:
        a = np.linalg.inv(heads[j]) @ heads[i]
        if np.degrees(np.linalg.norm(log_rot(a[:3, :3]))) < MIN_PAIR_ROT_DEG:
            continue
        a_list.append(a)
        b_list.append(markers[j] @ np.linalg.inv(markers[i]))
    return a_list, b_list


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def refine_positions(by_id, l0, iters=12, trim=0.9):
    """Position-only refinement of L. This is the estimator that actually works here.

    AX = XB (above) is driven entirely by the RELATIVE ROTATIONS of the marker poses, and
    solvePnP's orientation for a small planar marker is the worst-conditioned thing in the whole
    pipeline -- measured on a real capture it scatters by 8-15deg while the marker's POSITION is
    good to a few mm. Feeding the noisy channel to a rotation-only estimator is what made the
    closed form return an unusable answer.

    So treat the marker as a POINT and use only what is trustworthy:
        p_world = H_i (R_L p_i + t_L),  p_world an extra unknown, one per marker id.
    Linearising about the current estimate (R_L <- R_L (I + [d]x), t_L <- t_L + tau) gives
        residual_i = c_i - R_Hi R_L [p_i]x d + R_Hi tau - p_world
    which is linear in (d, tau, p_world) -- plain least squares, re-linearised a few times.
    Each observation constrains d only perpendicular to its own p_i, so the rotation becomes
    observable through the marker appearing in DIFFERENT parts of the image; check the reported
    conditioning if a capture kept the marker dead centre throughout.
    """
    l = l0.copy()
    ids = sorted(by_id.keys())
    cond = None
    for _ in range(iters):
        rows, rhs = [], []
        # One p_world unknown per marker id, packed after the shared (d, tau).
        width = 6 + 3 * len(ids)
        for slot, mid in enumerate(ids):
            heads, markers = by_id[mid]
            c = np.array([(h @ l @ m)[:3, 3] for h, m in zip(heads, markers)])
            err = np.linalg.norm(c - np.median(c, axis=0), axis=1)
            keep = err <= np.quantile(err, trim)     # trim the worst tail each round
            for h, m, ci, k in zip(heads, markers, c, keep):
                if not k:
                    continue
                block = np.zeros((3, width))
                block[:, 0:3] = -h[:3, :3] @ l[:3, :3] @ skew(m[:3, 3])
                block[:, 3:6] = h[:3, :3]
                block[:, 6 + 3 * slot:9 + 3 * slot] = -np.eye(3)
                rows.append(block)
                rhs.append(-ci)
        a = np.vstack(rows)
        sol = np.linalg.lstsq(a, np.concatenate(rhs), rcond=None)[0]
        cond = np.linalg.svd(a[:, 0:3], compute_uv=False)
        d, tau = sol[0:3], sol[3:6]
        th = np.linalg.norm(d)
        if th > 1e-12:
            k = skew(d / th)
            dr = np.eye(3) + np.sin(th) * k + (1.0 - np.cos(th)) * (k @ k)
        else:
            dr = np.eye(3)
        l = l.copy()
        l[:3, :3] = l[:3, :3] @ dr
        l[:3, 3] = l[:3, 3] + tau
        u, _, vt = np.linalg.svd(l[:3, :3])           # keep it a rotation
        l[:3, :3] = u @ vt
    return l, cond


def mean_rotation(rots):
    """Chordal mean of a set of rotations: sum them, project back onto SO(3)."""
    u, _, vt = np.linalg.svd(rots.sum(axis=0))
    m = u @ vt
    if np.linalg.det(m) < 0:
        u[:, -1] *= -1
        m = u @ vt
    return m


def sample_error_mm(heads, markers, l):
    """Distance of each sample's implied world position from the run's consensus position."""
    pos = np.array([(h @ l @ m)[:3, 3] for h, m in zip(heads, markers)])
    return np.linalg.norm(pos - np.median(pos, axis=0), axis=1) * 1000.0


def solve_robust(by_id, verbose=True):
    """Solve, score every sample against the consensus, drop the tail, repeat."""
    keep = {mid: np.ones(len(v[0]), dtype=bool) for mid, v in by_id.items()}
    l = None
    for rnd in range(ROBUST_ROUNDS):
        a_all, b_all = [], []
        for mid, (heads, markers) in sorted(by_id.items()):
            a, b = build_pairs(heads[keep[mid]], markers[keep[mid]])
            a_all += a
            b_all += b
        if len(a_all) < 10:
            return None, keep, len(a_all)
        l = solve_axxb(a_all, b_all)
        if rnd == ROBUST_ROUNDS - 1:
            break
        dropped = 0
        for mid, (heads, markers) in sorted(by_id.items()):
            err = sample_error_mm(heads, markers, l)
            inl = err[keep[mid]]
            mad = np.median(np.abs(inl - np.median(inl))) or 1.0
            limit = max(OUTLIER_MAD * mad, OUTLIER_FLOOR_MM)
            new = err <= limit
            dropped += int((keep[mid] & ~new).sum())
            keep[mid] = new
        if verbose:
            total = sum(int(k.sum()) for k in keep.values())
            print("  round %d: pairs=%-6d dropped %d outlier(s), %d samples remain"
                  % (rnd + 1, len(a_all), dropped, total))
        if dropped == 0:
            break
    return l, keep, len(a_all)


def solve_axxb(a_list, b_list):
    """Park & Martin: rotation from the log map, then translation by least squares."""
    # Rotation. M = sum(beta alpha^T); R = (M^T M)^-1/2 M^T.
    m = np.zeros((3, 3))
    for a, b in zip(a_list, b_list):
        alpha = log_rot(a[:3, :3])
        beta = log_rot(b[:3, :3])
        m += np.outer(beta, alpha)
    mtm = m.T @ m
    vals, vecs = np.linalg.eigh(mtm)
    # Clamped because a rank-deficient M (all rotation axes parallel -- e.g. only ever yawing)
    # produces a near-zero eigenvalue whose inverse square root would blow up.
    inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, 1e-12))) @ vecs.T
    rot = inv_sqrt @ m.T

    # Re-orthonormalise: the product above is only orthogonal up to numerical error.
    u, _, vt = np.linalg.svd(rot)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1
        rot = u @ vt

    # Translation. From A X = X B: (R_A - I) t_X = R_X t_B - t_A, stacked over all pairs.
    lhs, rhs = [], []
    for a, b in zip(a_list, b_list):
        lhs.append(a[:3, :3] - np.eye(3))
        rhs.append(rot @ b[:3, 3] - a[:3, 3])
    trans = np.linalg.lstsq(np.vstack(lhs), np.concatenate(rhs), rcond=None)[0]

    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = trans
    return out


def residuals(heads, markers, l):
    """Spread of H_i L M_i. It should be ONE world pose; how tightly it clusters is the score."""
    worlds = np.array([h @ l @ m for h, m in zip(heads, markers)])
    pos = worlds[:, :3, 3]
    pos_mm = np.linalg.norm(pos - np.median(pos, axis=0), axis=1) * 1000.0
    # Measured against the mean rotation, not against worlds[0] -- if sample 0 happened to be a
    # bad detection, every other sample would be scored against it and the run would look broken.
    ref = mean_rotation(worlds[:, :3, :3])
    rot_deg = np.array([np.degrees(np.linalg.norm(log_rot(ref.T @ w[:3, :3]))) for w in worlds])
    return pos_mm, rot_deg


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        print("usage: python handeye_solve.py <handeye_samples.jsonl>")
        return 1
    by_id = load(sys.argv[1])
    if not by_id:
        print("no samples found")
        return 1

    # Every marker id observes the same rigid L, so pooling their pairs constrains one solve
    # rather than producing one answer per marker. Pairs are still formed WITHIN an id -- the
    # constraint is that a given marker holds still, not that two markers coincide.
    for mid, (heads, _) in sorted(by_id.items()):
        print("marker %-4s samples=%d" % (mid, len(heads)))

    l_init, _keep, npairs = solve_robust(by_id, verbose=False)
    if l_init is None:
        print("\nnot enough separated pairs (%d). Recapture with more head ROTATION -- pitch and "
              "yaw both, not just position changes." % npairs)
        return 1
    print("\nAX=XB seed from %d pairs (starting point only -- see refine_positions)" % npairs)

    l, cond = refine_positions(by_id, l_init)

    print("\n--- T_view_cam ---")
    np.set_printoptions(precision=8, suppress=True)
    print(l)

    print("\nposition residual of H*L*M (the number that matters):")
    for mid, (heads, markers) in sorted(by_id.items()):
        pos_mm, rot_deg = residuals(heads, markers, l)
        print("  marker %-4s n=%-4d pos p50=%5.1fmm p90=%5.1fmm | rot p50=%5.2fdeg p90=%5.2fdeg"
              % (mid, len(heads), np.percentile(pos_mm, 50), np.percentile(pos_mm, 90),
                 np.percentile(rot_deg, 50), np.percentile(rot_deg, 90)))
    print("  rotation observability: %s (ratio %.1f; over ~20 means the marker stayed too"
          % (np.round(cond, 2), cond[0] / cond[-1]))
    print("  central in frame for one rotation axis to be pinned down)")
    print("\n  A few mm on POSITION is the target and means it converged. Centimetres means it")
    print("  did not -- check the marker never moved, no recentring happened mid-capture, and")
    print("  that the head both rotated and changed viewpoint.")
    print("  The ROTATION figure is NOT a calibration error: solvePnP's orientation for one small")
    print("  planar marker is inherently noisy at these distances, and it stays high no matter")
    print("  how good L is. It is why a patch visibly wobbles in orientation while sitting in the")
    print("  right place -- averaging over frames or using a multi-marker board is the fix for")
    print("  that, not a better lens pose.")

    # Nothing takes T_view_cam directly: the two values below are properties of the OpenCVProcessor
    # node, and rebuild_lens_pose() (src/OpenCVProcessor.cpp) derives lens_pose from them as
    # Transform3D(Basis((lens_rotation_raw * Q180X).inverse()), lens_translation), with
    # Q180X = Quaternion(1,0,0,0). Inverting that gives lens_rotation_raw = R_L^-1 * Q180X^-1,
    # so the literals below drop straight in with no code change.
    q_l = mat_to_quat(l[:3, :3])
    q_l_inv = np.array([-q_l[0], -q_l[1], -q_l[2], q_l[3]])
    raw = quat_mul(q_l_inv, np.array([-1.0, 0.0, 0.0, 0.0]))
    if raw[3] < 0:
        raw = -raw          # same rotation, and matches the sign convention of the dumped values
    print("\n--- paste onto the OpenCVProcessor node (project/aruco_markers.tscn, or its Inspector) ---")
    print("lens_rotation_raw = Quaternion(%.14f, %.14f, %.14f, %.14f)"
          % (raw[0], raw[1], raw[2], raw[3]))
    print("lens_translation = Vector3(%.14f, %.14f, %.14f)"
          % (l[0, 3], l[1, 3], l[2, 3]))
    print("\n  The setters rebuild lens_pose immediately, so these can go straight into the REMOTE")
    print("  inspector of a running deploy and take effect on the next frame -- no rebuild, no")
    print("  re-export. The compiled-in fallbacks are the initialisers in src/OpenCVProcessor.h.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
