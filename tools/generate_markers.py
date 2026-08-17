"""Generate printable ARUCO_MIP_36h12 markers for the Godot ArUco project.

The C++ side (src/OpenCVProcessor.cpp) detects DICT_ARUCO_MIP_36h12, so markers printed from
the older DICT_4X4_50 will NOT be found any more -- everything has to be reprinted once.

IMPORTANT -- what "size" means here:
    --size-mm is the side of the BLACK SQUARE, edge to edge, INCLUDING the marker's own 1-module
    black border. That is exactly the length solvePnP is given via the OpenCVProcessor node's
    aruco_patch_sizes / aruco_patch_size. The white quiet zone around it is NOT part of that
    measurement. Getting this wrong scales every pose by the same factor, so measure the printed
    black square with a ruler and put THAT number into the inspector.

Usage:
    python tools/generate_markers.py                          # ids 0-9 at 50 mm, 300 dpi
    python tools/generate_markers.py --ids 1 --size-mm 100     # the one 100 mm marker
    python tools/generate_markers.py --ids 0,3,7 --size-mm 80 --dpi 600
"""

import argparse
import os

import cv2
import numpy as np

# The detector's dictionary. Must stay in step with OpenCVProcessor.cpp's constructor.
DICT_NAME = "DICT_ARUCO_MIP_36h12"
# White margin around the marker, in marker modules. ArUco needs at least 1 module of quiet zone
# to find the outer contour at all; 2 is cheap insurance against a tight paper crop.
QUIET_ZONE_MODULES = 2
# 6x6 payload + 1 module of black border on each side.
MODULES_ACROSS = 8


def parse_ids(spec):
    """Accept "0-9", "0,3,7" or a mix of both."""
    ids = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            ids.extend(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(part))
    return ids


def main():
    ap = argparse.ArgumentParser(description="Generate printable ARUCO_MIP_36h12 markers.")
    ap.add_argument("--ids", default="0-9", help='marker ids, e.g. "0-9" or "0,3,7" (default 0-9)')
    ap.add_argument("--size-mm", type=float, default=50.0,
                    help="side of the black square in mm, border included (default 50)")
    ap.add_argument("--dpi", type=int, default=300, help="print resolution (default 300)")
    ap.add_argument("--out", default="tools/markers", help="output directory")
    ap.add_argument("--no-label", action="store_true", help="omit the id/size caption")
    args = ap.parse_args()

    if not hasattr(cv2.aruco, DICT_NAME):
        raise SystemExit(
            "This OpenCV build has no cv2.aruco.%s (needs opencv-contrib-python >= 4.7).\n"
            "Installed: %s" % (DICT_NAME, cv2.__version__))
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT_NAME))

    ids = parse_ids(args.ids)
    if max(ids) > 249:
        raise SystemExit("ARUCO_MIP_36h12 holds 250 codes -- ids must be 0..249.")

    # Round the marker side to a whole number of modules so every module lands on an exact pixel
    # boundary. An unrounded side makes the printer resample the bits, which is a real source of
    # misreads at small print sizes.
    px_per_mm = args.dpi / 25.4
    module_px = max(1, round(args.size_mm * px_per_mm / MODULES_ACROSS))
    side_px = module_px * MODULES_ACROSS
    quiet_px = module_px * QUIET_ZONE_MODULES
    actual_mm = side_px / px_per_mm

    os.makedirs(args.out, exist_ok=True)
    print("%s | %d dpi | %d px per module | black square %d px = %.2f mm (requested %.2f)"
          % (DICT_NAME, args.dpi, module_px, side_px, actual_mm, args.size_mm))
    if abs(actual_mm - args.size_mm) > 0.05:
        print("NOTE: rounded to whole modules -- measure and use %.2f mm in the inspector."
              % actual_mm)

    for marker_id in ids:
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, side_px)

        canvas = np.full((side_px + 2 * quiet_px, side_px + 2 * quiet_px), 255, dtype=np.uint8)
        canvas[quiet_px:quiet_px + side_px, quiet_px:quiet_px + side_px] = marker

        if not args.no_label:
            # Caption goes on its own strip UNDER the quiet zone, never inside it.
            strip = np.full((int(module_px * 1.5), canvas.shape[1]), 255, dtype=np.uint8)
            cv2.putText(strip, "id=%d  %.1fmm  36h12" % (marker_id, actual_mm),
                        (2, int(module_px * 1.05)), cv2.FONT_HERSHEY_SIMPLEX,
                        module_px / 40.0, 0, max(1, module_px // 15), cv2.LINE_AA)
            canvas = np.vstack([canvas, strip])

        path = os.path.join(args.out, "aruco_36h12_id%d_%.0fmm.png" % (marker_id, actual_mm))
        cv2.imwrite(path, canvas)
        print("  wrote %s" % path)

    print("\nPrint at 100%% scale / 'actual size' -- any 'fit to page' scaling silently changes\n"
          "the physical marker size and therefore every pose solvePnP returns.")


if __name__ == "__main__":
    main()
