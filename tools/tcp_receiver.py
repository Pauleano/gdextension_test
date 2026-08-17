import socket
import cv2  
import numpy as np
import os

save_dir = "tools/images"
os.makedirs(save_dir, exist_ok=True)

WIDTH = 1280
HEIGHT = 1280

# Per-frame logging. Leave off while streaming: a print per frame on a Windows console blocks
# long enough to hold the loop below under the sender's frame rate, and every frame this loop is
# late is a frame the Quest has to drop.
VERBOSE = False


def log(*a):
    if VERBOSE:
        print(*a)


# Corner log, toggled with 'l'. One CSV row per marker per frame -- what you want for looking at
# the red-green gap across a whole sweep, rather than at isolated frames the way 's' does.
#
# Only corner 0 of each is written, and the pairing is meaningful because BOTH orders start at the
# same physical corner of the marker: the detector emits its corners in the order solvePnP's object
# points assume ({-h,+h}, {+h,+h}, {+h,-h}, {-h,-h}), which is exactly why the pose solve works at
# all. Picking "whichever corner sits highest in the image" separately for each would silently pair
# different physical corners as soon as the marker is rotated.
LOG_HEADER = "frame,id,green_x,green_y,red_x,red_y,dx,dy,dist,baseline_deg,baseline_m\n"


def next_log_path():
    """First unused reproj_logN.csv.

    Scanned at open time rather than counted in a variable, so a RESTARTED receiver does not
    clobber the previous run either. These logs exist to be compared against each other -- one
    capture per configuration (lens pose, head-pose source, marker size) -- and a fixed filename
    would silently destroy the run you are comparing against, which is the one mistake that costs
    a whole session on the headset.
    """
    n = 0
    while os.path.exists(os.path.join(save_dir, "reproj_log%d.csv" % n)):
        n += 1
    return os.path.join(save_dir, "reproj_log%d.csv" % n)

#have to create a tcp connection on socket 7007
#adb reverse tcp:7007 tcp:7007
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("0.0.0.0", 7007)) #alternative to 0.0.0.0 is 127.0.0.1
s.listen(1)

print("Waiting for Quest connection...")
comm_socket, addr = s.accept()
print("Connected:", addr)

def recvall(sock, size):
    data = b""
    while len(data) < size:
        packet = sock.recv(size - len(data))
        if not packet:
            return None
        data += packet
    return data


def recv_markers(sock):
    """One marker block, as appended after the image payload by open_cv_processor.gd::
    _send_frame_tcp: count u32, then per marker id u32 + 8 big-endian f32 = the 4 corners as
    x0,y0,..,x3,y3.

    TWO blocks arrive per frame, in this order and both always present (count 0 included):

      1. DETECTED -- the corners the device's detector found, mapped back out of its downscaled
         detection frame, so they line up with the full-size image in the payload. Drawn green.
      2. REPROJECTED -- the marker world poses from the PREVIOUS streamed frame, projected into
         THIS frame by OpenCVProcessor::project_marker_corners through the same intrinsics,
         distortion and lens_pose the detection uses. Drawn red.

    The gap between the two is the diagnostic. It is deliberately cross-frame: reprojecting a
    frame's own poses with its own head pose cancels exactly and would always draw a perfect
    square. Only a head pose that has CHANGED since the latch makes the calibration stop
    cancelling, so the gap is read together with the two baselines in the frame's trailer, never
    on its own -- sitting still drives it to zero however wrong the calibration is.

    WHICH baseline grew is the diagnosis, and they are not interchangeable:

      TURNING  exposes lens_translation (gap ~ fx * turn * lens_error / range). Strafing cannot:
               with the head orientation fixed the relative motion is a pure translation and t_L
               cancels out of L^-1 A L entirely.
      STRAFING exposes anything that puts the marker at the wrong DEPTH -- aruco_patch_size or
               the focal length -- since parallax against a wrongly-placed point grows with
               viewpoint change (gap ~ fx * strafe/range * |1 - 1/scale_error|).
      EITHER   exposes lens_rotation, and a timing error -- but timing scales with head SPEED, so
               it is the one that vanishes if you move to a new pose and then hold still.

    What makes this worth having: it lands in CAMERA PIXELS, with passthrough compositing and the
    eye projection out of the loop -- the two stages that make the same comparison ambiguous when
    you judge it by eye through the headset.

    Returns (ids, corners) ready for cv2.aruco.drawDetectedMarkers, or None on a closed
    connection. BOTH blocks and the trailer MUST be read for every frame, including ones we cannot
    display -- unread bytes stay in the stream and desync every frame after it.
    """
    raw = recvall(sock, 4)
    if raw is None:
        return None
    count = int.from_bytes(raw, "big")
    if count > 1000:
        print("implausible marker count, stream desynced:", count)
        return None

    ids, corners = [], []
    for _ in range(count):
        record = recvall(sock, 4 + 8 * 4)
        if record is None:
            return None
        ids.append(int.from_bytes(record[0:4], "big"))
        pts = np.frombuffer(record, dtype=">f4", count=8, offset=4).astype(np.float32)
        # drawDetectedMarkers wants one (1, 4, 2) array per marker
        corners.append(pts.reshape(1, 4, 2))
    return ids, corners


def corner_gap(detected, reprojected):
    """Mean corner distance in px, per id present in BOTH overlays.

    Only ids in both are comparable: an id the detector lost this frame has no green to measure
    against, and one that has just appeared is not in the latched reference yet.
    """
    det = dict(zip(detected[0], detected[1]))
    out = {}
    for mid, rep in zip(reprojected[0], reprojected[1]):
        if mid in det:
            d = rep.reshape(4, 2) - det[mid].reshape(4, 2)
            out[mid] = float(np.mean(np.linalg.norm(d, axis=1)))
    return out

image_num=0
log_file = None
log_path = None
frame_num = 0

while True:
    frame_num += 1
    log("waiting for 16-byte header...")
    header = recvall(comm_socket, 16)

    if header is None:
        print("header is None / connection closed")
        break

    log("got header bytes:", len(header), header)

    width = int.from_bytes(header[0:4], "big")
    height = int.from_bytes(header[4:8], "big")
    fmt = int.from_bytes(header[8:12], "big")
    size = int.from_bytes(header[12:16], "big")

    log("parsed header:", width, height, fmt, size)

    if width <= 0 or height <= 0 or size <= 0 or size > 50_000_000:
        print("invalid header, stopping")
        break

    log("waiting for payload:", size)
    data = recvall(comm_socket, size)

    if data is None:
        print("payload is None / connection closed")
        break

    log("got payload:", len(data))

    # Before any `continue` below: both marker blocks belong to this frame and have to leave the
    # stream either way, in order.
    markers = recv_markers(comm_socket)

    if markers is None:
        print("marker block is None / connection closed")
        break

    reprojected = recv_markers(comm_socket)

    if reprojected is None:
        print("reprojection block is None / connection closed")
        break

    raw_baseline = recvall(comm_socket, 8)

    if raw_baseline is None:
        print("baseline trailer is None / connection closed")
        break

    baseline_deg, baseline_m = np.frombuffer(raw_baseline, dtype=">f4", count=2).astype(float)

    ids, corners = markers
    reproj_ids, reproj_corners = reprojected

    arr = np.frombuffer(data, dtype=np.uint8)

    # Always end up 3-channel: drawDetectedMarkers needs colour to draw its green borders and red
    # first-corner dot, and frombuffer/reshape alone would hand it a read-only array to draw into.
    if fmt in (0, 2):   # FORMAT_L8 (Android plugin Y-plane) / FORMAT_R8: 1-channel grayscale
        img = cv2.cvtColor(arr.reshape((height, width)), cv2.COLOR_GRAY2BGR)
    elif fmt == 5:
        rgba = arr.reshape((height, width, 4))
        img = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    else:
        print("unsupported format:", fmt)
        continue

    # Drawn into a COPY, so `img` stays the clean frame: the 's' key below writes into
    # tools/images, which is the input set for cameraCalibration.py -- a burnt-in overlay there
    # would corrupt the corner detection it runs.
    view = img.copy()

    if ids:
        log("drawing markers:", ids)
        cv2.aruco.drawDetectedMarkers(view, corners, np.array(ids, dtype=np.int32).reshape(-1, 1))

    if reproj_ids:
        log("drawing reprojected:", reproj_ids)
        # ids deliberately None: this is the SAME marker drawn a second time, so the labels would
        # just overprint the green ones. Red = where the latched world pose says the marker should
        # be; how far it sits from green is the lens_pose + timing error in pixels.
        cv2.aruco.drawDetectedMarkers(view, reproj_corners, None, (0, 0, 255))

    # The readout. Both baselines are shown FIRST and always, because they are what decide whether
    # the gap beside them means anything -- a gap without a baseline is just noise, and the two
    # baselines implicate DIFFERENT things (turning -> lens_translation, strafing -> marker size /
    # focal length, either -> lens_rotation). Which one you grew is the diagnosis, so the readout
    # never collapses them into one number.
    gaps = corner_gap(markers, reprojected)
    status = "turn %5.1fdeg  strafe %4.2fm" % (baseline_deg, baseline_m)
    if baseline_deg < 20.0 and baseline_m < 0.15:
        status += "  (move -- gap is not meaningful yet)"
    elif gaps:
        status += "".join("  id%d %.1fpx" % (m, g) for m, g in sorted(gaps.items()))
    if log_file is not None:
        status += "   [REC]"
    cv2.putText(view, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1,
                cv2.LINE_AA)

    if log_file is not None:
        det = dict(zip(ids, corners))
        rep = dict(zip(reproj_ids, reproj_corners))
        # Intersection only, same reason as corner_gap: a row needs both corners to mean anything.
        for mid in sorted(set(det) & set(rep)):
            g = det[mid].reshape(4, 2)[0]
            r = rep[mid].reshape(4, 2)[0]
            log_file.write("%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.4f,%.5f\n" % (
                frame_num, mid, g[0], g[1], r[0], r[1],
                r[0] - g[0], r[1] - g[1], float(np.hypot(r[0] - g[0], r[1] - g[1])),
                baseline_deg, baseline_m))
        # Flushed per frame: a run usually ends with Ctrl-C or a closed socket, neither of which
        # gets to close the handle, and an unflushed buffer would take the tail of the run with it.
        log_file.flush()

    log("showing frame")
    cv2.imshow("Quest Feed", view)

    # 1 ms, not 100: waitKey is the loop's only pause, so waitKey(100) caps this receiver at ~10
    # frames per second no matter how fast the Quest sends. 'q', 's' and 'l' still register.
    k = cv2.waitKey(1) & 0xFF

    if k  == ord('q'):
        break
    elif k == ord('l'):
        if log_file is None:
            log_path = next_log_path()
            log_file = open(log_path, "w")
            log_file.write(LOG_HEADER)
            print(f"corner log started: {log_path}")
        else:
            log_file.close()
            log_file = None
            print(f"corner log stopped: {log_path}")
    elif k== ord('s'):

        filename = os.path.join(save_dir, f"img{image_num}.jpg")
        success = cv2.imwrite(filename, img)

        if success:
            print(f"image saved: {filename}")
            image_num += 1
        else:
            print(f"could not save image: {filename}")

if log_file is not None:
    log_file.close()
    print(f"corner log closed: {log_path}")

comm_socket.close()
s.close()
cv2.destroyAllWindows()