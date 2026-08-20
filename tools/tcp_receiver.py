import collections
import socket
import cv2
import numpy as np
import os
import time

# Where each facility writes. One folder per KIND of capture rather than one shared dump: the three
# are produced by different switches, read by different scripts and thrown away on different
# schedules, and a single folder holding all of them made "which files belong to the run I am
# looking at" a question you had to answer by timestamp anyway.
#
# Anchored on THIS file, not on the working directory. The receiver is started from wherever the
# terminal happens to sit, and the old relative "tools/images" quietly created a second tree
# whenever that was not the repository root -- the frames went somewhere, just not where the
# calibration script looks.
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
# 's' key: single full-resolution frames. This is the input set cameraCalibration.py globs.
IMAGE_DIR = os.path.join(TOOLS_DIR, "saved_from_stream")
# 'l' key: the red-green corner log. plot_reproj_log.py writes its PNGs beside the CSV, i.e. in here.
REPROJ_DIR = os.path.join(TOOLS_DIR, "reproj_logs")
# The DEVICE switch (TcpDebugStream.record_global_marker_pos): marker world poses. plot_marker_pos.py
# writes into the marker_pose_plots/ subfolder rather than beside the CSV -- each recording produces
# up to four figures per marker, which buries the recordings themselves within a session or two.
POS_DIR = os.path.join(TOOLS_DIR, "marker_poses")

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
LOG_HEADER = ("frame,id,green_x,green_y,red_x,red_y,dx,dy,dist,"
              "baseline_deg,baseline_m,dx_m,dy_m,dz_m,green_px\n")

# Marker-position log, a SEPARATE facility from the corner log above. Not toggled from here at all:
# it follows TcpDebugStream.record_global_marker_pos on the device, so it can be started and stopped
# from the remote inspector with the headset on -- which is the whole point, since 'l' needs a hand
# on this keyboard. The two can run at the same time and neither knows about the other.
POS_HEADER = "frame,id,x,y,z,qx,qy,qz,qw\n"


def stamped_path(directory, stem, ext):
    """<directory>/<stem>_YYYYMMDD_HHMMSS.<ext>, with the directory created on demand.

    Stamped rather than counted (this used to hand out the first free <stem>N). Either scheme keeps
    a RESTARTED receiver from clobbering the previous run, which is the property that matters most
    -- these captures exist to be compared against each other, one per configuration (lens pose,
    head-pose source, marker size), and a fixed filename silently destroys the run you are comparing
    against, the one mistake that costs a whole session on the headset. What the stamp adds is WHEN:
    by the time three logs sit in a folder, "which one was reproj_log2" is not a question the
    filename can still answer, and it is the question you actually have. The stamp travels, too --
    both plot scripts derive their PNG names from the CSV stem, so a figure keeps its capture's
    identity instead of being pinned to a counter that means nothing a day later.

    Local time, not UTC: it is read against your memory of the session ("the run just before
    lunch"), and a stamp in a timezone you were not standing in cannot be read that way.

    A collision inside one second -- easy on the 's' key, the loop turns over every few ms -- gets a
    counter suffix rather than a finer stamp. `_2` reads as "same second, second shot", which is
    what happened; sub-second digits would just be noise in every other filename.
    """
    os.makedirs(directory, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(directory, "%s_%s.%s" % (stem, stamp, ext))
    n = 2
    while os.path.exists(path):
        path = os.path.join(directory, "%s_%s_%d.%s" % (stem, stamp, n, ext))
        n += 1
    return path

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


# 20 bytes: total_size, width, height, image_format, payload_size -- see _send_frame in
# project/tcp_debug_stream.gd, which is the other half of this format and has to change with it.
HEADER_SIZE = 20


class FrameError(Exception):
    """A frame that arrived WHOLE but did not parse.

    Distinct from a closed connection (read_frame returns None for that), and the distinction is
    the entire point of the total_size field: because the frame's own length said how many bytes
    belong to it, those bytes are off the socket before anything is interpreted. So a frame we
    cannot make sense of is a frame we can SKIP -- the next read still starts on a header. Before
    the length existed, every one of these was fatal and silent: the unread tail became the next
    frame's header and every frame after it was garbage.
    """


class Cursor:
    """Read head over one frame's bytes, in the wire's big-endian order.

    Every read is bounds-checked against the frame, so a field that claims more than the frame
    holds raises FrameError instead of quietly running into the next frame's data -- which, when
    this parsed straight off the socket, is precisely what it did.
    """

    def __init__(self, buf):
        self.buf = buf
        self.at = 0

    def take(self, n):
        end = self.at + n
        if n < 0 or end > len(self.buf):
            raise FrameError("field wants %d bytes, %d left in the frame" % (n, len(self.buf) - self.at))
        chunk = self.buf[self.at:end]
        self.at = end
        return chunk

    def u32(self):
        return int.from_bytes(self.take(4), "big")

    def f32(self, count):
        return np.frombuffer(self.take(4 * count), dtype=">f4").astype(float)

    def left(self):
        return len(self.buf) - self.at


def parse_markers(cur):
    """One marker block: count u32, then per marker id u32 + 8 big-endian f32 = the 4 corners as
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

    Returns (ids, corners) ready for cv2.aruco.drawDetectedMarkers.
    """
    count = cur.u32()
    ids, corners = [], []
    for _ in range(count):
        ids.append(cur.u32())
        # drawDetectedMarkers wants one (1, 4, 2) array per marker
        corners.append(cur.f32(8).astype(np.float32).reshape(1, 4, 2))
    return ids, corners


def parse_world(cur):
    """The world-position block, LAST in every frame, written by
    tcp_debug_stream.gd::_put_world_block: recording u32 (0/1), count u32, then per marker
    id u32 + x,y,z f32 + qx,qy,qz,qw f32 = the marker POSE in WORLD space, position first and
    orientation as a unit quaternion.

    The quaternion is stored verbatim rather than converted to euler here. Euler needs a
    convention, wraps at +-180deg and degenerates at gimbal lock, and a log that has already
    thrown the quaternion away cannot be re-derived past any of those. plot_marker_pos.py
    converts, and it does so RELATIVE to the median orientation, which sidesteps all three.

    Unlike the two corner blocks this one is not drawn -- it feeds a timestamped marker_pos CSV in
    marker_poses/, and it exists because the corner overlays answer a question in PIXELS while this
    answers one in METRES: where the pipeline thinks a marker actually is, and whether that estimate
    stays put.

    The recording flag is SENT rather than inferred from count, and reading it that way matters: a
    frame in which the detector found nothing also has count 0, so closing the file on an empty
    block would end the log at the first dropout instead of at the switch. Gaps in the file are
    therefore real gaps in DETECTION, not the end of the recording.

    Returns (recording, {id: (x, y, z, qx, qy, qz, qw)}).
    """
    recording = cur.u32()
    count = cur.u32()
    out = {}
    for _ in range(count):
        mid = cur.u32()
        out[mid] = cur.f32(7)
    return recording, out


# One received frame, fully parsed. A namedtuple rather than the eight loose locals the loop used to
# unpack, so "what a frame consists of" is stated once, in the one place that decodes it.
Frame = collections.namedtuple(
    "Frame", "width height fmt data detected reprojected baseline_deg d_local recording world")


def read_frame(sock):
    """Consume exactly one frame and return it parsed, or None once the stream is over.

    THE framing rule, and the only place that knows it: the first u32 is the frame's total length,
    counting itself. So this takes that many bytes off the socket and then parses them in memory,
    which is what makes every failure below survivable -- a field that lies, a format we cannot
    display, a sender that grew a block we do not know about. None of them can move the read head
    into the next frame, because the next read starts wherever total_size said it would.

    That property used to be maintained by hand instead: every block was read straight off the
    socket, and four comments through this file reminded whoever edited it that skipping one --
    a `continue` before the last block, an early `break` -- would desync the stream permanently.
    It is now structural, and this is the only function that has to be careful.

    Returns None when the connection closes or when the framing itself is unusable (a length that
    cannot be right -- there is no way to find the next frame after that). Raises FrameError when
    the frame arrived whole but its contents do not parse, which the caller can simply skip.
    """
    head = recvall(sock, HEADER_SIZE)
    if head is None:
        return None
    total, width, height, fmt, size = (
        int.from_bytes(head[i:i + 4], "big") for i in range(0, HEADER_SIZE, 4))

    # Fatal rather than a FrameError: with a length we cannot trust there is no next frame boundary
    # to resynchronise on. The upper bound is generous (a 4096x4096 RGBA frame is 64MB) and exists
    # only to fail loudly on a truncated or mismatched sender instead of trying to allocate 4GB.
    if total < HEADER_SIZE or total > 70_000_000:
        print("implausible frame length %d, stream is not recoverable" % total)
        return None

    body = recvall(sock, total - HEADER_SIZE)
    if body is None:
        return None

    cur = Cursor(body)
    if width <= 0 or height <= 0 or size <= 0:
        raise FrameError("bad frame geometry: %dx%d, payload %d" % (width, height, size))
    data = cur.take(size)
    detected = parse_markers(cur)
    reprojected = parse_markers(cur)
    # deg, then the displacement as three SIGNED components in the LATCHED HEAD's frame:
    # x right, y up, z backwards (Godot cameras look down -Z, so negative z moved toward the
    # marker). Sideways and vertical motion generate parallax and are what expose a wrong marker
    # size or focal length; forward/back changes range instead. The device sends axes rather than
    # a scalar so that distinction survives into the log -- the norm is derived, not sent.
    baseline_deg, dx_m, dy_m, dz_m = cur.f32(4)
    recording, world = parse_world(cur)

    # The mismatch alarm, and the reason the length is worth having even when nothing is wrong: a
    # receiver that is one field behind the device now says so on the first frame, by name, instead
    # of showing plausible garbage. Sender and receiver still have to be updated together.
    if cur.left():
        raise FrameError("%d bytes left over -- tcp_debug_stream.gd and this script disagree about "
                         "the frame format; update both together" % cur.left())

    return Frame(width, height, fmt, data, detected, reprojected,
                 float(baseline_deg), (float(dx_m), float(dy_m), float(dz_m)), recording, world)


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

log_file = None
log_path = None
# Second, independent log handle -- opened and closed by the DEVICE switch, never by a key here.
pos_file = None
pos_path = None
frame_num = 0

while True:
    log("waiting for frame...")
    try:
        frame = read_frame(comm_socket)
    except FrameError as exc:
        # Survivable by construction -- the frame's bytes are already off the socket, so the next
        # read starts on a header regardless of what went wrong in here. Counted like any other
        # frame so the logs' frame axis keeps meaning "frames received".
        frame_num += 1
        print("frame %d skipped: %s" % (frame_num, exc))
        continue

    if frame is None:
        print("connection closed")
        break

    frame_num += 1
    log("got frame:", frame.width, frame.height, frame.fmt, len(frame.data))

    markers = frame.detected
    reprojected = frame.reprojected
    baseline_deg = frame.baseline_deg
    dx_m, dy_m, dz_m = frame.d_local
    baseline_m = float(np.sqrt(dx_m * dx_m + dy_m * dy_m + dz_m * dz_m))
    recording, world_pos = frame.recording, frame.world

    # The log follows the device switch by EDGE, so the file spans exactly the interval
    # record_global_marker_pos was on. Turning the switch off is what closes and names the file --
    # there is deliberately no key for it, since it is meant to be driven with the headset on.
    if recording and pos_file is None:
        pos_path = stamped_path(POS_DIR, "marker_pos", "csv")
        pos_file = open(pos_path, "w")
        pos_file.write(POS_HEADER)
        print(f"marker position log started: {pos_path}")
    elif not recording and pos_file is not None:
        pos_file.close()
        pos_file = None
        print(f"marker position log stopped: {pos_path}")

    if pos_file is not None:
        for mid in sorted(world_pos):
            x, y, z, qx, qy, qz, qw = world_pos[mid]
            pos_file.write("%d,%d,%.6f,%.6f,%.6f,%.7f,%.7f,%.7f,%.7f\n"
                           % (frame_num, mid, x, y, z, qx, qy, qz, qw))
        # Flushed per frame, same reason as the corner log: a run usually ends with Ctrl-C or a dead
        # socket, neither of which reaches the close below, and an unflushed buffer would take the
        # tail of the recording with it.
        pos_file.flush()

    ids, corners = markers
    reproj_ids, reproj_corners = reprojected

    arr = np.frombuffer(frame.data, dtype=np.uint8)

    # Always end up 3-channel: drawDetectedMarkers needs colour to draw its green borders and red
    # first-corner dot, and frombuffer/reshape alone would hand it a read-only array to draw into.
    # The `continue` below is free of consequence now: read_frame took the whole frame off the
    # socket before any of this ran, so skipping one costs the frame and nothing after it.
    if frame.fmt in (0, 2):   # FORMAT_L8 (Android plugin Y-plane) / FORMAT_R8: 1-channel grayscale
        img = cv2.cvtColor(arr.reshape((frame.height, frame.width)), cv2.COLOR_GRAY2BGR)
    elif frame.fmt == 5:
        rgba = arr.reshape((frame.height, frame.width, 4))
        img = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    else:
        print("unsupported format:", frame.fmt)
        continue

    # Drawn into a COPY, so `img` stays the clean frame: the 's' key below writes into
    # tools/saved_from_stream, which is the input set for cameraCalibration.py -- a burnt-in overlay
    # there would corrupt the corner detection it runs.
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
    status = "turn %5.1fdeg  move %4.2fm (R%+.2f U%+.2f B%+.2f)" % (
        baseline_deg, baseline_m, dx_m, dy_m, dz_m)
    if baseline_deg < 20.0 and baseline_m < 0.15:
        status += "  (move -- gap is not meaningful yet)"
    elif gaps:
        status += "".join("  id%d %.1fpx" % (m, g) for m, g in sorted(gaps.items()))
    if log_file is not None:
        status += "   [REC]"
    # Separate marker, because these are separate facilities with separate switches -- seeing [REC]
    # and assuming positions are being recorded too is exactly the confusion worth avoiding.
    if pos_file is not None:
        status += "   [POS]"
    cv2.putText(view, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1,
                cv2.LINE_AA)

    if log_file is not None:
        det = dict(zip(ids, corners))
        rep = dict(zip(reproj_ids, reproj_corners))
        # Intersection only, same reason as corner_gap: a row needs both corners to mean anything.
        for mid in sorted(set(det) & set(rep)):
            gq = det[mid].reshape(4, 2)
            # Mean edge length of the DETECTED quad. Not decoration: it is inversely
            # proportional to the marker's range, so it removes the confound that walking
            # toward the marker shrinks r -- which sits in the denominator of the gap
            # formula and would otherwise masquerade as 'forward motion causes gap'.
            # Dimensionless too: gap/green_px needs no marker size to interpret.
            gpx = float(np.mean(np.linalg.norm(gq - np.roll(gq, 1, axis=0), axis=1)))
            g = gq[0]
            r = rep[mid].reshape(4, 2)[0]
            log_file.write("%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.4f,%.5f,%.5f,%.5f,%.5f,%.2f\n" % (
                frame_num, mid, g[0], g[1], r[0], r[1],
                r[0] - g[0], r[1] - g[1], float(np.hypot(r[0] - g[0], r[1] - g[1])),
                baseline_deg, baseline_m, dx_m, dy_m, dz_m, gpx))
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
            log_path = stamped_path(REPROJ_DIR, "reproj_log", "csv")
            log_file = open(log_path, "w")
            log_file.write(LOG_HEADER)
            print(f"corner log started: {log_path}")
        else:
            log_file.close()
            log_file = None
            print(f"corner log stopped: {log_path}")
    elif k== ord('s'):

        filename = stamped_path(IMAGE_DIR, "img", "jpg")
        success = cv2.imwrite(filename, img)

        if success:
            print(f"image saved: {filename}")
        else:
            print(f"could not save image: {filename}")

if log_file is not None:
    log_file.close()
    print(f"corner log closed: {log_path}")

# The device switch normally closes this one, but a Ctrl-C or a dropped socket never delivers the
# falling edge -- so the recording still ends as a valid file, just without the "stopped" line.
if pos_file is not None:
    pos_file.close()
    print(f"marker position log closed: {pos_path}")

comm_socket.close()
s.close()
cv2.destroyAllWindows()