# Debug frame streamer: sends the camera frame plus BOTH marker overlays to tools/tcp_receiver.py
# over TCP (port 7007, tunnelled with `adb reverse tcp:7007 tcp:7007`).
#
# Split out of open_cv_processor.gd because none of it is part of marker tracking: socket lifecycle,
# reconnect, drop accounting, the wire format and the reprojection latch are all pure diagnostics.
# Keeping them here means the app script no longer carries ~230 lines it never reads, and deleting
# the whole facility is deleting one node rather than untangling shared state.
#
# Lives as a CHILD of the OpenCVProcessor node: it needs project_marker_corners(), which is a method
# of the C++ class, so the parent IS the processor. Its own _process drives the socket, so the host
# does not tick it.
#
# The host's only contract is submit(), called once per finished detection.
class_name TcpDebugStream
extends Node

#stream image of quest to laptop via tcp
const TCP_HOST := "127.0.0.1"
const TCP_PORT := 7007			#view available ports with adb reverse --list

# Master switch, safe to flip at any time -- including from the remote inspector of a running
# deploy, which is the point: start tcp_receiver.py, turn this on, grab the frames you need, turn it
# off again. OFF by default, because a streamer nobody is listening to still costs a connect attempt
# every second for the whole session (and a log line with it), and on device that is the normal case.
# There is deliberately no setter: _poll_tcp runs every frame and reconciles the socket against this
# flag, so switching it on connects on the next frame and switching it off closes on the next frame,
# with no way for the two to disagree and nothing that depends on when the value was set. submit()
# also gates on a connected socket, so a closed one stops frames being encoded.
@export var enabled := false

# Records the WORLD poses of the detected markers -- position AND orientation, independent of the receiver's 'l' key (that
# one logs reprojection CORNERS into tools/reproj_logs/ and is driven from the laptop). Flip this on
# in the remote inspector of a running deploy, move around, flip it off: the receiver opens
# tools/marker_poses/marker_pos_<date>_<time>.csv on the rising edge and closes it on the falling
# one, so turning the switch OFF is what hands you the finished file.
#
# It rides inside the frame the streamer already sends rather than getting a channel of its own,
# for the same reason the marker blocks do: _tcp_out drains asynchronously, so a second writer would
# overtake the pixels still queued in front of it.
#
# What that costs, and it matters for reading the file: only STREAMED frames are recorded. The
# throttle (TCP_SEND_INTERVAL), a socket that is not connected, and every frame dropped while the
# receiver is behind all take their positions with them. The file is therefore a SAMPLE of the
# detections -- right for watching a static marker's estimate wander, wrong for counting detections
# or measuring detection rate.
# Needs `enabled` on: with the socket closed submit() returns long before any of this runs.
@export var record_global_marker_pos := false

# Throttle floor. submit() is already called only once per finished detection (~12/s on Quest), so
# this only throttles further -- it can no longer force a send.
const TCP_SEND_INTERVAL := 0.01

# The reprojection reference: marker world poses, the head pose they were solved at, and when they
# were latched.
#
# Held for REPROJ_BASELINE_MS rather than refreshed every frame, and that is the whole sensitivity
# of the test. The red-green gap goes as fx * head_turn * lens_error / range, so with fx=435 and a
# marker at 0.65m a 5mm lens error over the ~80ms between two detections (~5deg of head turn at a
# brisk 60deg/s) is 0.3px -- invisible. The same error over a second of head turning (~50deg) is
# ~3px, which is visible. Longer costs nothing when the calibration is RIGHT (a correct world pose
# is head-independent, so red stays on green however old the reference is); it only bounds how long
# the marker has to stay continuously in view.
#
# 5s rather than 1s so the SAME displacement can be covered slowly or quickly inside one hold --
# which is the test that separates the two error families, since a geometry error scales with how
# far the head moved while a timing error scales with how FAST it moved. At 1s a slow sweep could
# not finish before the reference re-latched, so both speeds could not be compared at equal
# baseline.
const REPROJ_BASELINE_MS := 5000.0
var _prev_poses: Dictionary = {}
var _prev_xform := Transform3D.IDENTITY
var _prev_ms := 0

var stream_peer: StreamPeerTCP
var _tcp_reconnect_timer := 0.0
var _last_tcp_status := -1
var _tcp_send_timer := 0.0

# Outbound frame buffer. StreamPeerTCP.put_data() BLOCKS until the last byte is gone, so calling it
# from the main thread hands our frame budget to the receiver: as soon as the Python script or the
# adb tunnel falls behind, TCP back-pressure stalls _process, the OpenXR frame submission stops with
# it and the runtime kills the app. So a frame is buffered here instead and drained with
# put_partial_data() from _poll_tcp, which never blocks. A frame arriving while the previous one is
# still draining is DROPPED (not queued) -- that caps the lag at one frame and makes the stream
# self-limit to whatever rate the link and the receiver can really take. Dropping is per whole frame
# on purpose: a half-sent frame cannot be replaced without desyncing the receiver's framing.
var _tcp_out := PackedByteArray()
var _tcp_dropped := 0

# The OpenCVProcessor this hangs under. Cached rather than looked up per frame, and used for exactly
# one thing now: project_marker_corners().
#
# Typed to the NATIVE class (the C++ one) rather than to the host's class_name, which is what makes
# the access checked at edit time: ArucoMarkerSource here would be a cyclic script reference, since
# the host holds a typed reference to THIS class, but OpenCVProcessor is native and sits in no cycle.
# The debug flag that used to be read off the host's script (and forced this to be an untyped Node)
# now comes from the C++ static instead -- see _debug below.
var _processor: OpenCVProcessor


func _ready() -> void:
	_processor = get_parent() as OpenCVProcessor
	if _processor == null:
		push_error("[opencv_aruco] [tcp_stream::_ready] parent is not an OpenCVProcessor: streaming disabled")


# Read from the C++ static rather than from the host's mirror of it. The host's exported
# debug_prints_enabled pushes every change into exactly this static (see its setter in
# open_cv_processor.gd), so this is the source and not a second copy -- and reading it needs no
# reference to the host's script at all, which is what keeps _processor typable above.
func _debug() -> bool:
	return OpenCVProcessor.get_debug_prints_enabled()


func _process(delta: float) -> void:
	_poll_tcp(delta)
	_tcp_send_timer += delta


# THE entry point. Call once per finished detection, with the frame that detection ran on, the
# corners it found, the WORLD poses it produced and the head pose those were baked with.
#
# The frame is passed in rather than read from a slot the host owns: the send has to happen while
# the image is still the frame those corners belong to, and an argument makes that structural
# instead of a comment about call order.
func submit(img: Image, corners: Dictionary, world_poses: Dictionary, head_xform: Transform3D) -> void:
	if img == null or _processor == null:
		return
	if _tcp_send_timer < TCP_SEND_INTERVAL:
		return
	_tcp_send_timer = 0.0
	if stream_peer == null:
		return
	# Polled again here, not just once per _process: submit() runs from the detection callback, so
	# without it a socket that died since the last frame would still read CONNECTED and take a frame.
	stream_peer.poll()
	if stream_peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
		return

	# Second overlay: the world poses latched up to REPROJ_BASELINE_MS ago, projected into THIS
	# frame. Deliberately cross-frame. Reprojecting a frame's own poses with its own head pose
	# cancels algebraically -- the C++ multiplied by head_pose * lens_pose and project_marker_corners
	# divides by it again, leaving M_1 exactly -- so it would draw a perfect square however wrong
	# lens_pose is. Only a head pose that has CHANGED since the latch makes lens_pose stop
	# cancelling, which is why the head has to move for this to mean anything and why the baselines
	# travel with the frame.
	var reproj := _processor.project_marker_corners(_prev_poses, head_xform)
	# How far the head has moved since the reference was latched -- measured against the reference
	# that produced the reprojection, so both have to be read BEFORE the re-latch below. Zero while
	# there is no reference yet, where the identity placeholder would otherwise report the head's
	# absolute pose as if it were a baseline.
	#
	# BOTH numbers travel, because they drive DIFFERENT errors and neither substitutes for the
	# other. Turning the head is what makes a lens_translation error show up. Strafing cannot: with
	# the head orientation fixed the relative motion A is a pure translation and
	# L^-1 A L = [I | R_L^T d], in which t_L has cancelled out entirely. What a strafe does reveal is
	# anything that puts the marker at the wrong DEPTH -- a wrong aruco_patch_size or focal length --
	# because parallax against a wrongly-placed point grows with viewpoint change, plus a
	# lens_rotation error through the surviving R_L.
	# The displacement travels as three SIGNED components in the LATCHED HEAD's own frame, not as a
	# scalar and not in world axes. World axes would only record how the user happened to be standing;
	# the latch frame separates the motions that mean different things:
	#   x = right, y = up   -> perpendicular to the view, so these are what generate PARALLAX and
	#                          therefore what exposes a wrong marker size or focal length
	#   z = backwards       -> Godot cameras look down -Z, so negative z is toward the marker. This
	#                          changes RANGE rather than viewpoint, and a scale error behaves
	#                          differently under it than under the other two.
	# The receiver still gets the L2 norm -- it just computes it, instead of us throwing the axes away.
	var baseline_deg := 0.0
	var d_local := Vector3.ZERO
	if not _prev_poses.is_empty():
		baseline_deg = rad_to_deg(head_xform.basis.get_rotation_quaternion().angle_to(
				_prev_xform.basis.get_rotation_quaternion()))
		# The current head POSITION expressed in the latched head's coordinate system. The latched
		# head sits at that system's origin, so this vector IS the displacement in latch-local axes.
		d_local = _prev_xform.affine_inverse() * head_xform.origin
	var now_ms := Time.get_ticks_msec()
	if _prev_poses.is_empty() or now_ms - _prev_ms >= REPROJ_BASELINE_MS:
		_prev_poses = world_poses.duplicate()
		_prev_xform = head_xform
		_prev_ms = now_ms
	# world_poses, NOT _prev_poses: the recording wants this frame's estimate, while the latch
	# above is the deliberately STALE reference the reprojection overlay needs.
	_send_frame(img, corners, reproj, baseline_deg, d_local, world_poses)


# Buffers one frame and pushes what fits right now. Never blocks; drops the frame outright while the
# previous one is still on its way out (see _tcp_out).
#
# Wire format, big-endian throughout, consumed by tools/tcp_receiver.py:
#   header:  total_size u32, width u32, height u32, image_format u32, payload_size u32   (20 bytes)
#   payload: payload_size raw Image bytes
#   markers: marker_count u32, then per marker id u32 + 8 f32               (36 bytes each)
#            = the 4 corners as x0,y0,..,x3,y3 in the payload's own pixel space.
#   reproj:  a SECOND block in exactly the same layout -- the latched world poses projected back
#            into this frame (see submit for what it tests).
#   trailer: baseline_deg f32, then dx f32, dy f32, dz f32 = how far the head has turned since those
#            poses were latched, and how far it has MOVED, as signed components in the latched
#            head's own frame (x right, y up, z backwards). Without them the pixel gap the receiver
#            prints is uninterpretable: the gap scales with the baseline and is zero at zero, so
#            "0.4px" means "calibrated" after 50deg of turning and means "you stood still" after
#            2deg. Rotation and translation are sent separately because they drive different errors
#            -- turning exposes lens_translation, strafing exposes marker size / focal length -- and
#            the translation keeps its axes because sideways and forwards do not do the same thing
#            either (see submit). The receiver derives the L2 norm from the three.
#   world:   recording u32 (0/1), count u32, then per marker id u32 + x,y,z f32 +
#            qx,qy,qz,qw f32 (32 bytes each) = the marker POSE in WORLD space, i.e. what
#            record_global_marker_pos records. Last in the frame.
#
# total_size counts ITSELF and everything after it, i.e. it is the whole frame, and it is the reason
# the rest of this format is safe to extend. It used to be absent, and the invariant that replaced it
# was "every block must be consumed for every frame, in order, or the stream desyncs forever" --
# defended by four comments in the receiver and one here, because a single `continue` past a block,
# or one field read with the wrong width, silently turned this frame's tail into the next frame's
# header and every frame after it was garbage. With a length up front the receiver takes total_size
# bytes off the socket and parses them in memory, so a parse bug costs the frame it is in and nothing
# else, and a sender/receiver mismatch is caught immediately (bytes left over) instead of drifting.
# All blocks are still always written, count 0 included, and the layout is still fixed -- the length
# makes a violation DETECTABLE, it does not make the format self-describing. Sender and receiver
# still have to be updated together; that is now an error message rather than a ruined session.
# Corner count per marker is fixed at 4 (guaranteed by the C++ side), hence no per-marker length.
#
# One StreamPeerBuffer for the whole frame, not put_u32 on the socket: _tcp_out drains
# asynchronously, so anything written to the socket by hand would overtake the pixels still queued in
# front of it. Being one buffer also makes a dropped frame drop its markers with it.
# (The by-hand big-endian encoding this used to do -- a _be_u32/_be_f32 pair allocating a
# PackedByteArray per scalar, plus a byte-reverse per float -- was reproducing what
# StreamPeerBuffer.big_endian already does natively.)
func _send_frame(img: Image, corners: Dictionary, reproj: Dictionary, baseline_deg: float,
		d_local: Vector3, world_poses: Dictionary) -> void:
	if not _tcp_out.is_empty():
		_tcp_dropped += 1
		if _debug() and _tcp_dropped % 100 == 0:
			print("[opencv_aruco] [tcp_stream::_send_frame] receiver behind: dropped=%d backlog=%d bytes" % [
					_tcp_dropped, _tcp_out.size()])
		return

	var bytes: PackedByteArray = img.get_data()

	var buf := StreamPeerBuffer.new()
	# THE thing that makes every put_* below big-endian. Set on the buffer, not on the socket: the
	# socket only ever sees put_partial_data() with bytes that are already encoded, so a big_endian
	# flag over there governs nothing (it used to be set on the StreamPeerTCP, where it was dead).
	buf.big_endian = true
	buf.put_u32(0)                       # total_size placeholder, patched in below
	buf.put_u32(img.get_width())
	buf.put_u32(img.get_height())
	buf.put_u32(img.get_format())
	buf.put_u32(bytes.size())
	buf.put_data(bytes)

	_put_marker_block(buf, corners)
	_put_marker_block(buf, reproj)
	buf.put_float(baseline_deg)
	buf.put_float(d_local.x)
	buf.put_float(d_local.y)
	buf.put_float(d_local.z)
	_put_world_block(buf, world_poses)

	# Patched in now that the size is known, rather than computed up front from the marker counts:
	# an arithmetic size beside the writer is a second description of the layout, and the two could
	# disagree -- which is exactly the desync the field exists to prevent. seek() moves the write
	# cursor only, so the megabyte already in the buffer is not copied or moved.
	var total := buf.get_size()
	buf.seek(0)
	buf.put_u32(total)

	_tcp_out = buf.data_array
	_flush_tcp()


# One marker block: count u32, then id u32 + 8 big-endian f32 per marker. Two of these go out per
# frame (detected corners first, then reprojected ones) and the receiver reads them with the same
# function, so the layout is factored out here rather than written twice.
# Writes into the caller's buffer instead of returning one: the frame is a single buffer by
# construction that way, with no intermediate PackedByteArray per block to concatenate.
func _put_marker_block(buf: StreamPeerBuffer, markers: Dictionary) -> void:
	buf.put_u32(markers.size())
	for id in markers:
		buf.put_u32(id)
		var pts: PackedVector2Array = markers[id]
		for p in pts:
			buf.put_float(p.x)
			buf.put_float(p.y)


# The world-position block: recording u32 (0/1), count u32, then id u32 + x,y,z f32 per marker.
#
# The flag TRAVELS rather than being inferred from count, and that is not redundancy. While
# recording, a frame in which the detector found nothing also has count 0, so a receiver reading
# only the count could not tell "recording, marker out of view" from "not recording" and would close
# the file on the first dropout. With the flag the file spans exactly the interval the switch was on,
# dropouts included -- which is what makes a gap in it readable as a gap rather than as the end.
#
# Position AND orientation. The orientation travels as a QUATERNION rather than as the euler angles
# one actually wants to read, and that is deliberate: euler needs a convention to be meaningful, it
# wraps at +-180deg, and it degenerates at gimbal lock -- three ways for a log to record a jump that
# never happened. A quaternion has none of those, and the receiver keeps it verbatim, so the choice
# of what to DISPLAY stays with the plot rather than being baked into the capture.
# Worth knowing while reading it: solvePnP's orientation on a single small planar marker scatters
# ~2.2deg median / 5.2deg p90 regardless of how good the calibration is (see the note at
# lens_rotation_raw in OpenCVProcessor.h). That is the noise floor these columns sit on -- a wobble
# of a couple of degrees is the marker being small and flat, not the calibration being wrong.
func _put_world_block(buf: StreamPeerBuffer, world_poses: Dictionary) -> void:
	if not record_global_marker_pos:
		# Eight bytes even when idle, because the receiver reads a FIXED structure per frame -- an
		# omitted block would leave the frame short of its own total_size (same contract as the two
		# marker blocks).
		buf.put_u32(0)
		buf.put_u32(0)
		return

	buf.put_u32(1)
	buf.put_u32(world_poses.size())
	for id in world_poses:
		buf.put_u32(id)
		var xform: Transform3D = world_poses[id]
		buf.put_float(xform.origin.x)
		buf.put_float(xform.origin.y)
		buf.put_float(xform.origin.z)
		# get_rotation_quaternion() rather than Quaternion(basis): it orthonormalises first, so a
		# basis carrying any scale still yields a unit quaternion instead of silently encoding the
		# scale into one that no longer represents a rotation.
		var rot := xform.basis.get_rotation_quaternion()
		buf.put_float(rot.x)
		buf.put_float(rot.y)
		buf.put_float(rot.z)
		buf.put_float(rot.w)


# Hands the socket as much of the buffered frame as it will take without waiting. Called once per
# _process from _poll_tcp and again right after buffering, so a frame that fits leaves the same frame.
func _flush_tcp() -> void:
	if _tcp_out.is_empty() or stream_peer == null:
		return

	var res: Array = stream_peer.put_partial_data(_tcp_out)
	var err: int = res[0]
	var sent: int = res[1]

	if err != OK:
		push_error("[opencv_aruco] [tcp_stream::_flush_tcp] put_partial_data failed: err=%d" % err)
		_tcp_out.clear()
		return

	if sent >= _tcp_out.size():
		_tcp_out.clear()
	elif sent > 0:
		_tcp_out = _tcp_out.slice(sent)


# No big_endian flag on the socket: every byte reaching it is already encoded (see _send_frame,
# which sets that flag on the StreamPeerBuffer that does the encoding). The flag used to be set here,
# where it governed nothing -- put_partial_data does not go through the endianness machinery -- and
# read as though the wire format depended on it.
func _connect_tcp() -> void:
	stream_peer = StreamPeerTCP.new()

	var err := stream_peer.connect_to_host(TCP_HOST, TCP_PORT)
	if _debug():
		print("[opencv_aruco] [tcp_stream::_connect_tcp] connect_to_host: err=%d" % err)


# Drop the socket and everything that belongs to it. The half-sent frame goes too: a later reconnect
# has to start at a header boundary or the receiver is desynced for good (same reason _poll_tcp
# clears it on an error). _last_tcp_status is reset so the next connection logs its transitions from
# scratch instead of silently matching the stale value.
func _close_tcp() -> void:
	stream_peer.disconnect_from_host()
	stream_peer = null
	_tcp_out.clear()
	_last_tcp_status = -1
	_tcp_reconnect_timer = 0.0
	if _debug():
		print("[opencv_aruco] [tcp_stream::_close_tcp] stream closed: dropped_frames=%d" % _tcp_dropped)


func _poll_tcp(delta: float) -> void:
	# The one place the socket is reconciled against `enabled`, in both directions.
	if not enabled:
		if stream_peer != null:
			_close_tcp()
		return

	if stream_peer == null:
		_connect_tcp()
		return

	stream_peer.poll()

	var status := stream_peer.get_status()

	if status != _last_tcp_status:
		if _debug():
			print("[opencv_aruco] [tcp_stream::_poll_tcp] status changed: from=%d to=%d" % [_last_tcp_status, status])
		_last_tcp_status = status

	if status == StreamPeerTCP.STATUS_CONNECTED:
		_tcp_reconnect_timer = 0.0
		_flush_tcp()
		return

	if status == StreamPeerTCP.STATUS_CONNECTING:
		return

	if status == StreamPeerTCP.STATUS_ERROR or status == StreamPeerTCP.STATUS_NONE:
		# The half-sent frame belongs to the dead socket; a reconnect starts at a header boundary.
		_tcp_out.clear()
		_tcp_reconnect_timer += delta
		if _tcp_reconnect_timer >= 1.0:
			_tcp_reconnect_timer = 0.0
			if _debug():
				print("[opencv_aruco] [tcp_stream::_poll_tcp] reconnecting")
			_connect_tcp()
