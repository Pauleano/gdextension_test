# Measurement apparatus for the marker pipeline: the hand-eye calibration capture, the marker drift
# readout, and the OpenXR timing validation. None of it is part of marker tracking, and nothing on
# the live path ever reads any of it back.
#
# Split out of open_cv_processor.gd for the reason TcpDebugStream was: these are things you switch
# on to MEASURE something, and they carried ~310 lines of state, file handling and statistics
# through a script whose actual job is the detection loop. Removing the whole facility is now
# deleting one node instead of untangling a dozen member variables from _process and
# _apply_detection_result.
#
# Lives as a CHILD of the OpenCVProcessor node: the hand-eye capture calls get_lens_pose(), a method
# of the C++ class, so the parent IS the processor -- exactly the arrangement TcpDebugStream needs
# for project_marker_corners().
#
# Three entry points, and they are deliberately three rather than one:
#   submit(head, markers)                        once per finished detection -- hand-eye + drift
#   sample(pdt, now, offset, ..., delta)         once per rendered frame     -- pdt sync + locate check
#   configure(locator, api, to_world, cam, org)  once, from _setup_xr_locator
# Everything sample() needs is state the HOST owns -- above all the clock offset, which the host
# resamples (_resample_clock_offsets) -- so it is handed over per frame rather than read back out of
# the parent. A second copy of that offset here could drift from the one the pose lookup uses, and
# the whole point of the pdt check is to police exactly that kind of divergence.
# The same handover rule now covers the host's XRCamera3D/XROrigin3D: they arrive as configure()
# arguments rather than being read off the parent, which is what lets _processor be typed to the
# NATIVE OpenCVProcessor (see its declaration) instead of to an untyped Node.
#
# The log lines keep their EVENT names ("marker drift", "openxr pdt sync", "openxr locate check",
# "openxr locate raw", "handeye samples"); only the function tag changes to [diag::...], the same way
# the streamer's is [tcp_stream::...]. So grepping logcat for what CLAUDE.md describes still works.
class_name DetectionDiagnostics
extends Node

# The OpenCVProcessor this hangs under, typed to the NATIVE class (the C++ one from the extension),
# not to the host's GDScript class_name. That distinction is the whole trick: ArucoMarkerSource here
# would be a cyclic script reference, because the host holds a typed reference to THIS class -- but
# OpenCVProcessor is native, sits in no cycle, and carries the only two methods this node calls
# anyway (get_lens_pose here, project_marker_corners in the streamer). So the access is checked at
# edit time instead of costing an unsafe-access warning per use.
# What that costs is that GDScript-side members of the host are no longer reachable through it -- the
# type genuinely does not have them. There were three, and all three are now handed over instead:
# debug_prints_enabled comes from the C++ static below, xr_camera/xr_origin arrive in configure().
# Still get_parent(): the child relationship is not incidental, it is what makes "this node
# instruments THAT processor" unwireable-wrong, and an @export node slot could be left empty.
var _processor: OpenCVProcessor


func _ready() -> void:
	_processor = get_parent() as OpenCVProcessor
	if _processor == null:
		push_error("[opencv_aruco] [diag::_ready] parent is not an OpenCVProcessor: diagnostics disabled")


# Single debug gate, shared with the host and the streamer. Read from the C++ static rather than
# from the host's mirror of it: the host's exported property pushes every change into exactly this
# static (see its setter in open_cv_processor.gd), so this is the source rather than a second copy,
# and reading it here needs no reference to the host script at all.
# This node has no debug switch of its own on purpose -- the hand-eye capture below has one because
# it WRITES something; a readout that only prints has nothing to gate beyond the prints themselves.
func _debug() -> bool:
	return OpenCVProcessor.get_debug_prints_enabled()


####################################################################################################
# --- Hand-eye calibration capture (measures T_view->cam directly, all six DOF) --------------------
#
# A measurement run, not a thing to leave on. The lens pose is decoded from Camera2's LENS_POSE_*,
# which is referenced to the GYROSCOPE, while the detection applies it to the OpenXR VIEW pose. The
# T_view->gyro factor between those frames is unmodelled -- it is not a number any API hands out
# (OpenXR has no IMU reference space, Camera2 never mentions the view), so no amount of editing the
# raw dump converges on it. This collects the raw material for solving it instead: for a STATIONARY
# marker, H_i * L * M_i must be the SAME world pose for every observation, so L = T_view->cam falls
# out of enough (H_i, M_i) pairs. That is the classic AX = XB hand-eye problem.
# M_i costs nothing to obtain: the C++ returns W = (cam_xform * lens_pose) * M_cam, so
# M_cam = (cam_xform * get_lens_pose()).affine_inverse() * W recovers the camera-space pose exactly.
# No second detection pass, no extra ~40ms on the frame, and the live path is untouched -- the
# capture reads the result that was going to be produced anyway.

# Safe to flip at any time, including from the remote inspector of a running deploy -- the file is
# opened lazily on the first KEPT sample, so with this off the capture costs one bool test per
# detection and never touches the filesystem. It sits on this node now rather than on the processor,
# so the switch you are looking for in the remote inspector is one level down from where it used to be.
@export var handeye_capture := false
# Samples are kept only when the head has MOVED since the last one. AX = XB is solved from the
# RELATIVE motion between observations, so a thousand samples taken while holding still carry no
# information whatsoever -- they only weight the fit toward whichever pose was held longest. The
# rotation threshold matters more than the translation one: pure translation leaves the rotational
# part of L unconstrained, which is exactly the part we are chasing.
const HANDEYE_MIN_ROT_DEG := 2.0
const HANDEYE_MIN_POS_M := 0.02
# Enough for a well-conditioned solve many times over; the cap only stops an unattended run from
# filling the headset's storage.
const HANDEYE_MAX_SAMPLES := 500
# One file per capture RUN, named for the time it was opened. FileAccess.WRITE truncates, so the
# fixed name this used to have meant a second run silently destroyed the first -- and two runs of
# the same rig are exactly what you compare when a solve comes back with centimetre residuals.
# Device local time, since it is read against your memory of the session rather than against a clock.
# The name is not predictable from the laptop any more, so it is printed on open and on close: that
# is the string to put in the adb pull (see the header of tools/handeye_solve.py).
const HANDEYE_PREFIX := "user://handeye_samples_"
var _handeye_path := ""
var _handeye_file: FileAccess
var _handeye_kept := 0                          # frames written, not lines (a frame can see several)
var _handeye_last_xform := Transform3D.IDENTITY
var _handeye_has_last := false


# Row-major 3x4 [R | t], so numpy reshapes it straight into a pose matrix. Godot's Basis stores
# its x/y/z as COLUMNS (basis.x == get_column(0) == the image of the local X axis), so the rows
# written here are built component-wise rather than by dumping basis.x/y/z in order -- doing that
# would silently transpose every rotation and the solve would converge on nonsense.
func _xform_to_array(t: Transform3D) -> Array:
	return [
		t.basis.x.x, t.basis.y.x, t.basis.z.x, t.origin.x,
		t.basis.x.y, t.basis.y.y, t.basis.z.y, t.origin.y,
		t.basis.x.z, t.basis.y.z, t.basis.z.z, t.origin.z,
	]


# One JSONL line per (frame, marker): the head pose in WORLD space and that marker's pose in RAW
# CAMERA space. Main thread only, reached from submit() -- the detection worker never touches any
# of this.
# Two things invalidate a capture run, and neither is detectable from the data afterwards:
# the marker must not MOVE (the whole constraint is that its world pose is constant), and the
# reference frame must not shift under you -- no XRServer.center_on_hmd(), no moving XROrigin3D,
# since the head poses are logged in world space.
func _handeye_record(head: Transform3D, markers: Dictionary) -> void:
	if markers.is_empty() or _handeye_kept >= HANDEYE_MAX_SAMPLES:
		return
	if _handeye_has_last:
		var moved_deg := rad_to_deg(head.basis.get_rotation_quaternion().angle_to(
				_handeye_last_xform.basis.get_rotation_quaternion()))
		var moved_m := head.origin.distance_to(_handeye_last_xform.origin)
		if moved_deg < HANDEYE_MIN_ROT_DEG and moved_m < HANDEYE_MIN_POS_M:
			return
	# Opened lazily, on the first kept sample: with the flag off this function is never even reached
	# and nothing here touches the filesystem.
	if _handeye_file == null:
		var t := Time.get_datetime_dict_from_system()
		_handeye_path = HANDEYE_PREFIX + ("%04d%02d%02d_%02d%02d%02d.jsonl" % [
				t.year, t.month, t.day, t.hour, t.minute, t.second])
		_handeye_file = FileAccess.open(_handeye_path, FileAccess.WRITE)
		if _handeye_file == null:
			push_error("[opencv_aruco] [diag::_handeye_record] cannot open %s: err=%d (capture disabled)" % [
					_handeye_path, FileAccess.get_open_error()])
			handeye_capture = false
			return
		print("[opencv_aruco] [diag::_handeye_record] handeye capture started: path=%s" % ProjectSettings.globalize_path(_handeye_path))
	# Exactly the pose the C++ pre-multiplied onto every solvePnP result, so inverting it undoes
	# that one step and nothing else -- what comes back is the raw camera-space marker pose,
	# independent of whatever lens pose is currently configured. That independence is the point: the
	# samples stay valid even if the lens pose is edited between capture and solve. get_lens_pose()
	# reads the same decoded transform the detection used, so the two cannot drift apart -- and it is
	# a method of the PARENT, which is why this node has to be its child. Inferred rather than
	# annotated: _processor is typed to the native OpenCVProcessor, so the return type is known here.
	var lens_pose := _processor.get_lens_pose()
	var inv := (head * lens_pose).affine_inverse()
	for id in markers:
		_handeye_file.store_line(JSON.stringify({
			"id": id,
			"head": _xform_to_array(head),
			"marker_cam": _xform_to_array(inv * markers[id]),
		}))
	# Flushed per sample because a Quest session ends with `adb shell am force-stop`, which gives
	# the app no chance to close anything -- an unflushed buffer would take the run with it.
	_handeye_file.flush()
	_handeye_kept += 1
	_handeye_last_xform = head
	_handeye_has_last = true
	if _handeye_kept % 25 == 0:
		print("[opencv_aruco] [diag::_handeye_record] handeye samples=%d/%d (vary head pitch AND yaw; rotation is what constrains the solve)" % [
				_handeye_kept, HANDEYE_MAX_SAMPLES])


####################################################################################################
# --- Marker drift readout -------------------------------------------------------------------------

# id -> [world position, head rotation] as of the FIRST detection of that id. Reference point for the
# drift readout below, which exists to answer one question the rest of the pipeline cannot: when the
# mesh appears to slide against a stationary marker, is the world ESTIMATE moving, or only its
# apparent alignment? Passthrough is a reprojected image, not optical see-through, so a pose that is
# rock steady in world space can still look like it slides as the head turns -- and no amount of
# lens-pose work touches that. Comparing the estimate against itself takes the passthrough out of the
# loop entirely.
# Never cleared, so the reference survives a dropout and the drift stays comparable across the whole
# session; bounded by the marker count, same as the host's _marker_poses.
var _drift_ref: Dictionary = {}
# id -> the world positions seen so far while the reference is still forming; dropped once it locks.
var _drift_seed: Dictionary = {}
# id -> head rotation at the FIRST detection. head_deg is measured from this, and it is deliberately
# the first sample rather than a median: it is an angle datum, not a noisy measurement, and the head
# is held still while the reference forms anyway.
var _drift_rot: Dictionary = {}
# How many detections the position reference is averaged over before it locks. Seeding from ONE
# detection makes every later drift_mm hostage to that sample's noise -- measured on device the
# per-frame position jitter is ~2.5mm peak-to-peak, so a lone outlier at seeding time offsets the
# whole session by a constant and reads exactly like a real standing error. ~1.5s of detections.
const DRIFT_REF_SAMPLES := 20


# Component-wise median of the seed window for `id`, recomputed until the window fills and then
# frozen. Median rather than mean so a single bad detection during seeding cannot drag the datum.
func _drift_reference_pos(id: int, pos: Vector3) -> Vector3:
	if _drift_ref.has(id):
		return _drift_ref[id]
	var seed: Array = _drift_seed.get(id, [])
	seed.append(pos)
	_drift_seed[id] = seed
	var xs: Array = []
	var ys: Array = []
	var zs: Array = []
	for p in seed:
		xs.append(p.x)
		ys.append(p.y)
		zs.append(p.z)
	xs.sort()
	ys.sort()
	zs.sort()
	var mid: int = seed.size() / 2
	var med := Vector3(xs[mid], ys[mid], zs[mid])
	if seed.size() >= DRIFT_REF_SAMPLES:
		_drift_ref[id] = med           # locked; the seed window is dead weight from here on
		_drift_seed.erase(id)
	return med


# How far each marker's WORLD estimate has wandered from where it was first seen, against how far the
# head has turned since. The marker does not move, so every millimetre of drift_mm is error -- and the
# correlation is the diagnosis:
#   drift_mm grows with head_deg -> the error is conjugated by the head transform, i.e. it lives in
#     the lens pose or the head pose, and calibration is the fix.
#   drift_mm stays flat while head_deg swings -> the pose chain is solid and what you SEE moving is
#     the render against passthrough's reprojection, which calibration cannot touch.
# dist_m is printed because it separates the two halves of a lens-pose error: a ROTATIONAL one scales
# with it, a TRANSLATIONAL one does not. The hand-eye capture spanned only 0.51-0.60m, so that split
# is the least-constrained thing in the current calibration -- watch drift_mm as you walk toward and
# away from a marker, not just as you turn your head.
# Every detection is printed, not one frame in DEBUG_FLOW_EVERY: the value of this is reading a whole
# sweep as a sequence, which a sampled trace cannot give you.
func _drift_report(head: Transform3D, markers: Dictionary) -> void:
	# Constant across the frame -- the loop below compares each marker's own reference against this
	# one head rotation.
	var head_rot := head.basis.get_rotation_quaternion()
	for id in markers:
		var pos: Vector3 = markers[id].origin
		if not _drift_rot.has(id):
			_drift_rot[id] = head_rot
		var ref_pos := _drift_reference_pos(id, pos)
		var ref_rot: Quaternion = _drift_rot[id]
		# Lines still marked SEEDING have a reference that is still moving under them, so their
		# drift_mm is not comparable with the rest -- drop them before reading the sweep.
		print("[opencv_aruco] [diag::_drift_report] marker drift: id=%d drift_mm=%.1f head_deg=%.1f dist_m=%.2f world_pos=%v%s" % [
				id, (pos - ref_pos).length() * 1000.0,
				rad_to_deg(head_rot.angle_to(ref_rot)),
				head.origin.distance_to(pos),
				pos,
				"" if _drift_ref.has(id) else " SEEDING"])


####################################################################################################

# THE per-detection entry point. Call once per finished detection, on the main thread, with the WORLD
# poses that detection produced and the head pose they were baked with -- i.e. the pose at CAPTURE
# time, not the live one, which is what makes a calibration solved from these samples valid for the
# live path.
#
# Both consumers take exactly these two arguments, which is why they share one call: the hand-eye
# capture writes them to disk, the drift readout compares them against their own history. Each gates
# itself, so an idle node costs two bool tests per detection.
func submit(head: Transform3D, markers: Dictionary) -> void:
	if handeye_capture:
		_handeye_record(head, markers)
	if _debug():
		_drift_report(head, markers)


####################################################################################################
# --- OpenXR timing validation ---------------------------------------------------------------------

# Everything below is set once by configure(), from the host's _setup_xr_locator. Null/empty means
# there is no OpenXR at all (desktop run without a headset), and sample() then does nothing.
var _locator: OpenXRHeadLocator
var _xr_api: OpenXRAPIExtension       # for get_next_frame_time(); pdt is handed in per frame
var _to_world: Callable               # the host's _play_space_to_world
var _xr_camera: XRCamera3D
var _xr_origin: XROrigin3D

# How far into the past the (B) probe asks, in ms. Chosen to sit at the far end of the real
# camera latency (sensor timestamp lags ~30-60ms here), so a runtime that only keeps a short
# tracking history fails the probe rather than passing it and then failing on real frames.
const LOCATE_PAST_PROBE_MS := 60.0
var _locate_check_timer := 0.0
# Accumulators for the pdt/_process lockstep check. Sampled EVERY frame, printed once per second: a
# per-frame print through logcat at the render rate would slow down the very loop being measured.
# _pdt_prev survives the window reset, so the first delta of a window is measured against the last
# frame of the previous one.
var _pdt_prev := 0                     # pdt of the previous _process; 0 = nothing sampled yet
var _pdt_prev_now_usec := 0            # Time.get_ticks_usec() of that same _process
var _pdt_deltas := 0                   # frame-to-frame deltas accumulated in this window
var _pdt_dupes := 0                    # deltas of exactly 0 -> pdt did NOT advance (the failure mode)
var _pdt_advance_ns := 0               # summed pdt deltas over the window
var _pdt_wall_advance_usec := 0        # summed wall-clock deltas over the same frames
var _pdt_delta_min_ns := 0
var _pdt_delta_max_ns := 0
var _lead_sum_ms := 0.0
var _lead_min_ms := 0.0
var _lead_max_ms := 0.0
var _lead_print_timer := 0.0


# One-time handshake, from the host's _setup_xr_locator -- i.e. from the single place that decides
# whether this deploy has OpenXR at all. Never called on a headset-less desktop run, which is exactly
# how sample() knows to stay quiet.
#
# to_world is the host's _play_space_to_world passed as a Callable rather than reimplemented here or
# poked at through the parent: the live locate path uses the same function, and check (A) below is
# only a test of the TIME argument as long as both sides convert identically. A second copy would
# turn a conversion bug into a check that passes.
#
# xr_camera / xr_origin travel the same way, and for the mundane version of the same reason: they are
# @onready members of the host's SCRIPT, so reading them back through _processor would force that
# reference to be untyped (see its declaration). They are already resolved by the time
# _setup_xr_locator runs, so handing them over costs two arguments and buys the typing.
func configure(locator: OpenXRHeadLocator, xr_api: OpenXRAPIExtension, to_world: Callable,
		xr_camera: XRCamera3D, xr_origin: XROrigin3D) -> void:
	_locator = locator
	_xr_api = xr_api
	_to_world = to_world
	_xr_camera = xr_camera
	_xr_origin = xr_origin


# THE per-frame entry point, called once from the host's _process.
#
# pdt / now_usec / xr_clock_offset_ns are read and owned by the host (the offset is resampled there),
# and pdt_stamping says whether the pose history is actually being pdt-stamped this run -- the pdt
# statistics are meaningless otherwise. They are handed over rather than mirrored here because a
# second copy of the clock offset could diverge from the one the pose lookup uses, which is the very
# failure this check exists to catch.
func sample(pdt: int, now_usec: int, xr_clock_offset_ns: int, pdt_stamping: bool, delta: float) -> void:
	if not _debug():
		return
	if pdt_stamping:
		_sample_pdt(pdt, now_usec, xr_clock_offset_ns, delta)
	# Its own timer and its own condition, NOT folded into the pdt block above: this is what decides
	# whether use_xr_locate_space may be turned on, so it has to keep working on a deploy where the
	# pdt stamping is off.
	if _locator != null:
		_locate_check_timer += delta
		if _locate_check_timer >= 1.0:
			_locate_check_timer = 0.0
			_check_xr_locate(pdt)


# Health check for the host's pose stamping: it is only valid if the pdt read on a frame belongs to
# the pose read beside it, i.e. if pdt advances exactly once per _process. When it does, the summed
# pdt deltas over any window equal the summed wall-clock deltas, because both telescope to
# (last - first):
#   ratio ~ 1.00, dupes = 0   -> lockstep, stamping is sound
#   dupes > 0, ratio < 1      -> pdt repeats across _process calls; it is stale on those frames and
#                                the stamps carry a VARIABLE error
#   ratio > 1                 -> pdt skips ahead of the main loop
# Per-line ratio scatter of a few percent is expected and benign: it is exactly the lead's drift
# across that window divided by the window length, not a desync. pdt_delta_ms should land on integer
# multiples of the display period (13.89ms at 72Hz), the minimum being one.
# NOTE this cannot detect a CONSTANT one-frame stagger -- that keeps ratio at 1.00 with no dupes,
# hiding entirely in the absolute offset. It rules out the variable failure only; the constant
# residual is what the host's POSE_LOOKUP_TRIM_MS is for.
func _sample_pdt(pdt: int, now_usec: int, xr_clock_offset_ns: int, delta: float) -> void:
	if pdt != 0:
		# The very first frame only seeds the reference; every later frame contributes one delta.
		if _pdt_prev != 0:
			var d_pdt_ns := pdt - _pdt_prev
			var lead_ms := float(pdt - (now_usec * 1000 + xr_clock_offset_ns)) / 1.0e6
			if _pdt_deltas == 0:      # first delta of a fresh window seeds the extremes
				_pdt_delta_min_ns = d_pdt_ns
				_pdt_delta_max_ns = d_pdt_ns
				_lead_min_ms = lead_ms
				_lead_max_ms = lead_ms
			else:
				_pdt_delta_min_ns = mini(_pdt_delta_min_ns, d_pdt_ns)
				_pdt_delta_max_ns = maxi(_pdt_delta_max_ns, d_pdt_ns)
				_lead_min_ms = minf(_lead_min_ms, lead_ms)
				_lead_max_ms = maxf(_lead_max_ms, lead_ms)
			if d_pdt_ns == 0:
				_pdt_dupes += 1
			_pdt_deltas += 1
			_pdt_advance_ns += d_pdt_ns
			_pdt_wall_advance_usec += now_usec - _pdt_prev_now_usec
			_lead_sum_ms += lead_ms
		_pdt_prev = pdt
		_pdt_prev_now_usec = now_usec

	_lead_print_timer += delta
	if _lead_print_timer >= 1.0 and _pdt_deltas > 0 and _pdt_wall_advance_usec > 0:
		_lead_print_timer = 0.0
		print("[opencv_aruco] [diag::_sample_pdt] openxr pdt sync: frames=%d dupes=%d ratio=%.4f pdt_delta_ms=%.2f..%.2f lead_ms=%.2f (%.2f..%.2f)" % [
				_pdt_deltas, _pdt_dupes,
				float(_pdt_advance_ns) / float(_pdt_wall_advance_usec * 1000),
				_pdt_delta_min_ns / 1.0e6, _pdt_delta_max_ns / 1.0e6,
				_lead_sum_ms / _pdt_deltas, _lead_min_ms, _lead_max_ms])
		_pdt_deltas = 0
		_pdt_dupes = 0
		_pdt_advance_ns = 0
		_pdt_wall_advance_usec = 0
		_lead_sum_ms = 0.0


# One probe: head pose at xr_time, expressed in world space and compared against `live`. Returns
# the two scalars the checks below print, or just valid=false if the runtime would not answer.
# Factored out because the whole point of the check is comparing the SAME measurement at three
# different times -- doing that inline three times invites the three copies drifting apart.
func _locate_delta(xr_time: int, live: Transform3D) -> Dictionary:
	var loc: Dictionary = _locator.locate_head(xr_time)
	if not loc.get("valid", false):
		return {"valid": false, "result": loc["result"], "flags": loc["flags"]}
	var raw: Transform3D = loc["transform"]
	var pose: Transform3D = _to_world.call(raw)
	return {
		"valid": true,
		"tracked": loc["tracked"],
		"flags": loc["flags"],
		"pos_mm": pose.origin.distance_to(live.origin) * 1000.0,
		"rot_deg": rad_to_deg(pose.basis.get_rotation_quaternion().angle_to(live.basis.get_rotation_quaternion())),
		# The play-space pose exactly as the runtime gave it, before any conversion -- see the
		# raw diagnostic line below.
		"ps_origin": raw.origin,
	}


# Once-a-second device check for the xrLocateSpace path (debug prints only). MOVE YOUR HEAD while
# reading it -- every number below is a difference between two head poses, so standing still makes
# all of them zero and proves nothing.
#
#   (A) Which instant does XRCamera3D's pose actually describe, and does our conversion reproduce
#       it? Probed at BOTH times OpenXR offers -- get_predicted_display_time() and
#       get_next_frame_time() (= pdt + one display period) -- because Godot could plausibly use
#       either: it samples the head pose in the main loop, and whether that happens before or
#       after the frame's xrWaitFrame decides which one is current. Whichever comes back at ~0 is
#       the one XRCamera3D belongs to; the other sits a display period of head motion away.
#       MEASURED ON DEVICE: pdt wins, by a wide margin -- it reproduces XRCamera3D to 0.02-0.3mm
#       (often bit-for-bit) while one display period is worth ~7mm at normal head speed. So the
#       pdt-stamped history carries NO one-frame stagger, and that suspicion can come off
#       POSE_LOOKUP_TRIM_MS's list; what is left for it is start-vs-middle of exposure.
#       If NEITHER probe is near zero, the VIEW space or _play_space_to_world is wrong and
#       nothing else here means anything until that is fixed.
#   (B) Does this runtime answer for times in the PAST? That is the entire premise of the switch:
#       a camera frame is 30-60ms old by the time it arrives. valid=true at
#       pdt - LOCATE_PAST_PROBE_MS says yes, and pos_mm/rot_deg then show how far the head
#       travelled over that interval -- which is exactly the error the history path has to cover
#       with predicted poses. MEASURED: valid, tracked, and 1-31mm of movement over the 60ms.
#       valid=false would mean the runtime will not serve history: turn use_xr_locate_space off.
func _check_xr_locate(pdt: int) -> void:
	if pdt == 0:
		print("[opencv_aruco] [diag::_check_xr_locate] openxr locate check: no predicted display time yet")
		return
	if not _locator.is_ready():
		print("[opencv_aruco] [diag::_check_xr_locate] openxr locate check: locator not ready (no session or play space yet)")
		return

	# Read once, so all three probes are compared against the same reference pose.
	var live := _xr_camera.global_transform
	var next_time: int = _xr_api.get_next_frame_time()

	var at_pdt := _locate_delta(pdt, live)
	var at_next := _locate_delta(next_time, live)
	print("[opencv_aruco] [diag::_check_xr_locate] openxr locate check (A) vs XRCamera3D: pdt=%s next_frame=%s gap_ms=%.2f (MOVE YOUR HEAD; the one at ~0 is the time XRCamera3D's pose belongs to)" % [
			_fmt_locate(at_pdt), _fmt_locate(at_next), (next_time - pdt) / 1.0e6])

	var in_past := _locate_delta(pdt - int(LOCATE_PAST_PROBE_MS * 1.0e6), live)
	print("[opencv_aruco] [diag::_check_xr_locate] openxr locate check (B) at pdt-%.0fms: %s (valid => historical queries work, and the offsets are the motion over that interval)" % [
			LOCATE_PAST_PROBE_MS, _fmt_locate(in_past)])

	# Raw values, because the two lines above only ever show DIFFERENCES -- and a difference of
	# "head height" looks the same whether the located pose is wrong or simply absent (which is
	# exactly how the transform_from_pose() bug first presented).
	#   ps == cam_local            -> the pose and the whole conversion are right
	#   ps=(0,0,0) with valid=true -> the runtime did not write our XrSpaceLocation
	# ps_move is the premise in one number: the play-space origin at pdt against the one 60ms
	# earlier. Exactly 0.00 while your head is moving would mean the runtime ignores `time` and
	# the switch is worthless; measured here it runs 1-31mm, i.e. historical queries work.
	if at_pdt["valid"] and in_past["valid"]:
		print("[opencv_aruco] [diag::_check_xr_locate] openxr locate raw: ps=%v cam_local=%v cam_world=%v origin=%v ref=%v scale=%.2f ps_move_mm=%.2f" % [
				at_pdt["ps_origin"],
				_xr_camera.transform.origin, live.origin,
				_xr_origin.global_transform.origin, XRServer.get_reference_frame().origin,
				XRServer.world_scale,
				at_pdt["ps_origin"].distance_to(in_past["ps_origin"]) * 1000.0])


func _fmt_locate(d: Dictionary) -> String:
	if not d["valid"]:
		return "INVALID(result=%d flags=%d)" % [d["result"], d["flags"]]
	return "[pos_mm=%.2f rot_deg=%.3f tracked=%s]" % [d["pos_mm"], d["rot_deg"], d["tracked"]]


####################################################################################################

# Children get NOTIFICATION_EXIT_TREE BEFORE their parent, so this runs before the host drains its
# in-flight detection task. Harmless: submit() only ever reaches _handeye_record from the main
# thread via _apply_detection_result, which cannot run during teardown, and every sample is flushed
# as it is written -- so a force-stop (which skips this entirely) loses nothing either.
func _exit_tree() -> void:
	if _handeye_file != null:
		_handeye_file.close()
		_handeye_file = null
		print("[opencv_aruco] [diag::_exit_tree] handeye capture closed: samples=%d path=%s" % [
				_handeye_kept, ProjectSettings.globalize_path(_handeye_path)])
