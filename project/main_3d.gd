# Named so consumers can type their reference to it (@export var marker_source: ArucoMarkerSource)
# and get the public marker API below checked at edit time instead of as a runtime "Invalid call".
class_name ArucoMarkerSource
extends Node3D

var processor: OpenCVProcessor
# id -> patch Node3D, the ONE registry of live patches. Nodes are created on a marker's first
# detection and freed once it has been missing for PATCH_LOST_TIMEOUT_MS; the scene file holds
# none of them. Because an id is only ever added through _get_or_create_patch (which checks this
# dictionary first) and only ever removed together with its node, one id can never own two nodes
# nor swap onto another id's node. Main thread only -- detection tasks never touch it.
var marker_nodes: Dictionary = {}
# id -> Time.get_ticks_usec() of the last detection result that contained it. Drives the patch
# deletion grace period AND, since it is no longer erased when a patch is freed, the public
# marker_age_ms() below -- a consumer asking "how stale is my last good pose?" must still get an
# answer after the mesh is gone.
var _marker_last_seen: Dictionary = {}
# id -> resolved size in meters for the C++ side. Rebuilt by _sync_marker_sizes, which runs in
# _ready and then once per detection task start -- and ONLY there, because that is the one point
# in the frame at which no task is in flight, so the unlocked read in _detect_frame can never
# overlap a write.
var _marker_size_table: Dictionary = {}

# Fallback physical side length, in meters, for every marker id WITHOUT a table entry below.
# Used as the solvePnP marker size (sets the pose's metric scale) AND as the rendered cuboid's
# side length (see _ready), so the two can never disagree.
@export_range(0.01, 0.3, 0.001, "or_greater", "suffix:m") var aruco_patch_size := 0.1

# Ground-truth lookup table: index = marker id (0-9), value = physical side length in meters.
# 0 = unset -> that id falls back to aruco_patch_size (as do all ids >= 10 of ARUCO_MIP_36h12), so
# an untouched table behaves exactly like the single-size setup before it existed.
@export_range(0.01, 0.3, 0.001, "or_greater", "suffix:m") var aruco_patch_sizes: Array[float] = [
	0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
]

# Rendered thickness (z) of a patch cuboid in meters -- x/y come from that id's marker size, so the
# box overlays the printed marker at true size. Was the authored z-scale of the scene meshes.
const PATCH_THICKNESS := 0.01
# How long a marker may be absent from detection results before its node is freed. Detection is
# noisy (motion blur, glancing angles drop a marker for a frame or two), so deleting on the first
# miss would make the mesh flicker; half a second bridges the dropouts and still feels immediate.
const PATCH_LOST_TIMEOUT_MS := 500.0
# ONE unit-cube mesh shared by every patch, as the scene's SubResource was before. Size is per
# patch and lives in the MeshInstance3D's scale, never in the mesh.
var _patch_mesh := BoxMesh.new()

# Single switch for ALL debug output, this script's and the C++ extension's -- _ready pushes it into
# OpenCVProcessor before instantiating it. Off by default: with no tcp_receiver.py listening the TCP
# reconnect logs alone would print once per second for the whole session, and the per-frame timing
# below fires at the camera rate. Errors are never gated and print regardless. DEBUG_FLOW is a
# SUB-switch of this one: flow tracing needs both (see _tracing).
# Format for every debug line: "[opencv_aruco] [main_3d::function] event: key=value" -- the fixed
# prefix is what makes them findable in logcat (adb logcat | grep opencv_aruco).
@export var debug_prints_enabled := false:
	set(value):
		debug_prints_enabled = value
		# Mirror every change into the C++ static right away. The extension gates its ACV_DBG lines
		# on a flag that only GDScript can set, so pushing it once in _ready meant a toggle from the
		# remote inspector of a RUNNING deploy silenced this script's prints while the extension kept
		# logging. Assigning the property inside its own setter does not recurse in GDScript, and
		# scene instantiation applies exported values before _ready -> the flag is already correct
		# when the (printing) C++ constructor runs.
		OpenCVProcessor.set_debug_prints_enabled(value)
		# Same deal for the second extension class; both statics are pushed from this one place so
		# a toggle in the remote inspector reaches every C++ log line, not just some of them.
		OpenXRHeadLocator.set_debug_prints_enabled(value)

# --- Camera calibration (left Quest passthrough camera "50", native 640x480 frame) ---
# Exported so they can be tuned in the inspector instead of hunting through code. Read-only
# after _ready: detection tasks read them without a lock (same pattern as _marker_size_table),
# so treat inspector edits as pre-run configuration, not live tuning. Careful with that: the
# inspector's default float step is 0.001 and neither export below carries a range hint, so
# EDITING a distortion coefficient there rounds it to 3 decimals -- which zeroes both tangential
# terms outright. Change these in the file, not in the spinboxes.
# Intrinsics (fx, fy, cx, cy) in pixels for the NATIVE 640x480 frame. _detect_frame scales all
# four with the downscale factor at use time -- never bake that factor into these values.
# Calibrated with tools/cameraCalibration.py from TCP-streamed frames captured through the
# GodotAndroidCamera (CameraX) path, i.e. the pipeline that actually runs. That provenance is the
# point: the previous set came from the CameraServer readback path and agreed with this one to
# 1.5px and 0.2% on the focal lengths, which is what ruled out CameraX picking a different sensor
# crop -- a different field of view would have moved fx by percent, not by 0.08%. A calibration is
# only valid for the capture path that produced its images, so RE-SHOOT these after any change to
# the resolution, the output format, or the camera backend.
@export var camera_intrinsics := Vector4(435.01136927, 435.03491996, 321.81974795, 240.07772099)
# OpenCV distCoeffs (k1, k2, p1, p2, k3) for the Quest passthrough lens; an EMPTY array means
# "no distortion". These used to be hardcoded in the C++ side.
@export var camera_distortion: PackedFloat64Array = [-0.00993192, 0.11168738, 0.00062258, 0.00113467, -0.23033717]
# Detection resolution knob: 1.0 = native frame, 0.5 = half width AND half height, i.e. a quarter
# of the pixels -> markedly cheaper detection, at the price of small or distant markers dropping
# below the resolution the detector needs. _detect_frame hands it to the C++ side AND scales the
# intrinsics above by the same factor; the two must always move together, which is why the factor
# belongs here and never baked into camera_intrinsics.
@export_range(0.1, 1.0, 0.05) var image_downscale_factor := 1.0
# Passthrough camera expressed in the OpenXR VIEW frame -- MEASURED, not read off the device.
# Stored in the Quest's raw ACAMERA_LENS_POSE_* convention only because _ready decodes that
# convention: the raw quaternion is ~168.8deg about X = the Android sensor->camera-optical 180deg
# X-flip PLUS the camera's real ~11deg pitch, and since the C++ marker pose already contains that
# same 180deg flip (its negate-Y/Z change of basis), _ready multiplies by Quaternion(1,0,0,0)
# (=180deg about X) to cancel it and keep ONLY the physical mounting tilt (-> _lens_pose).
# Translation is in the sensor frame (X right, Y up, Z toward viewer), which matches Godot camera
# axes -> no sign flips.
# These two are ONE calibration and must be replaced as a pair -- a rotation from one solve beside
# a translation from another describes no camera that exists.
#
# DO NOT "fix" these by pasting in what init_quest_intrinsics() logs at startup, however
# authoritative that dump looks. It is gyro-referenced: LENS_POSE_REFERENCE == GYROSCOPE means the
# metadata describes the camera relative to the IMU, while cam_to_world applies it to the VIEW
# pose. Those frames differ by the IMU's mounting rotation, which NEITHER api exposes -- OpenXR
# has no IMU reference space, Camera2 never mentions the view -- so the difference cannot be
# looked up and has to be measured. Running the raw dump is what produced a ~11mm offset that no
# pose-path change could touch (present with the head completely still, on both the history and
# the xrLocateSpace path) while still swinging the patch around as the head turned, the error
# being conjugated by the head transform.
#
# Solved by tools/handeye_solve.py from 439 captured samples (see handeye_capture above):
# H_i * L * M_i collapses to one world pose to 2.4mm median / 6.1mm p90, against 3.7mm/7.3mm for
# the previous hand-carried dump. Re-measure with that tool rather than editing by eye.
# Note what the same solve says about ORIENTATION: the marker's world rotation still scatters
# 7.7deg median / 14.5deg p90, and that is NOT calibration error -- it is solvePnP's inherent
# noise on a single small planar marker at ~0.7m, and it stays no matter how good this pose is.
# A patch that sits in the right place but visibly wobbles in orientation is that, and the fix is
# temporal averaging or a multi-marker board, not another lens-pose hunt.
@export var lens_rotation_raw := Quaternion(-0.99501617258316, -0.00309562040391, 0.00819784967978, 0.09932788477010)
@export var lens_translation := Vector3(-0.03126009993300, -0.01277714405469, -0.06316742365001)
# Derived ONCE from the exported raw values in _ready (before any detection task can exist).
var _lens_pose := Transform3D.IDENTITY

# --- Hand-eye calibration capture (measures T_view->cam directly, all six DOF) ---
# A measurement run, not a thing to leave on. Everything above sources _lens_pose from Camera2's
# LENS_POSE_*, which is referenced to the GYROSCOPE, while cam_to_world applies it to the OpenXR
# VIEW pose. The T_view->gyro factor between those frames is unmodelled -- it is not a number any
# API hands out (OpenXR has no IMU reference space, Camera2 never mentions the view), so no
# amount of editing the raw dump converges on it. This collects the raw material for solving it
# instead: for a STATIONARY marker, H_i * L * M_i must be the SAME world pose for every
# observation, so L = T_view->cam falls out of enough (H_i, M_i) pairs. That is the classic
# AX = XB hand-eye problem.
# M_i costs nothing to obtain: the C++ returns W = (cam_xform * _lens_pose) * M_cam, so
# M_cam = (cam_xform * _lens_pose).affine_inverse() * W recovers the camera-space pose exactly.
# No second detection pass, no extra ~40ms on the frame, and the live path is untouched -- the
# capture reads the result that was going to be produced anyway.
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
const HANDEYE_PATH := "user://handeye_samples.jsonl"
var _handeye_file: FileAccess
var _handeye_kept := 0                          # frames written, not lines (a frame can see several)
var _handeye_last_xform := Transform3D.IDENTITY
var _handeye_has_last := false


# Godot has a native CameraServer (Camera2) backend on Android since 4.5, so
# CameraServerExtension is only needed on desktop (Windows). Keep this var UNTYPED and
# instantiate via ClassDB so the script still parses on Android, where the
# CameraServerExtension class isn't registered.
var camera_extension
var cam_texture: CameraTexture

# Android/Quest: frames come from the GodotAndroidCamera plugin (CameraX ImageAnalysis) as raw
# CPU bytes -- no CameraTexture, no GPU->CPU readback -- plus the SENSOR timestamp (start of
# exposure) of every frame, so the head pose can be looked up at the true capture time.
var android_camera: AndroidCamera
var _android_cam_started := false
var _cam_clock_offset_ns := 0          # camera timestamp clock ns - Godot ticks ns
var _cam_ts_realtime := false          # feed stamps on boottime ("realtime") vs CLOCK_MONOTONIC ("unknown")
# Second clock bridge, for XrTime (CLOCK_MONOTONIC ns on the Quest) -> Godot ticks. Kept separate
# from _cam_clock_offset_ns because that one follows the FEED's clock, which is boottime for a
# "realtime" feed and would be wrong for pdt. On the Quest passthrough path ("unknown" -> also
# monotonic) the two hold the same value, and that is what makes an error in the offset cancel
# out of the pose lookup entirely -- see _resample_clock_offsets and _process.
var _xr_clock_offset_ns := 0
var _xr_stamp_poses := false           # stamp history entries with pdt instead of "now" (Quest only)
var _xr_lead_usec := 0                 # last measured pdt - now, carried over if pdt is briefly 0
var _cam_frame_count := 0
var _cam_fallback_count := 0           # frames whose timestamp failed the plausibility guard
var _preview_texture: ImageTexture     # debug preview fed from plugin frames (desktop uses cam_texture)
var _xr_api: OpenXRAPIExtension        # access to xrWaitFrame's predicted display time (XrTime)
var _lead_print_timer := 0.0

# --- xrLocateSpace head pose (the measured alternative to the predicted history) ---
# The pose history below holds poses OpenXR PREDICTED for a display time; even stamped with the
# instant they describe, they are forecasts, and the predictor's error rides into every marker
# pose. _head_locator asks the runtime for the head pose at an arbitrary XrTime instead -- for a
# time in the PAST that is the fused, after-the-fact estimate, i.e. measured rather than guessed.
# Null when OpenXR is not initialised (desktop without a headset); every use is guarded.
var _head_locator: OpenXRHeadLocator
# true = xrLocateSpace at the frame's capture time, false = the pdt-stamped pose history and
# _head_pose_at. ON since the once-a-second "openxr locate check" lines passed on the Quest:
# (A) locating at pdt reproduces XRCamera3D to 0.02-0.3mm, so the VIEW space and the world
# conversion are right, and (B) pdt-60ms comes back valid and tracked with 1-31mm of head motion
# against the live pose, so this runtime really does serve measured history rather than clamping
# to the newest pose. Flip it back to compare the two on device -- a frame the locator cannot
# answer already falls back to the history by itself, so neither setting can strand the app.
@export var use_xr_locate_space := true
var _locate_check_timer := 0.0
var _locate_fallback_count := 0        # frames on which the locator had no valid pose
# How far into the past the (B) probe asks, in ms. Chosen to sit at the far end of the real
# camera latency (sensor timestamp lags ~30-60ms here), so a runtime that only keeps a short
# tracking history fails the probe rather than passing it and then failing on real frames.
const LOCATE_PAST_PROBE_MS := 60.0
# Accumulators for the pdt/_process lockstep check (see the block in _process). Sampled EVERY
# frame, printed once per second: a per-frame print through logcat at the render rate would slow
# down the very loop being measured. _pdt_prev survives the window reset, so the first delta of a
# window is measured against the last frame of the previous one.
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
@onready var cam_preview: TextureRect = $CameraLayer/CameraPreview
@onready var xr_origin: XROrigin3D = $XROrigin3D
@onready var xr_camera: XRCamera3D = $XROrigin3D/XRCamera3D

# --- Detection worker (Teil B) ---
# On the Quest the OpenCV detection costs ~80ms, which run synchronously would cap the whole
# app at ~10fps. We run ONLY the detection (detectMarkers + solvePnP) off the main thread, as
# one-shot WorkerThreadPool tasks -- Godot owns the threads, so there is no Thread/Mutex/
# Semaphore lifecycle to manage here. At most ONE task is in flight at a time; get_image()
# and all scene-tree writes stay on the main thread (neither is thread-safe).
var _detect_task_id := -1              # WorkerThreadPool task id; -1 = no task in flight
# input slot (main thread ONLY, so no lock): frames arriving while a task runs wait here;
# only the newest frame is kept (frame drop). The head pose sampled at the frame's capture
# time travels with the frame (see _on_android_camera_frame).
var _pending_image: Image
var _pending_capture_usec := 0
var _pending_cam_xform := Transform3D.IDENTITY
var _has_pending := false
# output slot, written by the task; the main thread reads it only AFTER
# wait_for_task_completion(), which is the synchronization point (no lock needed)
var _result_markers: Dictionary = {}
var _result_capture_usec := 0
var _result_cam_xform := Transform3D.IDENTITY   # head pose the markers were baked with (trace only)

# --- Capture-latency compensation ---
# A frame's pixels are OLDER than "now" when they reach us (sensor -> ISP -> delivery takes
# 1-2 camera frames), so pairing them with the LIVE head pose bakes an error proportional to
# head speed. We keep a short timestamped head-pose history and sample the head pose from the
# frame's CAPTURE time instead (see _head_pose_at); the pose travels with the frame, and the
# C++ side bakes the markers straight to world space with it.
# On Android the plugin supplies the true sensor timestamp of every frame; CAMERA_LATENCY_MS is
# only the fallback guess for the desktop CameraServer path, where no timestamp exists.
const CAMERA_LATENCY_MS := 50.0

# Residual trim of the pose lookup on the Android path, in ms; positive = use an OLDER head pose.
# Expected to stay at 0: the two error sources this used to absorb are both measured now. The
# sensor timestamp removes the variable pipeline delay, and stamping every history entry with its
# own predicted display time (see _process) removes the OpenXR prediction lead -- which is not a
# constant anyway, it moves with render load and sawtooths ~40ms peak-to-peak. What is left is
# only what neither can see: a possible constant one-frame stagger between pdt and the pose read
# beside it, and whether the sensor timestamp marks the start or the middle of the exposure.
# Tune on device like CAMERA_LATENCY_MS: patch drags WITH the head during motion -> raise; patch
# lags BEHIND the real marker -> lower.
@export_range(-20,20,0.1,"or_greater","or_less","suffix:ms") var POSE_LOOKUP_TRIM_MS := 0.0

var _xr_cam_pose_history: Array = []   # [t_usec, head Transform3D] pairs, newest last; main thread only

# --- Debug: step-by-step flow tracing (learning aid, costs performance) ---
# Every DEBUG_FLOW_EVERY-th camera frame is "traced": each station of the pipeline prints one
# line tagged with that frame's id, so ONE frame can be followed end to end:
#   (1) arrival -> (2) timestamp -> (3) bias -> (4) head pose at capture -> (5) handoff
#   -> (6) worker -> (7) detection (baked to world space in C++) -> (8) applying result
#   -> (9) what the timing correction was worth -> (10) markers assigned
# Tracing by id (instead of printing everything) keeps the lines of one frame together even
# though stations run on two threads and ~200ms apart. Set DEBUG_FLOW = false to silence.
const DEBUG_FLOW := true
const DEBUG_FLOW_EVERY := 30       # 1 = every frame (very chatty); 30 = ~1 traced frame per second
var _flow_frame_counter := 0       # frame ids for the desktop path (Android uses _cam_frame_count)
var _pending_frame_id := 0         # id travelling with the frame in the input slot
var _result_frame_id := 0          # id travelling with the result in the output slot


#stream image of quest to laptop via tcp
const TCP_HOST := "127.0.0.1"
const TCP_PORT := 7007			#view available ports with adb reverse --list

var stream_peer: StreamPeerTCP
var _tcp_reconnect_timer := 0.0
var _last_tcp_status := -1
var _tcp_send_timer := 0.0
const TCP_SEND_INTERVAL := 0.01

# Outbound frame buffer for the debug streamer. StreamPeerTCP.put_data() BLOCKS until the last
# byte is gone, so calling it from the main thread hands our frame budget to the receiver: as soon
# as the Python script or the adb tunnel falls behind, TCP back-pressure stalls _process, the
# OpenXR frame submission stops with it and the runtime kills the app. So a frame is buffered here
# instead and drained with put_partial_data() from _poll_tcp, which never blocks. A frame arriving
# while the previous one is still draining is DROPPED (not queued) -- that caps the lag at one
# frame and makes the stream self-limit to whatever rate the link and the receiver can really take.
# Dropping is per whole frame on purpose: a half-sent frame cannot be replaced without desyncing
# the receiver's header/payload framing.
var _tcp_out := PackedByteArray()
var _tcp_dropped := 0

#######################################################################################################

# --- Public marker API -------------------------------------------------------------------------
# Consumers (avatar rigs, debug gizmos) address markers by ID, never by patch node. The patch nodes
# are created and freed at runtime, so a NodePath to one is null at scene load and dangling after a
# dropout; an id is stable -- it is the key the detector itself uses. The patches are a debug
# RENDERING of this data, not the data.
# Main thread only, same as everything else that touches marker_nodes.

## Ids contained in the detection result just applied. For one-shot reactions; polling the getters
## below from _process is equally fine and is what the rigs do (this node is their parent, so its
## _process has already run when theirs does).
signal markers_updated(ids: Array)

# id -> last known WORLD pose. Deliberately NOT pruned alongside the patch nodes: a consumer holding
# its last good pose through a dropout needs the pose to outlive the mesh. ARUCO_MIP_36h12 bounds
# this at 250 entries, so it cannot grow without limit.
var _marker_poses: Dictionary = {}


## Last known world pose. IDENTITY if never detected -- pair with has_marker() if that would be
## indistinguishable from a real pose for you.
func get_marker_pose(id: int) -> Transform3D:
	return _marker_poses.get(id, Transform3D.IDENTITY)

## True once this marker has been detected at least once. It may be stale by now.
func has_marker(id: int) -> bool:
	return _marker_poses.has(id)

## Milliseconds since this marker was last detected; INF if never. INF compares correctly against
## any max_age below, so a never-seen id is simply never fresh.
func marker_age_ms(id: int) -> float:
	if not _marker_last_seen.has(id):
		return INF
	return (Time.get_ticks_usec() - _marker_last_seen[id]) / 1000.0

## True only if EVERY id is fresh -- a pose averaged over several markers is only as good as its
## weakest one. Defaults to the patch nodes' own grace period, so "the debug box is on screen" and
## "the consumer is tracking" stay the same statement.
func markers_fresh(ids: Array, max_age_ms := PATCH_LOST_TIMEOUT_MS) -> bool:
	for id in ids:
		if marker_age_ms(id) > max_age_ms:
			return false
	return true

## True once every id has been seen at least once, stale or not. Use this to decide whether a
## consumer may be shown at all; markers_fresh() decides whether to move it.
func markers_ever_seen(ids: Array) -> bool:
	for id in ids:
		if not _marker_poses.has(id):
			return false
	return true

## Centroid + mean rotation of the given markers. Unknown ids are skipped; IDENTITY if none are
## known.
##
## Sum-then-normalise is the cheap quaternion mean, and for exactly two markers it is identical to
## slerp(q0, q1, 0.5). slerp cannot generalise it: it takes only two, and chaining it is not
## associative, so the result would depend on marker order. Each quaternion is sign-aligned against
## the first KNOWN one, because q and -q are the same rotation and would otherwise cancel instead
## of average.
func get_average_marker_pose(ids: Array) -> Transform3D:
	var centre := Vector3.ZERO
	var acc := Quaternion(0, 0, 0, 0)
	var ref := Quaternion.IDENTITY
	var n := 0
	for id in ids:
		if not _marker_poses.has(id):
			continue
		var x: Transform3D = _marker_poses[id]
		var q := x.basis.get_rotation_quaternion()
		# Counting KNOWN ids rather than using the loop index is what makes skipping safe: the
		# reference is the first quaternion actually accumulated, not the first id asked for.
		if n == 0:
			ref = q
		elif ref.dot(q) < 0.0:
			q = -q
		centre += x.origin
		acc = Quaternion(acc.x + q.x, acc.y + q.y, acc.z + q.z, acc.w + q.w)
		n += 1
	if n == 0:
		return Transform3D.IDENTITY
	return Transform3D(Basis(acc.normalized()), centre / float(n))

## The live debug patch node for an id, or null while the marker is undetected. Read poses through
## get_marker_pose() -- this exists only for code that wants to touch the rendered box itself
## (hiding it, recolouring it), and must be re-fetched every use since the node is freed on loss.
func get_marker_node(id: int) -> Node3D:
	return marker_nodes.get(id)

# Single source of truth for a marker id's physical size: table entry if present and set,
# aruco_patch_size otherwise. The bounds check doubles as the guard for arrays the inspector
# resized to fewer/more than 10 elements.
func _marker_size_for(id: int) -> float:
	if id >= 0 and id < aruco_patch_sizes.size():
		var s: float = aruco_patch_sizes[id]
		if s > 0.0:
			return s
	return aruco_patch_size


# Re-resolve aruco_patch_sizes/aruco_patch_size into the id -> size table the C++ side gets, and put
# the same numbers on the live patch meshes. Both consumers of a marker's size are refreshed here,
# so they can never drift apart.
# CALLER CONTRACT: main thread, and only while _detect_task_id == -1. _detect_frame reads
# _marker_size_table from a worker thread without a lock, so this is the one point in the frame at
# which rewriting it is safe -- which is why the only call site is _start_detection_task, covering
# both the fresh-frame and the pending-frame path with one call. Cheap enough to run per task: ten
# dictionary writes plus one scale compare per live patch.
func _sync_marker_sizes() -> void:
	for id in aruco_patch_sizes.size():
		_marker_size_table[id] = _marker_size_for(id)
	# Rows the inspector shrank the array past must go, else a deleted entry would keep feeding
	# solvePnP its old size instead of falling back to aruco_patch_size. keys() is a copy, so
	# erasing inside the loop is safe.
	for id in _marker_size_table.keys():
		if id >= aruco_patch_sizes.size():
			_marker_size_table.erase(id)
	# Existing patches got their scale once, at creation time (_get_or_create_patch). Without this
	# a size change would only reach the mesh after the marker had been lost for
	# PATCH_LOST_TIMEOUT_MS and the node was rebuilt -- the box would keep its old edge length while
	# the pose already used the new one. Compared approximately because scale components are 32-bit:
	# an exact != against the double from _marker_size_for would fire every single time.
	for id in marker_nodes:
		var size := _marker_size_for(id)
		# The one child added in _get_or_create_patch; the patch node itself must stay scale-free,
		# it carries the baked pose.
		var mesh_instance: MeshInstance3D = marker_nodes[id].get_child(0)
		if not is_equal_approx(mesh_instance.scale.x, size):
			mesh_instance.scale = Vector3(size, size, PATCH_THICKNESS)
			if debug_prints_enabled:
				print("[opencv_aruco] [main_3d::_sync_marker_sizes] patch resized: id=%d size=%.3f" % [id, size])


func _ready() -> void:
	# The property setter already pushed this into the extension at scene-instantiation time; repeat
	# it here so the flag is also correct when the scene does NOT override the default (the setter
	# never fires then) and a previous run left the static true. Still BEFORE new(): the C++
	# constructor prints (OpenCV build info + Quest intrinsics dump).
	OpenCVProcessor.set_debug_prints_enabled(debug_prints_enabled)
	processor = OpenCVProcessor.new()

	# Lens pose from the exported raw Camera2 values (see their declarations for the why of the
	# extra 180deg X-flip). Built once here, read by detection tasks without a lock afterwards.
	_lens_pose = Transform3D(Basis((lens_rotation_raw * Quaternion(1, 0, 0, 0)).inverse()), lens_translation)

	# No patch nodes to find: they are created on demand as markers turn up (see
	# _get_or_create_patch) and freed again when they stop being detected.

	# Resolve the size table for the C++ side (entries are pre-resolved through _marker_size_for, so
	# the C++ default only fires for ids >= the table length). Re-run before every detection task
	# from _start_detection_task, so inspector edits reach solvePnP and the patch meshes while the
	# scene is running instead of being frozen at whatever _ready saw.
	_sync_marker_sizes()

	# No worker setup needed: detection runs as one-shot WorkerThreadPool tasks, dispatched on
	# demand once frames arrive (see _dispatch_detection) -- long after _ready has finished.

	# BEFORE the camera branch below, which can start delivering frames synchronously: those
	# frames want the locator. Safe to do here -- xr_startup.gd sits on the XROrigin3D CHILD, and
	# Godot runs a child's _ready before its parent's, so OpenXR is already initialised.
	_setup_xr_locator()

	if OS.get_name() == "Android" and Engine.has_singleton("GodotAndroidCamera"):
		# Quest: the GodotAndroidCamera plugin (CameraX) pushes frames to _on_android_camera_frame
		# as raw CPU bytes with a sensor timestamp -- no CameraServer, no GPU->CPU readback.
		_setup_android_camera()
	else:
		if OS.get_name() == "Android":
			# Plugin singleton missing (APK exported without it): fall back to the native
			# CameraServer; request camera access so it surfaces feeds once granted.
			OS.request_permission("android.permission.CAMERA")
			OS.request_permission("horizonos.permission.HEADSET_CAMERA")
		elif ClassDB.class_exists("CameraServerExtension"):
			# Desktop (Windows): custom backend that registers the webcam as a feed.
			camera_extension = ClassDB.instantiate("CameraServerExtension")  # keep reference alive

		# Since Godot 4.5, monitoring_feeds must be true before feeds are enumerated.
		CameraServer.monitoring_feeds = true
		CameraServer.camera_feeds_updated.connect(_on_camera_feeds_updated)
		_on_camera_feeds_updated()                          # in case a feed is already present


	_connect_tcp()

# Build the OpenXR head locator, or leave it null when there is no OpenXR at all (desktop run
# without a headset) -- every caller checks, and _xr_api is tied to the same condition so nothing
# below ever pokes a dead OpenXRAPI singleton once per frame.
# The VIEW space the C++ side creates is a CHILD of the XR session: the runtime destroys it with
# the session, so it has to be given up on session_stopping. The node's destructor is too late --
# by scene teardown the session is usually already gone (release() detects that and skips the
# xrDestroySpace, but the signal is the path that actually cleans up properly).
func _setup_xr_locator() -> void:
	var xr_interface: XRInterface = XRServer.find_interface("OpenXR")
	if xr_interface == null or not xr_interface.is_initialized():
		if debug_prints_enabled:
			print("[opencv_aruco] [main_3d::_setup_xr_locator] no OpenXR: head locator disabled, pose history stays the only path")
		return
	_xr_api = OpenXRAPIExtension.new()
	_head_locator = OpenXRHeadLocator.new()
	add_child(_head_locator)
	# connect() by NAME, not xr_interface.session_stopping.connect(): the var is statically typed
	# XRInterface, which has no such signal -- only the OpenXRInterface behind it does.
	if xr_interface.has_signal("session_stopping"):
		xr_interface.connect("session_stopping", Callable(_head_locator, "release"))
	if debug_prints_enabled:
		print("[opencv_aruco] [main_3d::_setup_xr_locator] head locator created: use_xr_locate_space=%s (validation prints once a second while debug is on)" % use_xr_locate_space)

func _on_camera_feeds_updated() -> void:
	if cam_texture != null:
		return                                          # already initialised
	var feed_count := CameraServer.get_feed_count()
	if feed_count == 0:
		return

	# log every available feed so we can see which index is the (passthrough) camera on Quest
	if debug_prints_enabled:
		for i in range(feed_count):
			var f := CameraServer.get_feed(i)
			print("[opencv_aruco] [main_3d::_on_camera_feeds_updated] feed: index=%d id=%d name=%s" % [i, f.get_id(), f.get_name()])

	# Quest exposes 3 feeds: "1 | FRONT" plus the passthrough pair "50 | BACK" / "51 | BACK".
	# The world-facing ("BACK") cameras are the passthrough ones we want; feed 0 (FRONT) is the
	# wrong camera. Desktop has a single feed, so it falls through to 0.
	var feed: CameraFeed = null
	for i in range(feed_count):
		var f := CameraServer.get_feed(i)
		if "BACK" in f.get_name():
			feed = f
			break
	if feed == null:
		feed = CameraServer.get_feed(0)

	# Format MUST be chosen before activating, else "format index -1" and no frames.
	var formats := feed.get_formats()
	if debug_prints_enabled:
		for j in range(formats.size()):
			print("[opencv_aruco] [main_3d::_on_camera_feeds_updated] format: index=%d value=%s" % [j, formats[j]])
	if formats.size() > 2:
		feed.set_format(2, {})        # feed format:10 1280x1280 YUV_420_888 (for now, choice can be altered) (good ArUco res, light on CPU)
	elif formats.size() > 0:
		feed.set_format(0, {})

	feed.set_active(true)                               # start delivering frames
	cam_texture = CameraTexture.new()
	cam_texture.camera_feed_id = feed.get_id()
	cam_texture.which_feed = CameraServer.FEED_RGBA_IMAGE
	cam_preview.texture = cam_texture
	if debug_prints_enabled:
		print("[opencv_aruco] [main_3d::_on_camera_feeds_updated] feed activated: id=%d name=%s feed_count=%d" % [feed.get_id(), feed.get_name(), feed_count])

# --- GodotAndroidCamera plugin path (Quest) ---

func _setup_android_camera() -> void:
	android_camera = AndroidCamera.new()
	add_child(android_camera)
	android_camera.camera_frame.connect(_on_android_camera_frame)
	# Start once BOTH permissions (android CAMERA + horizonos HEADSET_CAMERA) are granted; the
	# result signal fires once per permission, so re-check on every grant.
	get_tree().on_request_permissions_result.connect(_on_permission_result)
	if android_camera.request_camera_permissions():
		_start_android_camera()

func _on_permission_result(_permission: String, granted: bool) -> void:
	if granted and not _android_cam_started and android_camera.request_camera_permissions():
		_start_android_camera()

func _start_android_camera() -> void:
	_android_cam_started = true
	# Pick the LEFT passthrough camera ("50"): the hardcoded intrinsics + lens pose in
	# _detect_frame belong to it. Fall back to any world-facing feed on non-Quest devices.
	var cam_id := ""
	var cameras: Dictionary = android_camera.get_available_cameras()
	for id in cameras:
		var info: Dictionary = cameras[id]
		if info.get("source", "") == "passthrough" and info.get("position", "") == "left":
			cam_id = id
			break
	if cam_id == "":
		for id in cameras:
			if cameras[id].get("facing", "") == "back":
				cam_id = id
				break
	# Map camera timestamps onto Time.get_ticks_usec(), the clock the head-pose history is
	# stamped with. WHICH camera clock to calibrate against depends on the feed:
	# "realtime" = boottime (elapsedRealtimeNanos); "unknown" (Quest passthrough) = typically
	# CLOCK_MONOTONIC, the very clock Godot's ticks run on. Using the boottime offset for a
	# monotonic feed is off by the headset's accumulated doze time since boot -- a silent,
	# run-dependent bias that made patches lag behind head motion.
	_cam_ts_realtime = str(cameras.get(cam_id, {}).get("timestamp_source", "unknown")) == "realtime"
	_resample_clock_offsets()
	# From here on the head-pose history is stamped with xrWaitFrame's predicted display time
	# rather than "now" (see _process). _xr_api is built in _setup_xr_locator, which runs first
	# and is the single point that decides whether this deploy has OpenXR at all; a frame where
	# the runtime reports no pdt falls back gracefully.
	_xr_stamp_poses = _xr_api != null
	if debug_prints_enabled:
		print("[opencv_aruco] [main_3d::_start_android_camera] starting camera: id=%s clock=%s feeds=%s" % [
				cam_id, "realtime" if _cam_ts_realtime else "monotonic", cameras])
		# Every frame's exposure time is converted with: cap_usec = (timestamp_ns - offset_ns) / 1000
		print("[opencv_aruco] [main_3d::_start_android_camera] flow setup: clock bridge calibrated, camera clock is ahead of Time.get_ticks_usec() by offset_s=%.3f offset_ns=%d" % [
				_cam_clock_offset_ns / 1.0e9, _cam_clock_offset_ns])
	# 640x480 matches the hardcoded intrinsics; LUMA is the camera's native Y plane, which is
	# exactly the 1-channel grayscale the C++ detector consumes -- no conversion anywhere.
	android_camera.start_camera(640, 480, false, cam_id, 0, 0, AndroidCamera.OutputFormat.LUMA)

# Both clock bridges in ONE place so they can never drift apart. That matters: the pose lookup
# compares a camera timestamp mapped with _cam_clock_offset_ns against a history entry mapped with
# _xr_clock_offset_ns, so an error COMMON to both cancels out, while a divergence between them
# becomes a silent bias. On the Quest passthrough feed ("unknown" -> monotonic) they are literally
# the same number; only a "realtime" feed splits them, and then each is individually correct.
func _resample_clock_offsets() -> void:
	_xr_clock_offset_ns = android_camera.get_monotonic_clock_offset_nanos()
	_cam_clock_offset_ns = android_camera.get_clock_offset_nanos() if _cam_ts_realtime \
			else _xr_clock_offset_ns

# Runs on the main thread (plugin signals are marshalled onto the engine loop). data is the
# tight-packed Y plane; timestamp_ns is the sensor timestamp (start of exposure) of THIS frame.
func _on_android_camera_frame(timestamp_ns: int, data: PackedByteArray, width: int, height: int) -> void:
	# Sensor timestamp -> Godot clock. If the result is implausible (unexpected clock base, or
	# the offset went stale across a headset doze), resample the offset once and fall back to
	# the fixed-latency guess if it still disagrees.
	var now_usec := Time.get_ticks_usec()
	var cap_usec := (timestamp_ns - _cam_clock_offset_ns) / 1000
	var used_fallback := false
	if cap_usec > now_usec or now_usec - cap_usec > 500_000:
		_resample_clock_offsets()
		cap_usec = (timestamp_ns - _cam_clock_offset_ns) / 1000
		if cap_usec > now_usec or now_usec - cap_usec > 500_000:
			_cam_fallback_count += 1
			used_fallback = true
			cap_usec = now_usec - int(CAMERA_LATENCY_MS * 1000.0)
	var lag_ms := (now_usec - cap_usec) / 1000.0

	_cam_frame_count += 1
	var frame_id := _cam_frame_count
	var traced := _tracing(frame_id)
	if traced:
		# (1) what the camera handed us, (2) where that lands on Godot's clock.
		print("[opencv_aruco] [main_3d::_on_android_camera_frame] flow #%d (1) frame arrives: width=%d height=%d bytes=%d timestamp_ns=%d (grayscale Y-plane, 1 byte/pixel)" % [
				frame_id, width, height, data.size(), timestamp_ns])
		print("[opencv_aruco] [main_3d::_on_android_camera_frame] flow #%d (2) exposure time on godot clock: cap_usec=%d age_ms=%.1f source=%s" % [
				frame_id, cap_usec, lag_ms, "FALLBACK_GUESS_timestamp_rejected" if used_fallback else "sensor_timestamp"])

	# The lookup target is the exposure time itself: the history entries carry the time their pose
	# actually describes, so the two timelines already line up and only the residual trim is left.
	# cap_usec itself is NOT shifted -- it travels on as the frame's true age (see frame_age_ms).
	var lookup_usec := cap_usec - int(POSE_LOOKUP_TRIM_MS * 1000.0)
	if traced:
		print("[opencv_aruco] [main_3d::_on_android_camera_frame] flow #%d (3) pose lookup target: target_usec=%d trim_ms=%.1f (history is pdt-stamped, so no prediction lead to undo)" % [
				frame_id, lookup_usec, POSE_LOOKUP_TRIM_MS])

	# Head pose at the frame's capture time, sampled at ARRIVAL. Either way the lookup is by TIME,
	# so sampling here or after the detection gives the same pose -- but sampled here it can travel
	# with the frame, and the C++ side applies it together with the lens pose (markers come back
	# in world space).
	var cam_xform := Transform3D.IDENTITY
	var pose_source := "history"
	if use_xr_locate_space and _head_locator != null:
		# Sensor timestamp -> XrTime. Both clock offsets are resampled together (see
		# _resample_clock_offsets), so on the Quest passthrough feed -- monotonic, the same clock
		# XrTime runs on -- the two cancel and timestamp_ns goes in untouched; only a "realtime"
		# feed actually needs the detour through Godot's clock. The trim applies here too, for the
		# same residual it covers on the history path.
		var xr_time_ns := timestamp_ns - _cam_clock_offset_ns + _xr_clock_offset_ns \
				- int(POSE_LOOKUP_TRIM_MS * 1.0e6)
		var loc: Dictionary = _head_locator.locate_head(xr_time_ns)
		if loc.get("valid", false):
			cam_xform = _play_space_to_world(loc["transform"])
			# valid but not tracked = the runtime extrapolated through a tracking loss. Still the
			# best answer available, so it is used -- just labelled, so a bad stretch is visible
			# in the trace instead of silently looking like a good one.
			pose_source = "xrLocateSpace" if loc.get("tracked", false) else "xrLocateSpace_untracked"
		else:
			_locate_fallback_count += 1
	if pose_source == "history":
		cam_xform = _head_pose_at(lookup_usec)
	if traced:
		print("[opencv_aruco] [main_3d::_on_android_camera_frame] flow #%d (4) head pose at capture: pos=%v source=%s locate_fallbacks=%d (travels with the frame)" % [
				frame_id, cam_xform.origin, pose_source, _locate_fallback_count])

	if debug_prints_enabled and (_cam_frame_count == 1 or _cam_frame_count % 300 == 0):
		print("[opencv_aruco] [main_3d::_on_android_camera_frame] frame: count=%d width=%d height=%d lag_ms=%.1f clock=%s fallbacks=%d" % [
				_cam_frame_count, width, height, lag_ms,
				"realtime" if _cam_ts_realtime else "monotonic", _cam_fallback_count])

	var img := Image.create_from_data(width, height, false, Image.FORMAT_L8, data)

	# debug outputs (grayscale): preview overlay + TCP frame streamer
	if cam_preview.visible:
		if _preview_texture == null:
			_preview_texture = ImageTexture.create_from_image(img)
			cam_preview.texture = _preview_texture
		else:
			_preview_texture.update(img)
	if stream_peer != null and stream_peer.get_status() == StreamPeerTCP.STATUS_CONNECTED:
		_send_frame_tcp(img)

	_dispatch_detection(img, cap_usec, cam_xform, frame_id)

####################################################################################################

# True if this frame's journey should be traced. The SAME id gives the SAME answer at every
# station, so all prints belonging to one frame appear together (see DEBUG_FLOW).
func _tracing(frame_id: int) -> bool:
	return debug_prints_enabled and DEBUG_FLOW and frame_id % DEBUG_FLOW_EVERY == 0
func _process(_delta: float) -> void:
	_poll_tcp(_delta)

	# Head-pose history: one [t_usec, pose] sample per rendered frame, so a finished detection can
	# be baked with the pose the head actually had at the frame's capture time (see _head_pose_at).
	#
	# Each sample is stamped with the time its pose DESCRIBES, not the time it was read. OpenXR
	# hands out poses PREDICTED for xrWaitFrame's display time, so stamping them with "now" was a
	# per-sample lie that a single constant could only ever cancel on average. Measured on the
	# Quest, that lead is 40-80ms and not constant in either direction: it shrinks as the render
	# rate rises (~79ms at 34fps down to ~65ms at 56fps) and sawtooths ~40ms peak-to-peak within
	# that, because pdt steps in whole 13.89ms display quanta while the wall clock runs on. Giving
	# every entry its own pdt cancels all of it, hitches included.
	# The camera's sensor timestamp is mapped onto Godot ticks with the same monotonic offset
	# (see _resample_clock_offsets), so an error in that offset cancels here instead of biasing.
	var now_usec := Time.get_ticks_usec()
	# Read once per frame and shared with the locate check further down, which needs the same pdt
	# to compare against the same head pose. _xr_api is null exactly when there is no OpenXR, so
	# this never touches a dead singleton.
	var pdt := _xr_api.get_predicted_display_time() if _xr_api != null else 0
	var pose_usec := now_usec + _xr_lead_usec    # desktop: lead stays 0 -> stamped with "now"
	if _xr_stamp_poses and pdt != 0:
		pose_usec = (pdt - _xr_clock_offset_ns) / 1000
		_xr_lead_usec = pose_usec - now_usec
	_xr_cam_pose_history.append([pose_usec, xr_camera.global_transform])
	# Pruned against the NEWEST stamp rather than now_usec: the entries sit in the future once
	# they are pdt-stamped, and measuring the window from "now" would silently shorten it.
	while _xr_cam_pose_history.size() > 1 and _xr_cam_pose_history[0][0] < pose_usec - 500_000:
		_xr_cam_pose_history.pop_front()

	# Health check for the stamping above: it is only valid if the pdt read this frame belongs to
	# the pose read beside it, i.e. if pdt advances exactly once per _process. When it does, the
	# summed pdt deltas over any window equal the summed wall-clock deltas, because both telescope
	# to (last - first):
	#   ratio ~ 1.00, dupes = 0   -> lockstep, stamping is sound
	#   dupes > 0, ratio < 1      -> pdt repeats across _process calls; it is stale on those
	#                                frames and the stamps carry a VARIABLE error
	#   ratio > 1                 -> pdt skips ahead of the main loop
	# Per-line ratio scatter of a few percent is expected and benign: it is exactly the lead's
	# drift across that window divided by the window length, not a desync. pdt_delta_ms should
	# land on integer multiples of the display period (13.89ms at 72Hz), the minimum being one.
	# NOTE this cannot detect a CONSTANT one-frame stagger -- that keeps ratio at 1.00 with no
	# dupes, hiding entirely in the absolute offset. It rules out the variable failure only; the
	# constant residual is what POSE_LOOKUP_TRIM_MS is for.
	# Only on the Android plugin path with a monotonic offset; skipped entirely on desktop.
	if debug_prints_enabled and _xr_stamp_poses and not _cam_ts_realtime:
		if pdt != 0:
			# The very first frame only seeds the reference; every later frame contributes one delta.
			if _pdt_prev != 0:
				var d_pdt_ns := pdt - _pdt_prev
				var lead_ms := float(pdt - (now_usec * 1000 + _xr_clock_offset_ns)) / 1.0e6
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

		_lead_print_timer += _delta
		if _lead_print_timer >= 1.0 and _pdt_deltas > 0 and _pdt_wall_advance_usec > 0:
			_lead_print_timer = 0.0
			print("[opencv_aruco] [main_3d::_process] openxr pdt sync: frames=%d dupes=%d ratio=%.4f pdt_delta_ms=%.2f..%.2f lead_ms=%.2f (%.2f..%.2f)" % [
					_pdt_deltas, _pdt_dupes,
					float(_pdt_advance_ns) / float(_pdt_wall_advance_usec * 1000),
					_pdt_delta_min_ns / 1.0e6, _pdt_delta_max_ns / 1.0e6,
					_lead_sum_ms / _pdt_deltas, _lead_min_ms, _lead_max_ms])
			_pdt_deltas = 0
			_pdt_dupes = 0
			_pdt_advance_ns = 0
			_pdt_wall_advance_usec = 0
			_lead_sum_ms = 0.0

	# Device validation for the xrLocateSpace switch. Its own timer and its own condition, NOT
	# folded into the pdt block above: this is what decides whether use_xr_locate_space may be
	# turned on, so it has to keep working on a deploy where the pdt stamping is off.
	if debug_prints_enabled and _head_locator != null:
		_locate_check_timer += _delta
		if _locate_check_timer >= 1.0:
			_locate_check_timer = 0.0
			_check_xr_locate(pdt)

	# (a) Collect the latest finished detection and bake it into the scene (main thread ->
	# scene-tree writes are safe here); also re-fills the task slot from the pending frame.
	_poll_detection_task()

	# (b) Desktop-only: pull the newest CameraServer frame via GPU->CPU readback. On Android the
	# plugin pushes frames through _on_android_camera_frame instead and cam_texture stays null.
	if cam_texture == null:
		return
	var img := cam_texture.get_image()
	if img == null:
		return
	# format lookup table https://docs.godotengine.org/en/stable/classes/class_image.html#enum-image-format
	if debug_prints_enabled:
		print("[opencv_aruco] [main_3d::_process] readback: image_format=%d" % img.get_format())

	_tcp_send_timer += _delta
	if _tcp_send_timer >= TCP_SEND_INTERVAL:
		_tcp_send_timer = 0.0
		if stream_peer != null and stream_peer.get_status() == StreamPeerTCP.STATUS_CONNECTED:
			_send_frame_tcp(img)

	# No sensor timestamp on this path -- approximate the capture time as CAMERA_LATENCY_MS ago
	# and sample the head pose for that moment right here (it travels with the frame).
	_flow_frame_counter += 1
	var capture_usec := now_usec - int(CAMERA_LATENCY_MS * 1000.0)
	if _tracing(_flow_frame_counter):
		print("[opencv_aruco] [main_3d::_process] flow #%d (1-4) desktop path: no sensor timestamp, capture time guessed as now - camera_latency_ms=%.1f" % [
				_flow_frame_counter, CAMERA_LATENCY_MS])
	_dispatch_detection(img, capture_usec, _head_pose_at(capture_usec), _flow_frame_counter)

# Hand a frame + its capture time + the head pose at that time to the detection task; at most
# one task runs at a time, and only the newest frame waits for the slot (frame drop).
func _dispatch_detection(img: Image, capture_usec: int, cam_xform: Transform3D, frame_id: int) -> void:
	_poll_detection_task()               # a just-finished task frees the slot for this frame
	if _detect_task_id != -1:
		# Task still running -> park the frame; overwrites any unconsumed frame -> frame drop.
		var was_pending := _has_pending
		var dropped_id := _pending_frame_id
		_pending_image = img
		_pending_capture_usec = capture_usec
		_pending_cam_xform = cam_xform
		_pending_frame_id = frame_id
		_has_pending = true
		if _tracing(frame_id):
			print("[opencv_aruco] [main_3d::_dispatch_detection] flow #%d (5) task busy, frame parked: %s" % [
					frame_id,
					("dropped_frame=%d (newest frame wins)" % dropped_id) if was_pending else "pending_slot_was_free=true"])
		return
	_start_detection_task(img, capture_usec, cam_xform, frame_id)
	if _tracing(frame_id):
		print("[opencv_aruco] [main_3d::_dispatch_detection] flow #%d (5) handed to worker: task_id=%d" % [
				frame_id, _detect_task_id])

# Main thread only. If the in-flight task has finished: clean it up, apply its result, and start
# the pending frame (if any). Called from _process AND before every dispatch, so a finished task
# is collected at render rate or camera rate, whichever fires first.
func _poll_detection_task() -> void:
	if _detect_task_id == -1 or not WorkerThreadPool.is_task_completed(_detect_task_id):
		return
	# Mandatory cleanup of every finished task; returns immediately here (the task is done) and
	# doubles as the memory barrier that makes the task's _result_* writes visible to this thread.
	WorkerThreadPool.wait_for_task_completion(_detect_task_id)
	_detect_task_id = -1
	_apply_detection_result()
	if _has_pending:
		var img := _pending_image
		var capture_usec := _pending_capture_usec
		var cam_xform := _pending_cam_xform
		var frame_id := _pending_frame_id
		_pending_image = null
		_has_pending = false
		_start_detection_task(img, capture_usec, cam_xform, frame_id)
		if _tracing(frame_id):
			print("[opencv_aruco] [main_3d::_poll_detection_task] flow #%d (5b) pending frame handed to worker: task_id=%d" % [
					frame_id, _detect_task_id])

func _start_detection_task(img: Image, capture_usec: int, cam_xform: Transform3D, frame_id: int) -> void:
	# The one safe moment to rewrite _marker_size_table: main thread, no task in flight (the id is
	# assigned on the very next line). Both entry points -- a fresh frame from _dispatch_detection
	# and a parked one from _poll_detection_task -- come through here, so the contract holds without
	# either caller having to know about it.
	_sync_marker_sizes()
	_detect_task_id = WorkerThreadPool.add_task(_detect_frame.bind(img, capture_usec, cam_xform, frame_id),
			false, "opencv_aruco marker detection")

# The patch node for a marker id, created on first sight (main thread only -- scene-tree writes).
# The marker_nodes lookup is what guarantees one node per id: an id already in the registry gets
# its existing node back, so no second node can ever be built for it.
# Patches live under the (stationary) XROrigin3D, never under XRCamera3D: solvePnP gives a
# CAMERA-relative pose, which as a camera child would be re-multiplied by the LIVE head transform
# every frame, so a stale detection would ride the head and "swim" when you move. Anchored in
# world space instead, the pose is baked once at detection time and stays on the real marker.
func _get_or_create_patch(id: int) -> Node3D:
	if marker_nodes.has(id):
		return marker_nodes[id]
	var patch := Node3D.new()
	patch.name = "aruco_patch%d" % id      # cosmetic (remote scene tree); identity is the dict key
	var mesh_instance := MeshInstance3D.new()
	mesh_instance.mesh = _patch_mesh
	# The BoxMesh is a unit cube and the baked pose is rigid (scale 1), so this local scale IS the
	# cuboid's size in meters -- x/y from the id's real marker size, z the authored thickness.
	var size := _marker_size_for(id)
	mesh_instance.scale = Vector3(size, size, PATCH_THICKNESS)
	patch.add_child(mesh_instance)
	# Into the tree BEFORE the caller assigns global_transform (which needs a tree position).
	xr_origin.add_child(patch)
	marker_nodes[id] = patch
	if debug_prints_enabled:
		print("[opencv_aruco] [main_3d::_get_or_create_patch] patch created: id=%d size=%.3f patches=%d" % [
				id, size, marker_nodes.size()])
	return patch


# Apply the finished detection (main thread only). The markers come back from the C++ side
# already in WORLD space -- baked with the head pose at the frame's capture time, which
# travelled with the frame -- so applying is a plain assignment.
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
# CAMERA space. Main thread only, called from _apply_detection_result -- the worker never touches
# any of this.
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
	# Opened lazily, on the first kept sample: with the flag off this function costs one bool test
	# per detection and never touches the filesystem.
	if _handeye_file == null:
		_handeye_file = FileAccess.open(HANDEYE_PATH, FileAccess.WRITE)
		if _handeye_file == null:
			push_error("[opencv_aruco] [main_3d::_handeye_record] cannot open %s: err=%d (capture disabled)" % [
					HANDEYE_PATH, FileAccess.get_open_error()])
			handeye_capture = false
			return
		print("[opencv_aruco] [main_3d::_handeye_record] handeye capture started: path=%s" % ProjectSettings.globalize_path(HANDEYE_PATH))
	# Exactly the pose the C++ pre-multiplied onto every solvePnP result, so inverting it undoes
	# that one step and nothing else -- what comes back is the raw camera-space marker pose,
	# independent of whatever _lens_pose currently holds. That independence is the point: the
	# samples stay valid even if the lens pose is edited between capture and solve.
	var inv := (head * _lens_pose).affine_inverse()
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
		print("[opencv_aruco] [main_3d::_handeye_record] handeye samples=%d/%d (vary head pitch AND yaw; rotation is what constrains the solve)" % [
				_handeye_kept, HANDEYE_MAX_SAMPLES])


func _apply_detection_result() -> void:
	var markers: Dictionary = _result_markers
	var result_id: int = _result_frame_id
	if handeye_capture:
		# _result_cam_xform is the head pose the markers were actually baked with, i.e. the pose at
		# CAPTURE time rather than the live one -- the same pairing the live path uses, so a
		# calibration solved from these samples is valid for it.
		_handeye_record(_result_cam_xform, markers)
	if _tracing(result_id):
		print("[opencv_aruco] [main_3d::_apply_detection_result] flow #%d (8) applying result: capture_usec=%d result_age_ms=%.1f markers=%d" % [
				result_id, _result_capture_usec,
				(Time.get_ticks_usec() - _result_capture_usec) / 1000.0, markers.size()])
		# (9) what the whole timing correction was worth: the head pose the markers were baked
		# with vs. the LIVE one -- that difference is exactly the swim we avoid.
		var live_xform := xr_camera.global_transform
		var drift_cm := live_xform.origin.distance_to(_result_cam_xform.origin) * 100.0
		var turn_deg := rad_to_deg(_result_cam_xform.basis.get_rotation_quaternion().angle_to(
				live_xform.basis.get_rotation_quaternion()))
		print("[opencv_aruco] [main_3d::_apply_detection_result] flow #%d (9) head pose at capture: pos=%v live_pos=%v drift_cm=%.1f turn_deg=%.1f" % [
				result_id, _result_cam_xform.origin, live_xform.origin, drift_cm, turn_deg])
	var now_usec := Time.get_ticks_usec()
	var seen_ids: Array = []
	for id in markers:
		# markers[id] is already WORLD space; freeze it there, so the head can move between
		# detections without dragging the patch along.
		_marker_poses[id] = markers[id]        # the id-keyed record the public API serves
		_get_or_create_patch(id).global_transform = markers[id]
		_marker_last_seen[id] = now_usec
		seen_ids.append(id)
		if _tracing(result_id):
			# frozen at this world pose until the next detection
			print("[opencv_aruco] [main_3d::_apply_detection_result] flow #%d (10) marker assigned: id=%d world_pos=%v" % [
					result_id, id, markers[id].origin])

	# Drop patches whose marker has been missing for longer than the grace period. Pruning runs
	# HERE, on a fresh detection result, not on a timer: absence is only evidence that a marker is
	# gone once a frame that could have contained it has been looked at. If the camera stalls, the
	# patches stay put instead of evaporating on "no news".
	# Iterating over keys() takes a copy, so erasing inside the loop is safe.
	for id in marker_nodes.keys():
		var unseen_usec: int = now_usec - int(_marker_last_seen.get(id, now_usec))
		if unseen_usec <= int(PATCH_LOST_TIMEOUT_MS * 1000.0):
			continue
		marker_nodes[id].queue_free()
		marker_nodes.erase(id)
		# _marker_last_seen is NOT erased with the node: it is the public freshness record behind
		# marker_age_ms(), and a consumer asking "how stale is my last good pose?" must still get an
		# answer after the mesh is gone. Safe because this loop keys off marker_nodes.keys(), so a
		# leftover entry cannot resurrect a patch, and _get_or_create_patch rebuilds the node
		# cleanly if the marker comes back. Bounded at 250 entries by ARUCO_MIP_36h12, same as
		# _marker_poses.
		if debug_prints_enabled:
			print("[opencv_aruco] [main_3d::_apply_detection_result] patch deleted: id=%d unseen_ms=%.0f patches=%d" % [
					id, unseen_usec / 1000.0, marker_nodes.size()])

	# After the prune, so a handler reacting to this signal sees the final scene state.
	if not seen_ids.is_empty():
		markers_updated.emit(seen_ids)

# Runs on a WorkerThreadPool thread: ONE frame's OpenCV detection (detectMarkers + solvePnP) off
# the main thread. Touches only `processor`, read-only config and the _result_* slot -- never the
# scene tree. Writing _result_* without a lock is safe: the main thread reads the slot only after
# wait_for_task_completion() on this task.
func _detect_frame(img: Image, capture_usec: int, cam_xform: Transform3D, frame_id: int) -> void:
	# No conversion: the C++ side handles 1ch (Quest Y-plane), 3ch (RGB), and 4ch (RGBA).
	var t0 := Time.get_ticks_usec()
	var traced := _tracing(frame_id)
	if traced:
		print("[opencv_aruco] [main_3d::_detect_frame] flow #%d (6) worker picked it up: frame_age_ms=%.1f" % [
				frame_id, (t0 - capture_usec) / 1000.0])
	# The exported intrinsics are for the native 640x480 frame; all four components scale with
	# the image, so image_downscale_factor is applied at use time.
	var intrinsics := camera_intrinsics * image_downscale_factor
	# The two transforms that hold for EVERY marker -- head pose at capture time and physical
	# lens offset -- combined into ONE camera->world pose. The C++ side pre-multiplies it onto
	# each solvePnP pose, so the returned Dictionary is already in WORLD space.
	var cam_to_world := cam_xform * _lens_pose
	var markers: Dictionary = processor.get_6dof_of_all_aruco_patches_from_godot_image(img, _marker_size_table, aruco_patch_size, image_downscale_factor, intrinsics, camera_distortion, cam_to_world)
	# Guarded inline rather than via a helper function: a helper would build this string on every
	# detection before it could check the flag.
	if debug_prints_enabled:
		var detect_ms := (Time.get_ticks_usec() - t0) / 1000.0
		var tracking_fps := 1000.0 / detect_ms if detect_ms > 0.0 else 0.0
		print("[opencv_aruco] [main_3d::_detect_frame] detect_ms=%.1f tracking_fps=%.1f render_fps=%d frame_age_ms=%.1f markers=%d" % [
				detect_ms, tracking_fps, Engine.get_frames_per_second(), (t0 - capture_usec) / 1000.0, markers.size()])
		if traced:
			# The marker poses are baked with the head pose AT EXPOSURE TIME -- however long the
			# detection took, that fact does not age.
			print("[opencv_aruco] [main_3d::_detect_frame] flow #%d (7) detection done: detect_ms=%.1f markers=%d (world space; parking result for the main thread)" % [
					frame_id, detect_ms, markers.size()])
	_result_markers = markers
	_result_capture_usec = capture_usec
	_result_cam_xform = cam_xform
	_result_frame_id = frame_id


# Head pose at t_usec, interpolated between the two nearest history samples (the raw history has
# one sample per RENDERED frame -- ~19ms at the ~55fps this hits on the Quest, which is below the
# 72Hz display rate; interpolating removes that quantisation).
# Falls back to the oldest/newest sample (or the live pose) at the edges of the history.
func _head_pose_at(t_usec: int) -> Transform3D:
	if _xr_cam_pose_history.is_empty():
		return xr_camera.global_transform
	if t_usec <= _xr_cam_pose_history[0][0]:
		return _xr_cam_pose_history[0][1]
	for i in range(_xr_cam_pose_history.size() - 1, -1, -1):
		if _xr_cam_pose_history[i][0] <= t_usec:
			if i == _xr_cam_pose_history.size() - 1:
				return _xr_cam_pose_history[i][1]
			var t0: int = _xr_cam_pose_history[i][0]
			var t1: int = _xr_cam_pose_history[i + 1][0]
			var w := clampf(float(t_usec - t0) / float(t1 - t0), 0.0, 1.0)
			var p0: Transform3D = _xr_cam_pose_history[i][1]
			var p1: Transform3D = _xr_cam_pose_history[i + 1][1]
			return p0.interpolate_with(p1, w)
	return _xr_cam_pose_history[0][1]


# OpenXR play space -> world. The locator returns the head relative to the PLAY space, which is
# what XROrigin3D stands for -- but XRCamera3D's own transform additionally carries XRServer's
# reference frame (whatever center_on_hmd() last set) and the world scale. Reapplying both here is
# what makes check (A) below a test of the TIME argument rather than of this conversion; with an
# untouched XROrigin3D, no recentering and world_scale 1 all three factors are identity, so this
# costs nothing today and stops the poses drifting off the moment locomotion or scaling appears.
func _play_space_to_world(head: Transform3D) -> Transform3D:
	var scaled := Transform3D(head.basis, head.origin * XRServer.world_scale)
	return xr_origin.global_transform * XRServer.get_reference_frame() * scaled


# One probe: head pose at xr_time, expressed in world space and compared against `live`. Returns
# the two scalars the checks below print, or just valid=false if the runtime would not answer.
# Factored out because the whole point of the check is comparing the SAME measurement at three
# different times -- doing that inline three times invites the three copies drifting apart.
func _locate_delta(xr_time: int, live: Transform3D) -> Dictionary:
	var loc: Dictionary = _head_locator.locate_head(xr_time)
	if not loc.get("valid", false):
		return {"valid": false, "result": loc["result"], "flags": loc["flags"]}
	var raw: Transform3D = loc["transform"]
	var pose := _play_space_to_world(raw)
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
		print("[opencv_aruco] [main_3d::_check_xr_locate] openxr locate check: no predicted display time yet")
		return
	if not _head_locator.is_ready():
		print("[opencv_aruco] [main_3d::_check_xr_locate] openxr locate check: locator not ready (no session or play space yet)")
		return

	# Read once, so all three probes are compared against the same reference pose.
	var live := xr_camera.global_transform
	var next_time: int = _xr_api.get_next_frame_time()

	var at_pdt := _locate_delta(pdt, live)
	var at_next := _locate_delta(next_time, live)
	print("[opencv_aruco] [main_3d::_check_xr_locate] openxr locate check (A) vs XRCamera3D: pdt=%s next_frame=%s gap_ms=%.2f (MOVE YOUR HEAD; the one at ~0 is the time XRCamera3D's pose belongs to)" % [
			_fmt_locate(at_pdt), _fmt_locate(at_next), (next_time - pdt) / 1.0e6])

	var in_past := _locate_delta(pdt - int(LOCATE_PAST_PROBE_MS * 1.0e6), live)
	print("[opencv_aruco] [main_3d::_check_xr_locate] openxr locate check (B) at pdt-%.0fms: %s (valid => historical queries work, and the offsets are the motion over that interval)" % [
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
		print("[opencv_aruco] [main_3d::_check_xr_locate] openxr locate raw: ps=%v cam_local=%v cam_world=%v origin=%v ref=%v scale=%.2f ps_move_mm=%.2f" % [
				at_pdt["ps_origin"],
				xr_camera.transform.origin, live.origin,
				xr_origin.global_transform.origin, XRServer.get_reference_frame().origin,
				XRServer.world_scale,
				at_pdt["ps_origin"].distance_to(in_past["ps_origin"]) * 1000.0])


func _fmt_locate(d: Dictionary) -> String:
	if not d["valid"]:
		return "INVALID(result=%d flags=%d)" % [d["result"], d["flags"]]
	return "[pos_mm=%.2f rot_deg=%.3f tracked=%s]" % [d["pos_mm"], d["rot_deg"], d["tracked"]]



func _exit_tree() -> void:
	if android_camera != null and _android_cam_started:
		android_camera.stop_camera()
	# A detection task may still be running on the pool; block until it is done so its bound
	# callable (which captures self) doesn't outlive the scene. Costs at most one detection (~80ms).
	if _detect_task_id != -1:
		WorkerThreadPool.wait_for_task_completion(_detect_task_id)
		_detect_task_id = -1
	# Closed AFTER the task drain above, so a detection finishing during teardown cannot write into
	# a closed handle. Every sample is already flushed, so this only tidies up on a clean exit --
	# a force-stop skips it and loses nothing.
	if _handeye_file != null:
		_handeye_file.close()
		_handeye_file = null
		print("[opencv_aruco] [main_3d::_exit_tree] handeye capture closed: samples=%d path=%s" % [
				_handeye_kept, ProjectSettings.globalize_path(HANDEYE_PATH)])

# Buffers one frame for the debug streamer and pushes what fits right now. Never blocks; drops the
# frame outright while the previous one is still on its way out (see _tcp_out).
func _send_frame_tcp(img: Image) -> void:
	if stream_peer == null:
		return

	stream_peer.poll()

	if stream_peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
		return

	if not _tcp_out.is_empty():
		_tcp_dropped += 1
		if debug_prints_enabled and _tcp_dropped % 100 == 0:
			print("[opencv_aruco] [main_3d::_send_frame_tcp] receiver behind: dropped=%d backlog=%d bytes" % [
					_tcp_dropped, _tcp_out.size()])
		return

	var bytes: PackedByteArray = img.get_data()

	# 16-byte header, big-endian: width, height, Image.Format, payload size (tools/tcp_receiver.py).
	# Written by hand rather than via put_u32 because header and payload have to reach the buffer as
	# one block -- a partially written header would desync the receiver for good.
	var frame := PackedByteArray()
	frame.append_array(_be_u32(img.get_width()))
	frame.append_array(_be_u32(img.get_height()))
	frame.append_array(_be_u32(img.get_format()))
	frame.append_array(_be_u32(bytes.size()))
	frame.append_array(bytes)

	_tcp_out = frame
	_flush_tcp()


func _be_u32(value: int) -> PackedByteArray:
	return PackedByteArray([
			(value >> 24) & 0xFF, (value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF])


# Hands the socket as much of the buffered frame as it will take without waiting. Called once per
# _process from _poll_tcp and again right after buffering, so a frame that fits leaves the same frame.
func _flush_tcp() -> void:
	if _tcp_out.is_empty() or stream_peer == null:
		return

	var res: Array = stream_peer.put_partial_data(_tcp_out)
	var err: int = res[0]
	var sent: int = res[1]

	if err != OK:
		push_error("[opencv_aruco] [main_3d::_flush_tcp] put_partial_data failed: err=%d" % err)
		_tcp_out.clear()
		return

	if sent >= _tcp_out.size():
		_tcp_out.clear()
	elif sent > 0:
		_tcp_out = _tcp_out.slice(sent)

func _connect_tcp() -> void:
	stream_peer = StreamPeerTCP.new()
	stream_peer.big_endian = true

	var err := stream_peer.connect_to_host(TCP_HOST, TCP_PORT)
	if debug_prints_enabled:
		print("[opencv_aruco] [main_3d::_connect_tcp] connect_to_host: err=%d" % err)


func _poll_tcp(delta: float) -> void:
	if stream_peer == null:
		_connect_tcp()
		return

	stream_peer.poll()

	var status := stream_peer.get_status()

	if status != _last_tcp_status:
		if debug_prints_enabled:
			print("[opencv_aruco] [main_3d::_poll_tcp] status changed: from=%d to=%d" % [_last_tcp_status, status])
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
			if debug_prints_enabled:
				print("[opencv_aruco] [main_3d::_poll_tcp] reconnecting")
			_connect_tcp()
#command to stop (once adb is added to PATH)
# adb shell am force-stop de.unigreifswald.opencvaruco
