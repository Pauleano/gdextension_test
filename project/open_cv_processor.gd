# Named so consumers can type their reference to it (@export var marker_source: ArucoMarkerSource)
# and get the public marker API below checked at edit time instead of as a runtime "Invalid call".
# NOT a Node3D: this script sits on the OpenCVProcessor node (a plain Node) under the XROrigin3D
# scene root, so a consumer holding an ArucoMarkerSource holds a Node. Nothing here ever used its
# own transform -- every pose goes through xr_origin/xr_camera below -- so the loss costs nothing,
# but code expecting to parent something to a marker source has to reach for get_marker_node().
class_name ArucoMarkerSource
extends OpenCVProcessor

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
# The marker sizes (aruco_patch_size, aruco_patch_sizes) are NATIVE properties of this node now --
# see OpenCVProcessor.h. They still show up in the inspector, and get_marker_size(id) resolves an
# id against them; this script calls that same function to scale the rendered box, so the metric
# scale solvePnP used and the mesh drawn over the marker come from one place by construction.

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
# OpenCVProcessor before calling dump_build_info_and_intrinsics(). Off by default: with no tcp_receiver.py listening the
# TCP reconnect logs alone would print once per second for the whole session, and the per-frame
# timing below fires at the camera rate. Errors are never gated and print regardless. DEBUG_FLOW is
# a SUB-switch of this one: flow tracing needs both (see _tracing).
# Format for every debug line: "[opencv_aruco] [aruco::function] event: key=value" -- the fixed
# prefix is what makes them findable in logcat (adb logcat | grep opencv_aruco).
@export var debug_prints_enabled := false:
	set(value):
		debug_prints_enabled = value
		# Mirror every change into the C++ static right away. The extension gates its ACV_DBG lines
		# on a flag that only GDScript can set, so pushing it once in _ready meant a toggle from the
		# remote inspector of a RUNNING deploy silenced this script's prints while the extension kept
		# logging. Assigning the property inside its own setter does not recurse in GDScript.
		# This setter is NOT early enough for the C++ side on its own: the node is built by
		# PackedScene before any property is applied, which is why everything that prints moved out
		# of the constructor and into dump_build_info_and_intrinsics() (called from _ready, right
		# after this flag).
		OpenCVProcessor.set_debug_prints_enabled(value)
		# Same deal for the second extension class; both statics are pushed from this one place so
		# a toggle in the remote inspector reaches every C++ log line, not just some of them.
		OpenXRHeadLocator.set_debug_prints_enabled(value)

# --- Camera calibration ---
# All of it -- camera_intrinsics, camera_distortion, image_downscale_factor, lens_rotation_raw,
# lens_translation -- lives on the C++ side now, as native properties of this very node (see
# OpenCVProcessor.h, which also carries the provenance of every number and the warnings about
# re-measuring them). They still appear in the inspector, and every one of them is re-read at the
# top of each detection, so an edit in the remote inspector of a running deploy takes effect on the
# next frame. That is new for the lens pose: it used to be derived once here in _ready, so editing
# it mid-run did nothing at all until a restart.
# get_lens_pose() returns the decoded transform the hand-eye capture needs -- which is why that
# capture had to become a CHILD of this node rather than a sibling.

# --- Hand-eye calibration capture ---
# Gone to the DetectionDiagnostics child node (project/detection_diagnostics.gd), together with the
# drift readout and the OpenXR timing checks. It is a measurement RUN -- a file, a motion gate and a
# sample cap of its own -- and nothing here ever read it back. The switch is `handeye_capture` on
# THAT node now, one level down in the remote inspector; the only thing this script still knows
# about it is the submit() call in _apply_detection_result.


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
# Counted on the LIVE path (see _on_android_camera_frame) and printed by flow trace (4), so it stays
# here even though its siblings -- the once-a-second pdt/locate checks and their accumulators -- moved
# to the DetectionDiagnostics child.
var _locate_fallback_count := 0        # frames on which the locator had no valid pose
# All of these reach UP: this script sits on the OpenCVProcessor node, a CHILD of the XROrigin3D
# scene root, so the nodes it works with are addressed through "..".
@onready var cam_preview: TextureRect = $"../CameraLayer/CameraPreview"
@onready var xr_origin: XROrigin3D = $".."
@onready var xr_camera: XRCamera3D = $"../XRCamera3D"
# Parent for the runtime patch nodes. They are created and freed continuously, so they get their own
# container instead of being mixed in among the authored siblings -- otherwise the remote scene tree
# of a running deploy churns in the middle of the nodes you are trying to inspect. It sits at
# identity under the origin and _apply_detection_result assigns global_transform, so it is free.
@onready var patch_root: Node3D = $"../Patches"

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
# output slots, written by the task; the main thread reads them only AFTER
# wait_for_task_completion(), which is the synchronization point (no lock needed)
var _result_markers: Dictionary = {}
# id -> PackedVector2Array of the 4 marker corners in the frame's own pixel space, straight from
# the C++ detector. Debug data for the TCP overlay ONLY -- the poses above are what the app runs on.
var _result_corners: Dictionary = {}
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


# The frame the in-flight (or just finished) detection task is working on. Handed to the debug
# streamer once the worker is done, because that detection's corners only exist then -- a frame sent
# at arrival time could never carry them. Overwritten on every dispatch.
var _detecting_img: Image
# Debug frame streamer (project/tcp_debug_stream.gd). A CHILD node, because it calls
# project_marker_corners() on this one. Optional on purpose: no such child in the scene simply means
# no streaming, so the diagnostic can be removed by deleting a node rather than editing this script.
@onready var _debug_stream: TcpDebugStream = get_node_or_null("TcpDebugStream")
# Measurement apparatus (project/detection_diagnostics.gd): hand-eye capture, marker drift readout,
# OpenXR timing validation. A CHILD for the same reason as the streamer -- the hand-eye capture calls
# get_lens_pose() on this one -- and optional in the same way, so the whole facility comes off by
# deleting a node. Three touch points and no more: configure() in _setup_xr_locator, sample() in
# _process, submit() in _apply_detection_result.
@onready var _diagnostics: DetectionDiagnostics = get_node_or_null("DetectionDiagnostics")


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

# Put the current marker sizes on the live patch meshes. The id -> size resolution itself is
# get_marker_size(), a native method of this node, so there is nothing left to mirror into a table
# for the C++ side -- it reads its own properties at the top of every detection.
# Existing patches got their scale once, at creation time (_get_or_create_patch). Without this a
# size change would only reach the mesh after the marker had been lost for PATCH_LOST_TIMEOUT_MS and
# the node was rebuilt -- the box would keep its old edge length while the pose already used the new
# one. Compared approximately because scale components are 32-bit: an exact != against the value
# from get_marker_size would fire every single time.
# Main thread only (it writes to the scene tree); the old "no task in flight" contract went with the
# table it used to rebuild.
func _sync_marker_sizes() -> void:
	for id in marker_nodes:
		var size := get_marker_size(id)
		# The one child added in _get_or_create_patch; the patch node itself must stay scale-free,
		# it carries the baked pose.
		var mesh_instance: MeshInstance3D = marker_nodes[id].get_child(0)
		if not is_equal_approx(mesh_instance.scale.x, size):
			mesh_instance.scale = Vector3(size, size, PATCH_THICKNESS)
			if debug_prints_enabled:
				print("[opencv_aruco] [aruco::_sync_marker_sizes] patch resized: id=%d size=%.3f" % [id, size])

####################################################################################################

func _ready() -> void:
	# The property setter already pushed this into the extension at scene-instantiation time; repeat
	# it here so the flag is also correct when the scene does NOT override the default (the setter
	# never fires then) and a previous run left the static true.
	# This ORDER is the whole reason dump_build_info_and_intrinsics() exists as a separate call: the
	# C++ side prints (OpenCV build info + Quest intrinsics dump), and its constructor now runs when
	# the SCENE is instantiated -- before any exported property is applied, and with no scripted node
	# above us to push the flag any earlier. So the constructor was emptied of everything that talks,
	# and the talking part waits here, one line after the flag is set. Nothing in the detection
	# depends on it; drop the call and only the log lines go away.
	OpenCVProcessor.set_debug_prints_enabled(debug_prints_enabled)
	dump_build_info_and_intrinsics()

	# Nothing to derive here any more: the lens pose is rebuilt by the C++ setters of
	# lens_rotation_raw / lens_translation, so it is already correct and stays correct after an
	# inspector edit rather than being frozen at whatever _ready saw.

	# No patch nodes to find: they are created on demand as markers turn up (see
	# _get_or_create_patch) and freed again when they stop being detected.

	# Nothing to size yet either (no patches exist), but the call keeps the "sizes reach the meshes"
	# path in one place; it re-runs before every detection task from _start_detection_task.
	_sync_marker_sizes()

	# No worker setup needed: detection runs as one-shot WorkerThreadPool tasks, dispatched on
	# demand once frames arrive (see _dispatch_detection) -- long after _ready has finished.

	# BEFORE the camera branch below, which can start delivering frames synchronously: those frames
	# want the locator. Safe to do here because OpenXR is initialised by the ENGINE, before the main
	# scene is loaded at all -- project.godot has xr/openxr/enabled=true, and xr_startup.gd only ever
	# looks the interface up and flips viewport flags, it never calls initialize() itself. So the
	# is_initialized() check in _setup_xr_locator does not depend on node order. XRStartup sitting
	# above this node among the root's children (children _ready in tree order) is still the right
	# arrangement -- passthrough and use_xr are set before any frame is dispatched -- but it is
	# belt-and-braces, not the thing that makes the locator work.
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

	# Nothing to do for the debug stream here: the TcpDebugStream child owns its socket end to end
	# and reconciles it against its own `enabled` flag from its own _process.

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
			print("[opencv_aruco] [aruco::_setup_xr_locator] no OpenXR: head locator disabled, pose history stays the only path")
		return
	_xr_api = OpenXRAPIExtension.new()
	_head_locator = OpenXRHeadLocator.new()
	add_child(_head_locator)
	# connect() by NAME, not xr_interface.session_stopping.connect(): the var is statically typed
	# XRInterface, which has no such signal -- only the OpenXRInterface behind it does.
	if xr_interface.has_signal("session_stopping"):
		xr_interface.connect("session_stopping", Callable(_head_locator, "release"))
	# The once-a-second validation lines live on the diagnostics node. This is the single place that
	# decides whether the deploy has OpenXR at all, so it is also the only place that may hand the
	# locator over -- a node that never gets configured stays silent by itself. _play_space_to_world
	# travels as a Callable rather than being reimplemented there: check (A) is a test of the TIME
	# argument only as long as both sides convert identically. xr_camera/xr_origin go the same way
	# rather than being read back off this node, which is what lets that node type its reference to
	# us as the NATIVE OpenCVProcessor instead of an untyped Node (see its declaration there).
	if _diagnostics != null:
		_diagnostics.configure(_head_locator, _xr_api, _play_space_to_world, xr_camera, xr_origin)
	if debug_prints_enabled:
		print("[opencv_aruco] [aruco::_setup_xr_locator] head locator created: use_xr_locate_space=%s (validation prints once a second while debug is on)" % use_xr_locate_space)

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
			print("[opencv_aruco] [aruco::_on_camera_feeds_updated] feed: index=%d id=%d name=%s" % [i, f.get_id(), f.get_name()])

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
			print("[opencv_aruco] [aruco::_on_camera_feeds_updated] format: index=%d value=%s" % [j, formats[j]])
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
		print("[opencv_aruco] [aruco::_on_camera_feeds_updated] feed activated: id=%d name=%s feed_count=%d" % [feed.get_id(), feed.get_name(), feed_count])

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
		print("[opencv_aruco] [aruco::_start_android_camera] starting camera: id=%s clock=%s feeds=%s" % [
				cam_id, "realtime" if _cam_ts_realtime else "monotonic", cameras])
		# Every frame's exposure time is converted with: cap_usec = (timestamp_ns - offset_ns) / 1000
		print("[opencv_aruco] [aruco::_start_android_camera] flow setup: clock bridge calibrated, camera clock is ahead of Time.get_ticks_usec() by offset_s=%.3f offset_ns=%d" % [
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
		print("[opencv_aruco] [aruco::_on_android_camera_frame] flow #%d (1) frame arrives: width=%d height=%d bytes=%d timestamp_ns=%d (grayscale Y-plane, 1 byte/pixel)" % [
				frame_id, width, height, data.size(), timestamp_ns])
		print("[opencv_aruco] [aruco::_on_android_camera_frame] flow #%d (2) exposure time on godot clock: cap_usec=%d age_ms=%.1f source=%s" % [
				frame_id, cap_usec, lag_ms, "FALLBACK_GUESS_timestamp_rejected" if used_fallback else "sensor_timestamp"])

	# The lookup target is the exposure time itself: the history entries carry the time their pose
	# actually describes, so the two timelines already line up and only the residual trim is left.
	# cap_usec itself is NOT shifted -- it travels on as the frame's true age (see frame_age_ms).
	var lookup_usec := cap_usec - int(POSE_LOOKUP_TRIM_MS * 1000.0)
	if traced:
		print("[opencv_aruco] [aruco::_on_android_camera_frame] flow #%d (3) pose lookup target: target_usec=%d trim_ms=%.1f (history is pdt-stamped, so no prediction lead to undo)" % [
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
		print("[opencv_aruco] [aruco::_on_android_camera_frame] flow #%d (4) head pose at capture: pos=%v source=%s locate_fallbacks=%d (travels with the frame)" % [
				frame_id, cam_xform.origin, pose_source, _locate_fallback_count])

	if debug_prints_enabled and (_cam_frame_count == 1 or _cam_frame_count % 300 == 0):
		print("[opencv_aruco] [aruco::_on_android_camera_frame] frame: count=%d width=%d height=%d lag_ms=%.1f clock=%s fallbacks=%d" % [
				_cam_frame_count, width, height, lag_ms,
				"realtime" if _cam_ts_realtime else "monotonic", _cam_fallback_count])

	var img := Image.create_from_data(width, height, false, Image.FORMAT_L8, data)

	# debug output (grayscale): preview overlay. The TCP streamer is NOT fed here -- it sends the
	# frame together with that frame's detected corners, which only exist once the worker is done
	# (see TcpDebugStream.submit).
	if cam_preview.visible:
		if _preview_texture == null:
			_preview_texture = ImageTexture.create_from_image(img)
			cam_preview.texture = _preview_texture
		else:
			_preview_texture.update(img)

	_dispatch_detection(img, cap_usec, cam_xform, frame_id)


# True if this frame's journey should be traced. The SAME id gives the SAME answer at every
# station, so all prints belonging to one frame appear together (see DEBUG_FLOW).
func _tracing(frame_id: int) -> bool:
	return debug_prints_enabled and DEBUG_FLOW and frame_id % DEBUG_FLOW_EVERY == 0

####################################################################################################
func _process(_delta: float) -> void:
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
	# Read once per frame and handed to the diagnostics below, whose locate check needs the same pdt
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

	# Health check for the stamping above (is pdt in lockstep with _process?) plus the once-a-second
	# xrLocateSpace device probes -- both live on the DetectionDiagnostics child, which owns their
	# timers, accumulators and the "openxr pdt sync" / "openxr locate check" lines.
	# Everything they need is handed over per frame rather than read back out of this node, above all
	# _xr_clock_offset_ns: it is resampled here (_resample_clock_offsets), and a second copy over
	# there could drift from the one the pose lookup actually uses -- which is the exact class of
	# error the check exists to catch. The pdt statistics are only meaningful while the history is
	# genuinely pdt-stamped, which is what the fourth argument says; the locate probes have their own
	# condition and keep running either way.
	if _diagnostics != null:
		_diagnostics.sample(pdt, now_usec, _xr_clock_offset_ns,
				_xr_stamp_poses and not _cam_ts_realtime, _delta)

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
		print("[opencv_aruco] [aruco::_process] readback: image_format=%d" % img.get_format())

	# No sensor timestamp on this path -- approximate the capture time as CAMERA_LATENCY_MS ago
	# and sample the head pose for that moment right here (it travels with the frame).
	_flow_frame_counter += 1
	var capture_usec := now_usec - int(CAMERA_LATENCY_MS * 1000.0)
	if _tracing(_flow_frame_counter):
		print("[opencv_aruco] [aruco::_process] flow #%d (1-4) desktop path: no sensor timestamp, capture time guessed as now - camera_latency_ms=%.1f" % [
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
			print("[opencv_aruco] [aruco::_dispatch_detection] flow #%d (5) task busy, frame parked: %s" % [
					frame_id,
					("dropped_frame=%d (newest frame wins)" % dropped_id) if was_pending else "pending_slot_was_free=true"])
		return
	_start_detection_task(img, capture_usec, cam_xform, frame_id)
	if _tracing(frame_id):
		print("[opencv_aruco] [aruco::_dispatch_detection] flow #%d (5) handed to worker: task_id=%d" % [
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
	# Only now do the frame and its corners both exist, so this is the earliest point at which the
	# overlay can be streamed as one consistent pair -- and it has to happen BEFORE the pending frame
	# below is handed on, since that overwrites _detecting_img with the next frame. Passing the image
	# as an argument is what makes that ordering safe rather than merely documented.
	if _debug_stream != null:
		_debug_stream.submit(_detecting_img, _result_corners, _result_markers, _result_cam_xform)
	if _has_pending:
		var img := _pending_image
		var capture_usec := _pending_capture_usec
		var cam_xform := _pending_cam_xform
		var frame_id := _pending_frame_id
		_pending_image = null
		_has_pending = false
		_start_detection_task(img, capture_usec, cam_xform, frame_id)
		if _tracing(frame_id):
			print("[opencv_aruco] [aruco::_poll_detection_task] flow #%d (5b) pending frame handed to worker: task_id=%d" % [
					frame_id, _detect_task_id])

func _start_detection_task(img: Image, capture_usec: int, cam_xform: Transform3D, frame_id: int) -> void:
	# Once per task is the right cadence for pushing size changes onto the patch meshes, and both
	# entry points -- a fresh frame from _dispatch_detection and a parked one from
	# _poll_detection_task -- come through here. (It no longer has to happen here for thread-safety;
	# the table this used to rebuild lives in C++ now and is resolved per detection.)
	_sync_marker_sizes()
	# Held so _poll_detection_task can hand THIS frame to the debug streamer once the task below has
	# produced the corners that belong to it.
	_detecting_img = img
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
	var size := get_marker_size(id)
	mesh_instance.scale = Vector3(size, size, PATCH_THICKNESS)
	patch.add_child(mesh_instance)
	# Into the tree BEFORE the caller assigns global_transform (which needs a tree position).
	patch_root.add_child(patch)
	marker_nodes[id] = patch
	if debug_prints_enabled:
		print("[opencv_aruco] [aruco::_get_or_create_patch] patch created: id=%d size=%.3f patches=%d" % [
				id, size, marker_nodes.size()])
	return patch


# Apply the finished detection (main thread only). The markers come back from the C++ side
# already in WORLD space -- baked with the head pose at the frame's capture time, which
# travelled with the frame -- so applying is a plain assignment.
func _apply_detection_result() -> void:
	var markers: Dictionary = _result_markers
	var result_id: int = _result_frame_id
	if _tracing(result_id):
		print("[opencv_aruco] [aruco::_apply_detection_result] flow #%d (8) applying result: capture_usec=%d result_age_ms=%.1f markers=%d" % [
				result_id, _result_capture_usec,
				(Time.get_ticks_usec() - _result_capture_usec) / 1000.0, markers.size()])
		# (9) what the whole timing correction was worth: the head pose the markers were baked
		# with vs. the LIVE one -- that difference is exactly the swim we avoid.
		var live_xform := xr_camera.global_transform
		var drift_cm := live_xform.origin.distance_to(_result_cam_xform.origin) * 100.0
		var turn_deg := rad_to_deg(_result_cam_xform.basis.get_rotation_quaternion().angle_to(
				live_xform.basis.get_rotation_quaternion()))
		print("[opencv_aruco] [aruco::_apply_detection_result] flow #%d (9) head pose at capture: pos=%v live_pos=%v drift_cm=%.1f turn_deg=%.1f" % [
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
			print("[opencv_aruco] [aruco::_apply_detection_result] flow #%d (10) marker assigned: id=%d world_pos=%v" % [
					result_id, id, markers[id].origin])

	# The measurement apparatus, fed once with this detection: the hand-eye capture writes the pair
	# to disk, the drift readout compares each marker against where it was first seen. Both want the
	# same two values, hence one call -- and _result_cam_xform rather than the live head pose,
	# because that is what the markers were actually baked with (the pose at CAPTURE time), so a
	# calibration solved from these samples is valid for the live path.
	# AFTER the loop above, so anything reading back through the public API sees poses that are
	# already applied; before the prune, which only touches nodes.
	if _diagnostics != null:
		_diagnostics.submit(_result_cam_xform, markers)

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
			print("[opencv_aruco] [aruco::_apply_detection_result] patch deleted: id=%d unseen_ms=%.0f patches=%d" % [
					id, unseen_usec / 1000.0, marker_nodes.size()])

	# After the prune, so a handler reacting to this signal sees the final scene state.
	if not seen_ids.is_empty():
		markers_updated.emit(seen_ids)

# Runs on a WorkerThreadPool thread: ONE frame's OpenCV detection (detectMarkers + solvePnP) off
# the main thread. Touches only our own C++ half, read-only config and the _result_* slot -- never
# the scene tree. Writing _result_* without a lock is safe: the main thread reads the slot only
# after wait_for_task_completion() on this task.
func _detect_frame(img: Image, capture_usec: int, cam_xform: Transform3D, frame_id: int) -> void:
	# No conversion: the C++ side handles 1ch (Quest Y-plane), 3ch (RGB), and 4ch (RGBA).
	var t0 := Time.get_ticks_usec()
	var traced := _tracing(frame_id)
	if traced:
		print("[opencv_aruco] [aruco::_detect_frame] flow #%d (6) worker picked it up: frame_age_ms=%.1f" % [
				frame_id, (t0 - capture_usec) / 1000.0])
	# Self-call: this node IS the OpenCVProcessor, so the detection runs on the very instance the
	# scene authored and the inspector configures -- no second, separately constructed detector.
	# The head pose at CAPTURE time is the only thing that has to travel with the frame; intrinsics,
	# distortion, downscale, marker sizes and the lens pose are all properties, re-read there. The
	# markers come back in WORLD space, with head_pose * lens_pose already applied to each.
	# Out-parameter for the debug overlay: Dictionaries are shared references in Godot, so the C++
	# side writes the detected pixel corners into THIS instance. Built fresh per detection (rather
	# than clearing _result_corners) so the main thread can never see a half-filled dictionary --
	# the slot is only re-pointed at the end, past the same barrier as _result_markers.
	var corners: Dictionary = {}
	var markers: Dictionary = detect_markers(img, cam_xform, corners)
	# Guarded inline rather than via a helper function: a helper would build this string on every
	# detection before it could check the flag.
	if debug_prints_enabled:
		var detect_ms := (Time.get_ticks_usec() - t0) / 1000.0
		var tracking_fps := 1000.0 / detect_ms if detect_ms > 0.0 else 0.0
		print("[opencv_aruco] [aruco::_detect_frame] detect_ms=%.1f tracking_fps=%.1f render_fps=%d frame_age_ms=%.1f markers=%d" % [
				detect_ms, tracking_fps, Engine.get_frames_per_second(), (t0 - capture_usec) / 1000.0, markers.size()])
		if traced:
			# The marker poses are baked with the head pose AT EXPOSURE TIME -- however long the
			# detection took, that fact does not age.
			print("[opencv_aruco] [aruco::_detect_frame] flow #%d (7) detection done: detect_ms=%.1f markers=%d (world space; parking result for the main thread)" % [
					frame_id, detect_ms, markers.size()])
	_result_markers = markers
	_result_corners = corners
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
# what makes check (A) on the DetectionDiagnostics node a test of the TIME argument rather than of
# this conversion -- which is why that node is handed THIS function as a Callable instead of keeping
# a copy of it. With an untouched XROrigin3D, no recentering and world_scale 1 all three factors are
# identity, so this costs nothing today and stops the poses drifting off the moment locomotion or
# scaling appears.
# Stays here rather than moving with the checks: the LIVE locate path in _on_android_camera_frame
# converts every frame's pose with it.
func _play_space_to_world(head: Transform3D) -> Transform3D:
	var scaled := Transform3D(head.basis, head.origin * XRServer.world_scale)
	return xr_origin.global_transform * XRServer.get_reference_frame() * scaled


func _exit_tree() -> void:
	if android_camera != null and _android_cam_started:
		android_camera.stop_camera()
	# A detection task may still be running on the pool; block until it is done so its bound
	# callable (which captures self) doesn't outlive the scene. Costs at most one detection (~80ms).
	# Nothing else to tidy: the hand-eye file went with the DetectionDiagnostics node, which closes
	# it from its own _exit_tree.
	if _detect_task_id != -1:
		WorkerThreadPool.wait_for_task_completion(_detect_task_id)
		_detect_task_id = -1

#command to stop (once adb is added to PATH)
# adb shell am force-stop de.unigreifswald.opencvaruco
