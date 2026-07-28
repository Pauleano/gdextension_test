extends Node3D

var processor: OpenCVProcessor
# id -> patch Node3D, the ONE registry of live patches. Nodes are created on a marker's first
# detection and freed once it has been missing for PATCH_LOST_TIMEOUT_MS; the scene file holds
# none of them. Because an id is only ever added through _get_or_create_patch (which checks this
# dictionary first) and only ever removed together with its node, one id can never own two nodes
# nor swap onto another id's node. Main thread only -- detection tasks never touch it.
var marker_nodes: Dictionary = {}
# id -> Time.get_ticks_usec() of the last detection result that contained it; drives the deletion
# grace period. Same lifetime as marker_nodes, kept in step with it.
var _marker_last_seen: Dictionary = {}
# id -> resolved size in meters for the C++ side; built once in _ready and read-only afterwards
# (detection tasks read it without a lock -- safe because no task is dispatched before _ready ends).
var _marker_size_table: Dictionary = {}

# Fallback physical side length, in meters, for every marker id WITHOUT a table entry below.
# Used as the solvePnP marker size (sets the pose's metric scale) AND as the rendered cuboid's
# side length (see _ready), so the two can never disagree.
@export_range(0.01, 0.3, 0.001, "or_greater", "suffix:m") var aruco_patch_size := 0.1

# Ground-truth lookup table: index = marker id (0-9), value = physical side length in meters.
# 0 = unset -> that id falls back to aruco_patch_size (as do all ids >= 10 of DICT_4X4_50), so
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

# --- Camera calibration (left Quest passthrough camera "50", native 640x480 frame) ---
# Exported so they can be tuned in the inspector instead of hunting through code. Read-only
# after _ready: detection tasks read them without a lock (same pattern as _marker_size_table),
# so treat inspector edits as pre-run configuration, not live tuning.
# Intrinsics (fx, fy, cx, cy) in pixels for the NATIVE 640x480 frame. _detect_frame scales all
# four with the downscale factor at use time -- never bake that factor into these values.
# (approximations: cx ~ width/2, cy ~ height/2)
@export var camera_intrinsics := Vector4(435.37335635, 435.96983202, 320.84589009, 241.55014114)
# OpenCV distCoeffs (k1, k2, p1, p2, k3) for the Quest passthrough lens; an EMPTY array means
# "no distortion". These used to be hardcoded in the C++ side.
@export var camera_distortion: PackedFloat64Array = [-0.00484306, 0.14036606, 0.00044449, -0.00108918, -0.29608385]
# Detection resolution knob: 1.0 = native frame, 0.5 = half width AND half height, i.e. a quarter
# of the pixels -> markedly cheaper detection, at the price of small or distant markers dropping
# below the resolution the detector needs. _detect_frame hands it to the C++ side AND scales the
# intrinsics above by the same factor; the two must always move together, which is why the factor
# belongs here and never baked into camera_intrinsics.
@export_range(0.1, 1.0, 0.05) var image_downscale_factor := 1.0
# Physical passthrough-camera pose relative to the gyro/IMU reference, RAW from the Quest's
# ACAMERA_LENS_POSE_ROTATION / _TRANSLATION (LENS_POSE_REFERENCE == GYROSCOPE).
# The raw quaternion is ~168.8deg about X = the Android sensor->camera-optical 180deg X-flip
# PLUS the camera's real ~11deg pitch. The C++ marker pose already contains that same 180deg
# flip (its negate-Y/Z change of basis), so _ready multiplies by Quaternion(1,0,0,0) (=180deg
# about X) to cancel the flip and keep ONLY the physical mounting tilt (-> _lens_pose).
# The translation is in the sensor frame (X right, Y up, Z toward viewer), which matches Godot
# camera axes -> raw values, no sign flips. If markers land in the wrong place, the axis
# convention is the knob: try the conjugate quaternion / flipped translation signs.
@export var lens_rotation_raw := Quaternion(-0.99519097805023, 0.00269138417207, 0.00294101587497, 0.09787271916866)
@export var lens_translation := Vector3(-0.03237725794315, -0.01770938560367, -0.06345107406378)
# Derived ONCE from the exported raw values in _ready (before any detection task can exist).
var _lens_pose := Transform3D.IDENTITY


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
var _cam_frame_count := 0
var _cam_fallback_count := 0           # frames whose timestamp failed the plausibility guard
var _preview_texture: ImageTexture     # debug preview fed from plugin frames (desktop uses cam_texture)
var _xr_api: OpenXRAPIExtension        # access to xrWaitFrame's predicted display time (XrTime)
var _lead_print_timer := 0.0
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

# Residual timing bias of the pose lookup on the Android path, in ms; positive = use an OLDER
# head pose. The sensor timestamp removes the VARIABLE pipeline delay, but two small static
# offsets remain unknowable from GDScript: the head poses Godot reports are OpenXR poses
# PREDICTED for the upcoming display time (~1-3 frame periods ahead of _process), and the clock
# base of an "unknown" timestamp_source is only typically monotonic. Tune on device like
# CAMERA_LATENCY_MS before: patch drags WITH the head during motion -> raise; patch lags
# BEHIND the real marker -> lower (may go negative).
@export_range(10,100,0.1,"or_greater","suffix:ms") var POSE_LOOKUP_BIAS_MS := 35.0

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

#######################################################################################################

# Single source of truth for a marker id's physical size: table entry if present and set,
# aruco_patch_size otherwise. The bounds check doubles as the guard for arrays the inspector
# resized to fewer/more than 10 elements.
func _marker_size_for(id: int) -> float:
	if id >= 0 and id < aruco_patch_sizes.size():
		var s: float = aruco_patch_sizes[id]
		if s > 0.0:
			return s
	return aruco_patch_size


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

	# Resolve the size table once for the C++ side (entries are pre-resolved through
	# _marker_size_for, so the C++ default only fires for ids >= the table length).
	for id in aruco_patch_sizes.size():
		_marker_size_table[id] = _marker_size_for(id)

	# No worker setup needed: detection runs as one-shot WorkerThreadPool tasks, dispatched on
	# demand once frames arrive (see _dispatch_detection) -- long after _ready has finished.

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
	_cam_clock_offset_ns = android_camera.get_clock_offset_nanos() if _cam_ts_realtime \
			else android_camera.get_monotonic_clock_offset_nanos()
	if debug_prints_enabled:
		print("[opencv_aruco] [main_3d::_start_android_camera] starting camera: id=%s clock=%s feeds=%s" % [
				cam_id, "realtime" if _cam_ts_realtime else "monotonic", cameras])
		# Every frame's exposure time is converted with: cap_usec = (timestamp_ns - offset_ns) / 1000
		print("[opencv_aruco] [main_3d::_start_android_camera] flow setup: clock bridge calibrated, camera clock is ahead of Time.get_ticks_usec() by offset_s=%.3f offset_ns=%d" % [
				_cam_clock_offset_ns / 1.0e9, _cam_clock_offset_ns])
	# 640x480 matches the hardcoded intrinsics; LUMA is the camera's native Y plane, which is
	# exactly the 1-channel grayscale the C++ detector consumes -- no conversion anywhere.
	android_camera.start_camera(640, 480, false, cam_id, 0, 0, AndroidCamera.OutputFormat.LUMA)

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
		_cam_clock_offset_ns = android_camera.get_clock_offset_nanos() if _cam_ts_realtime \
				else android_camera.get_monotonic_clock_offset_nanos()
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

	cap_usec -= int(POSE_LOOKUP_BIAS_MS * 1000.0)
	if traced:
		# (3) the stored head poses are predicted ~POSE_LOOKUP_BIAS_MS into the future, so the
		# lookup target is shifted back by the same amount to line the two timelines up.
		print("[opencv_aruco] [main_3d::_on_android_camera_frame] flow #%d (3) pose lookup target: target_usec=%d bias_ms=%.1f (compensates OpenXR predicting head poses ahead of time)" % [
				frame_id, cap_usec, POSE_LOOKUP_BIAS_MS])

	# Head pose at the frame's capture time, sampled at ARRIVAL. The lookup is history-based, so
	# sampling here or after the detection gives the same pose -- but sampled here it can travel
	# with the frame, and the C++ side applies it together with the lens pose (markers come back
	# in world space).
	var cam_xform := _head_pose_at(cap_usec)
	if traced:
		print("[opencv_aruco] [main_3d::_on_android_camera_frame] flow #%d (4) head pose at capture: pos=%v (travels with the frame)" % [
				frame_id, cam_xform.origin])

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
	var now_usec := Time.get_ticks_usec()
	_xr_cam_pose_history.append([now_usec, xr_camera.global_transform])
	while _xr_cam_pose_history.size() > 1 and _xr_cam_pose_history[0][0] < now_usec - 500_000:
		_xr_cam_pose_history.pop_front()

	# Measure the OpenXR pose-prediction lead: xrWaitFrame's predicted display time (XrTime =
	# CLOCK_MONOTONIC ns on the Quest) minus "now" on that same clock (the camera clock offset
	# maps Godot ticks -> monotonic ns). This is how far in the FUTURE the pose stored above
	# actually is -- the measured value to use for POSE_LOOKUP_BIAS_MS instead of guessing.
	# Only on the Android plugin path with a monotonic offset; skipped entirely on desktop.
	# Purely diagnostic, so it is gated as a whole: with debug off we neither allocate the
	# OpenXRAPIExtension nor query it.
	if debug_prints_enabled and _android_cam_started and not _cam_ts_realtime:
		_lead_print_timer += _delta
		if _lead_print_timer >= 1.0:
			_lead_print_timer = 0.0
			if _xr_api == null:
				_xr_api = OpenXRAPIExtension.new()
			var pdt := _xr_api.get_predicted_display_time()
			if pdt != 0:
				var lead_ms := float(pdt - (now_usec * 1000 + _cam_clock_offset_ns)) / 1.0e6
				print("[opencv_aruco] [main_3d::_process] openxr pose prediction: lead_ms=%.2f" % lead_ms)

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
func _apply_detection_result() -> void:
	var markers: Dictionary = _result_markers
	var result_id: int = _result_frame_id
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
	for id in markers:
		# markers[id] is already WORLD space; freeze it there, so the head can move between
		# detections without dragging the patch along.
		_get_or_create_patch(id).global_transform = markers[id]
		_marker_last_seen[id] = now_usec
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
		_marker_last_seen.erase(id)
		if debug_prints_enabled:
			print("[opencv_aruco] [main_3d::_apply_detection_result] patch deleted: id=%d unseen_ms=%.0f patches=%d" % [
					id, unseen_usec / 1000.0, marker_nodes.size()])

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
# one sample per rendered frame, ~14ms at 72fps; interpolating removes that quantisation).
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



func _exit_tree() -> void:
	if android_camera != null and _android_cam_started:
		android_camera.stop_camera()
	# A detection task may still be running on the pool; block until it is done so its bound
	# callable (which captures self) doesn't outlive the scene. Costs at most one detection (~80ms).
	if _detect_task_id != -1:
		WorkerThreadPool.wait_for_task_completion(_detect_task_id)
		_detect_task_id = -1

func _send_frame_tcp(img: Image) -> void:
	if stream_peer == null:
		return

	stream_peer.poll()

	if stream_peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
		return

	var bytes: PackedByteArray = img.get_data()

	stream_peer.put_u32(img.get_width())
	stream_peer.put_u32(img.get_height())
	stream_peer.put_u32(img.get_format())
	stream_peer.put_u32(bytes.size())

	var err := stream_peer.put_data(bytes)
	if err != OK:
		push_error("[opencv_aruco] [main_3d::_send_frame_tcp] put_data failed: err=%d" % err)

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
		return

	if status == StreamPeerTCP.STATUS_CONNECTING:
		return

	if status == StreamPeerTCP.STATUS_ERROR or status == StreamPeerTCP.STATUS_NONE:
		_tcp_reconnect_timer += delta
		if _tcp_reconnect_timer >= 1.0:
			_tcp_reconnect_timer = 0.0
			if debug_prints_enabled:
				print("[opencv_aruco] [main_3d::_poll_tcp] reconnecting")
			_connect_tcp()
#command to stop (once adb is added to PATH)
# adb shell am force-stop de.unigreifswald.opencvaruco
