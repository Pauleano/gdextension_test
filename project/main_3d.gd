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
# side length (see _get_or_create_patch), so the two can never disagree.
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

# Single switch for ALL debug output, this script's and the C++ extension's. Off by default: with
# no tcp_receiver.py listening the TCP reconnect logs alone would print once per second for the
# whole session, and the detection timing below fires ~12x/s on the Quest. Errors are never gated
# and print regardless.
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
@onready var cam_preview: TextureRect = $CameraLayer/CameraPreview
@onready var xr_origin: XROrigin3D = $XROrigin3D
@onready var xr_camera: XRCamera3D = $XROrigin3D/XRCamera3D

# --- Detection worker (Teil B) ---
# On the Quest the OpenCV detection costs ~80ms, which run synchronously would cap the whole
# app at ~10fps. We run ONLY the detection (detectMarkers + solvePnP) off the main thread, as
# one-shot WorkerThreadPool tasks -- Godot owns the threads, so there is no Thread/Mutex/
# Semaphore lifecycle to manage here. At most ONE task is in flight at a time; get_image()
# and all scene-tree writes stay on the main thread (neither is thread-safe).
# There is no pending-frame slot on this branch: frames are PULLED here, so instead of parking a
# frame while the worker is busy we simply do not read one back (see _process).
var _detect_task_id := -1              # WorkerThreadPool task id; -1 = no task in flight
# output slot, written by the task; the main thread reads it only AFTER
# wait_for_task_completion(), which is the synchronization point (no lock needed)
var _result_markers: Dictionary = {}

# --- Capture-latency compensation ---
# The Image get_image() returns is OLDER than "now": sensor -> ISP -> CameraServer texture takes
# 1-2 camera frames. Pairing those old pixels with the LIVE head pose bakes an error proportional
# to head speed, so a fresh detection first "drags" with the head, then settles. We keep a short
# timestamped head-pose history and sample the head pose from CAMERA_LATENCY_MS ago instead; that
# pose travels with the frame, and the C++ side bakes the markers straight to world space with it.
# This readback path has NO sensor timestamp, so this fixed guess is the only correction available
# -- tune it on device: patch still drags WITH the head -> raise; patch lags behind the real marker
# during motion -> lower.
@export_range(0, 300, 1.0, "or_greater", "suffix:ms") var CAMERA_LATENCY_MS := 90.0

var _xr_cam_pose_history: Array = []   # [t_usec, head Transform3D] pairs, newest last; main thread only


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
	# demand once frames arrive (see _process) -- long after _ready has finished.

	if OS.get_name() == "Android":
		# Quest: request camera access; the native CameraServer surfaces feeds once granted.
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

####################################################################################################

func _process(_delta: float) -> void:
	_poll_tcp(_delta)

	# (a) Sample the head pose EVERY render frame, before any early return below: _head_pose_at
	# interpolates between the two nearest samples, so its accuracy is bounded by the sampling
	# period. Recording only on detection frames would coarsen the history from ~14ms to ~80ms and
	# put most of the capture-latency compensation back as error.
	var now_usec := Time.get_ticks_usec()
	_xr_cam_pose_history.append([now_usec, xr_camera.global_transform])
	while _xr_cam_pose_history.size() > 1 and _xr_cam_pose_history[0][0] < now_usec - 500_000:
		_xr_cam_pose_history.pop_front()

	# (b) Collect the latest finished detection and bake it into the scene (main thread ->
	# scene-tree writes are safe here).
	_poll_detection_task()

	if cam_texture == null:
		return

	# (c) Hand the newest camera frame to a detection task. get_image() (the GPU->CPU readback) and
	# the head-pose lookup must happen on the main thread; the task only does the OpenCV work.
	#
	# Readback ONLY when no detection is running. Detection costs ~80ms while _process runs at the
	# render rate (~72fps on Quest), so an unconditional get_image() paid the full readback ~6x per
	# detection and threw all but the last one away. The readback is a GPU->CPU stall on the main
	# thread, i.e. render-frame time burned for nothing. Skipping it while a task runs also means
	# the frame we DO read back is the freshest one at the instant detection starts, which shortens
	# the pose-history lookback. This is also why this branch needs no pending-frame slot: frames
	# are pulled here, so declining to pull IS the frame drop.
	if _detect_task_id != -1:
		return

	var readback_t0 := Time.get_ticks_usec()
	var img := cam_texture.get_image()
	if img == null:
		return
	# format lookup table https://docs.godotengine.org/en/stable/classes/class_image.html#enum-image-format
	if debug_prints_enabled:
		print("[opencv_aruco] [main_3d::_process] readback_ms=%.2f image_format=%d" % [(Time.get_ticks_usec() - readback_t0) / 1000.0, img.get_format()])

	_tcp_send_timer += _delta
	if _tcp_send_timer >= TCP_SEND_INTERVAL:
		_tcp_send_timer = 0.0
		if stream_peer != null and stream_peer.get_status() == StreamPeerTCP.STATUS_CONNECTED:
			_send_frame_tcp(img)

	# Head pose AT capture time: NOT the live pose -- the pixels in img are ~CAMERA_LATENCY_MS old
	# (passthrough pipeline), so look that far back in the history filled in (a). The pose travels
	# with the frame and the C++ side applies it together with the lens pose, so the markers come
	# back in world space.
	var capture_usec := now_usec - int(CAMERA_LATENCY_MS * 1000.0)
	_start_detection_task(img, _head_pose_at(capture_usec))

# Main thread only. If the in-flight task has finished: clean it up and apply its result.
func _poll_detection_task() -> void:
	if _detect_task_id == -1 or not WorkerThreadPool.is_task_completed(_detect_task_id):
		return
	# Mandatory cleanup of every finished task; returns immediately here (the task is done) and
	# doubles as the memory barrier that makes the task's _result_markers write visible to us.
	WorkerThreadPool.wait_for_task_completion(_detect_task_id)
	_detect_task_id = -1
	_apply_detection_result()

func _start_detection_task(img: Image, cam_xform: Transform3D) -> void:
	_detect_task_id = WorkerThreadPool.add_task(_detect_frame.bind(img, cam_xform),
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
	var now_usec := Time.get_ticks_usec()
	for id in markers:
		# markers[id] is already WORLD space; freeze it there, so the head can move between
		# detections without dragging the patch along.
		_get_or_create_patch(id).global_transform = markers[id]
		_marker_last_seen[id] = now_usec

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
# the main thread. Touches only `processor`, read-only config and the _result_markers slot -- never
# the scene tree. Writing _result_markers without a lock is safe: the main thread reads the slot
# only after wait_for_task_completion() on this task.
func _detect_frame(img: Image, cam_xform: Transform3D) -> void:
	# No conversion: the C++ side handles 1ch (Quest Y-plane), 3ch (RGB), and 4ch (RGBA).
	var t0 := Time.get_ticks_usec()
	# The exported intrinsics are for the native 640x480 frame; all four components scale with
	# the image, so image_downscale_factor is applied at use time.
	var intrinsics := camera_intrinsics * image_downscale_factor
	# The two transforms that hold for EVERY marker -- head pose at capture time and physical
	# lens offset -- combined into ONE camera->world pose. The C++ side pre-multiplies it onto
	# each solvePnP pose, so the returned Dictionary is already in WORLD space.
	var cam_to_world := cam_xform * _lens_pose
	var markers: Dictionary = processor.get_6dof_of_all_aruco_patches_from_godot_image(img, _marker_size_table, aruco_patch_size, image_downscale_factor, intrinsics, camera_distortion, cam_to_world)
	# Guarded inline rather than via a helper function: a helper would build this string on every
	# detection (~12x/s on the Quest) before it could check the flag.
	if debug_prints_enabled:
		var detect_ms := (Time.get_ticks_usec() - t0) / 1000.0
		var tracking_fps := 1000.0 / detect_ms if detect_ms > 0.0 else 0.0
		print("[opencv_aruco] [main_3d::_detect_frame] detect_ms=%.1f tracking_fps=%.1f render_fps=%d markers=%d" % [
				detect_ms, tracking_fps, Engine.get_frames_per_second(), markers.size()])
	_result_markers = markers


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
