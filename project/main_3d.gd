extends Node3D

var processor: OpenCVProcessor
var marker_nodes: Dictionary = {}

# Physical side length of the ArUco marker, in meters. Single source of truth: used as the
# solvePnP marker size (sets the pose's metric scale) AND as the rendered cuboid's side length
# (see _ready), so the two can never disagree.
@export_range(0.01, 0.3, 0.001, "or_greater", "suffix:m") var aruco_patch_size := 0.1
@export  var lens_q_raw := Quaternion(-0.99519097805023, 0.00269138417207, 0.00294101587497, 0.09787271916866)

# Single switch for ALL debug output, this script's and the C++ extension's -- _ready pushes it into
# OpenCVProcessor before instantiating it. Off by default: with no tcp_receiver.py listening the TCP
# reconnect logs alone would print once per second for the whole session, and the detection timing
# below fires ~12x/s on the Quest. Errors are never gated and print regardless.
# Format for every debug line: "[opencv_aruco] [main_3d::function] event: key=value" -- the fixed
# prefix is what makes them findable in logcat (adb logcat | grep opencv_aruco).
@export var debug_prints_enabled := false

# Godot has a native CameraServer (Camera2) backend on Android since 4.5, so
# CameraServerExtension is only needed on desktop (Windows). Keep this var UNTYPED and
# instantiate via ClassDB so the script still parses on Android, where the
# CameraServerExtension class isn't registered.
var camera_extension
var cam_texture: CameraTexture
@onready var cam_preview: TextureRect = $CameraLayer/CameraPreview
@onready var xr_origin: XROrigin3D = $XROrigin3D
@onready var xr_camera: XRCamera3D = $XROrigin3D/XRCamera3D

# --- Detection worker thread (Teil B) ---
# On the Quest the OpenCV detection costs ~80ms, which run synchronously would cap the whole
# app at ~10fps. We move ONLY the detection (detectMarkers + solvePnP) onto a background thread;
# get_image() and all scene-tree writes stay on the main thread (neither is thread-safe).
var _detect_thread: Thread
var _detect_mutex: Mutex
var _detect_sem: Semaphore
var _detect_exit := false
# input slot (main -> worker), guarded by _detect_mutex; only the newest frame is kept (frame drop)
var _pending_image: Image
var _pending_cam_xform: Transform3D
var _has_pending := false
# true from the moment the worker takes a frame until it parks the result; together with
# _has_pending this is "the worker cannot use another frame right now", which is what gates
# the readback in _process (see there).
var _worker_busy := false
# output slot (worker -> main), guarded by _detect_mutex
var _result_markers: Dictionary = {}
var _result_cam_xform: Transform3D
var _worker_has_result := false

# --- Capture-latency compensation ---
# The Image get_image() returns is OLDER than "now": sensor -> ISP -> CameraServer texture takes
# 1-2 camera frames (~30-100ms). Pairing those old pixels with the LIVE head pose bakes an error
# proportional to head speed, so a fresh detection first "drags" with the head, then settles.
# We keep a short timestamped head-pose history and pair each frame with the pose from
# CAMERA_LATENCY_MS ago instead. Tune on device: patch still drags WITH the head -> raise;
# patch lags behind the real marker during motion -> lower.

const CAMERA_LATENCY_MS := 90.0 #estimated correction variable to match head_pose and marker_pose (relative to real_life_camera_pose) in time

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

func _ready() -> void:
	# Static setter, deliberately called BEFORE new(): the C++ constructor already prints (OpenCV
	# build info + the Quest intrinsics dump), so an instance property would be set one step too late.
	OpenCVProcessor.set_debug_prints_enabled(debug_prints_enabled)
	processor = OpenCVProcessor.new()

	# detect all available aruco_patch nodes, to later set their position
	for child in xr_origin.get_children():
		var n: String = child.name
		if n.begins_with("aruco_patch"):
			var id_str: String = n.substr(11)
			if id_str.is_valid_int():
				marker_nodes[id_str.to_int()] = child
	
	# Reparent the patches out of the camera and into the (stationary) tracking origin.
	# solvePnP gives a CAMERA-relative pose; as a child of XRCamera3D that pose would be
	# re-multiplied by the LIVE head transform every frame, so a stale detection rides the
	# head and "swims" when you move. Anchored in XROrigin3D world space instead, we bake the
	# pose once at detection time (see _process) and it stays fixed on the real marker.
	for id in marker_nodes:
		var patch: Node3D = marker_nodes[id]
		# Size the rendered cuboid to the real marker: x/y = aruco_patch_size; keep the authored
		# thickness (z). The BoxMesh is a unit cube and the baked pose is rigid (scale 1), so the
		# mesh's local scale IS its size in meters.
		for c in patch.get_children():
			if c is MeshInstance3D:
				c.scale = Vector3(aruco_patch_size, aruco_patch_size, c.scale.z)

	# Spin up the detection worker. processor and marker_nodes are ready by now; the loop only
	# needs the processor (no scene-tree access), so it is safe to start before any feed exists.
	_detect_mutex = Mutex.new()
	_detect_sem = Semaphore.new()
	_detect_thread = Thread.new()
	_detect_thread.start(_detection_loop)

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
	
	if cam_texture == null:
		return

	# (a) Apply the latest finished detection (main thread -> scene-tree writes are safe here).
	# markers + cam_xform come from the worker; cam_xform was sampled at the frame's CAPTURE time
	# (not now), so the world bake matches the frame the markers were detected in -- using the
	# live head pose here would re-introduce the ~80ms of head motion as swim.
	_detect_mutex.lock()
	var have_result := _worker_has_result
	var markers: Dictionary = _result_markers
	var cam_xform: Transform3D = _result_cam_xform
	_worker_has_result = false
	_detect_mutex.unlock()
	if have_result:
		for id in markers:
			if marker_nodes.has(id):
				# markers[id] is the marker pose in CAMERA space; cam_xform is the head pose at
				# capture time. Bake to world space and freeze it there, so the head can move
				# between detections without dragging the patch along.
				marker_nodes[id].global_transform = cam_xform * markers[id]

	# (b) Sample the head pose EVERY render frame, independently of whether we read back below:
	# _head_pose_at interpolates nothing, it picks the nearest-older sample, so its accuracy is the
	# sampling period. Recording only on detection frames would coarsen the history from ~14ms to
	# ~80ms and put most of the capture-latency compensation back as error.
	var now_usec := Time.get_ticks_usec()
	_xr_cam_pose_history.append([now_usec, xr_camera.global_transform])
	while _xr_cam_pose_history.size() > 1 and _xr_cam_pose_history[0][0] < now_usec - 500_000:
		_xr_cam_pose_history.pop_front()

	# (c) Hand the newest camera frame to the worker. get_image() (the GPU->CPU readback) and the
	# head-pose lookup must happen on the main thread; the worker only does the OpenCV work.
	#
	# Readback ONLY when the worker can actually consume the frame. Detection costs ~80ms while
	# _process runs at the render rate (~72fps on Quest), so an unconditional get_image() paid the
	# full readback ~6x per detection and threw all but the last one away in the frame-drop slot.
	# The readback is a GPU->CPU stall on the main thread, i.e. render-frame time we were burning
	# for nothing. Skipping it while the worker is busy also means the frame we DO read back is the
	# freshest one at the instant detection starts, which shortens the pose-history lookback.
	_detect_mutex.lock()
	var worker_can_take := not _has_pending and not _worker_busy
	_detect_mutex.unlock()
	if not worker_can_take:
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
	# (passthrough pipeline), so look that far back in the pose history (filled in (b) above).
	var cap_cam_xform := _head_pose_at(now_usec - int(CAMERA_LATENCY_MS * 1000.0))

	_detect_mutex.lock()
	_pending_image = img
	_pending_cam_xform = cap_cam_xform
	_has_pending = true                                 # was false: checked above, and only we set it
	_detect_mutex.unlock()
	_detect_sem.post()                                  # exactly one post per pending frame

# Worker thread: blocks on the semaphore, runs OpenCV detection off the main thread, and parks the
# result for _process to apply. Touches only `processor` and value types -- never the scene tree.
func _detection_loop() -> void:
	while true:
		_detect_sem.wait()
		if _detect_exit:
			return
		_detect_mutex.lock()
		var img: Image = _pending_image
		var cam_xform: Transform3D = _pending_cam_xform
		_has_pending = false
		_pending_image = null
		_worker_busy = img != null      # cleared again once the result is parked, below
		_detect_mutex.unlock()
		if img == null:
			continue
		# No conversion: the C++ side handles 1ch (Quest Y-plane), 3ch (RGB), and 4ch (RGBA).
		var t0 := Time.get_ticks_usec()
		var image_downscale_factor=1 #1 is original image, 0.5 means half width and half height
		var fx=435.37335635*image_downscale_factor
		var fy=435.96983202*image_downscale_factor
		var cx=320.84589009*image_downscale_factor #approxiamte cx is fx/2
		var cy=241.55014114*image_downscale_factor #approxiamte cy is fy/2
		# Physical passthrough-camera pose relative to the gyro/IMU reference, from the Quest's
		# ACAMERA_LENS_POSE_ROTATION / _TRANSLATION (LENS_POSE_REFERENCE == GYROSCOPE).
		# The raw quaternion is ~168.8deg about X = the Android sensor->camera-optical 180deg X-flip
		# PLUS the camera's real ~11deg pitch. The C++ marker pose already contains that same 180deg
		# flip (its negate-Y/Z change of basis), so multiply by Quaternion(1,0,0,0) (=180deg about X)
		# to cancel the flip and keep ONLY the physical mounting tilt.
		# The translation is in the sensor frame (X right, Y up, Z toward viewer), which matches Godot
		# camera axes -> use raw values, no sign flips.
		
		var lens_rotation := (lens_q_raw * Quaternion(1, 0, 0, 0)).inverse() # if visibly worse, try appending .inverse()
		var lens_translation := Vector3(-0.03237725794315, -0.01770938560367, -0.06345107406378) # raw LENS_POSE_TRANSLATION

		var intrinsics := Vector4(fx, fy, cx, cy)
		# OpenCV distCoeffs (k1, k2, p1, p2, k3) for the Quest passthrough lens; pass an empty
		# PackedFloat64Array() for no distortion. These used to be hardcoded in the C++ side.
		var distortion := PackedFloat64Array([-0.00484306,  0.14036606,  0.00044449, -0.00108918, -0.29608385])
		# Combine the lens rotation + translation into one rigid pose (matches the C++ lens_pose param).
		var lens_pose := Transform3D(Basis(lens_rotation), lens_translation)
		var markers: Dictionary = processor.get_6dof_of_all_aruco_patches_from_godot_image(img, aruco_patch_size, image_downscale_factor, intrinsics, distortion, lens_pose)
		# Guarded inline rather than via a helper function: a helper would build this string on every
		# detection (~12x/s on the Quest) before it could check the flag.
		if debug_prints_enabled:
			var detect_ms := (Time.get_ticks_usec() - t0) / 1000.0
			var tracking_fps := 1000.0 / detect_ms if detect_ms > 0.0 else 0.0
			print("[opencv_aruco] [main_3d::_detection_loop] detect_ms=%.1f tracking_fps=%.1f render_fps=%d markers=%d" % [detect_ms, tracking_fps, Engine.get_frames_per_second(), markers.size()])
		_detect_mutex.lock()
		_result_markers = markers
		_result_cam_xform = cam_xform
		_worker_has_result = true
		_worker_busy = false            # _process may read back the next frame from here on
		_detect_mutex.unlock()


# Newest recorded head pose that is not newer than t_usec (nearest-older sample; at 72fps the
# history has ~14ms granularity). Falls back to the oldest entry (or the live pose) while the
# history is still filling up.
func _head_pose_at(t_usec: int) -> Transform3D:
	for i in range(_xr_cam_pose_history.size() - 1, -1, -1):
		if _xr_cam_pose_history[i][0] <= t_usec:
			return _xr_cam_pose_history[i][1]
	if _xr_cam_pose_history.is_empty():
		return xr_camera.global_transform
	return _xr_cam_pose_history[0][1]



func _exit_tree() -> void:
	# Wake the worker out of its wait() and join it, so the thread doesn't outlive the scene.
	if _detect_thread != null and _detect_thread.is_started():
		_detect_mutex.lock()
		_detect_exit = true  #ends the detection_loop
		_detect_mutex.unlock()
		_detect_sem.post()
		_detect_thread.wait_to_finish()

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
