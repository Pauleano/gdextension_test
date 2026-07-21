extends Node3D

var processor: OpenCVProcessor
var marker_nodes: Dictionary = {}

# Physical side length of the ArUco marker, in meters. Single source of truth: used as the
# solvePnP marker size (sets the pose's metric scale) AND as the rendered cuboid's side length
# (see _ready), so the two can never disagree.
@export_range(0.01, 0.3, 0.001, "or_greater", "suffix:m") var aruco_patch_size := 0.1


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
var _pending_capture_usec := 0
var _has_pending := false
# output slot (worker -> main), guarded by _detect_mutex
var _result_markers: Dictionary = {}
var _result_capture_usec := 0
var _worker_has_result := false

# --- Capture-latency compensation ---
# A frame's pixels are OLDER than "now" when they reach us (sensor -> ISP -> delivery takes
# 1-2 camera frames), so pairing them with the LIVE head pose bakes an error proportional to
# head speed. We keep a short timestamped head-pose history and bake each detection with the
# head pose from the frame's CAPTURE time instead (see _head_pose_at).
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
const POSE_LOOKUP_BIAS_MS := 35.0

var _xr_cam_pose_history: Array = []   # [t_usec, head Transform3D] pairs, newest last; main thread only

# --- Debug: step-by-step flow tracing (learning aid, costs performance) ---
# Every DEBUG_FLOW_EVERY-th camera frame is "traced": each station of the pipeline prints one
# line tagged with that frame's id, so ONE frame can be followed end to end:
#   (1) arrival -> (2) timestamp -> (3) bias -> (4) handoff -> (5) worker -> (6) detection
#   -> (7) pose lookup -> (8) what the timing correction was worth -> (9) world bake
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

func _ready() -> void:
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
	for i in range(feed_count):
		var f := CameraServer.get_feed(i)
		print("camera feed index ", i, " id ", f.get_id(), " name ", f.get_name())

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
	for j in range(formats.size()):
		print("feed format ", j, ": ", formats[j])
	if formats.size() > 2:
		feed.set_format(2, {})        # feed format:10 1280x1280 YUV_420_888 (for now, choice can be altered) (good ArUco res, light on CPU)
	elif formats.size() > 0:
		feed.set_format(0, {})

	feed.set_active(true)                               # start delivering frames
	cam_texture = CameraTexture.new()
	cam_texture.camera_feed_id = feed.get_id()
	cam_texture.which_feed = CameraServer.FEED_RGBA_IMAGE
	cam_preview.texture = cam_texture
	print("camera feeds: ", feed_count)

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
	# _detection_loop belong to it. Fall back to any world-facing feed on non-Quest devices.
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
	print("AndroidCamera feeds: ", cameras, " -> starting id '", cam_id,
			"' (timestamp clock: ", "realtime" if _cam_ts_realtime else "monotonic", ")")
	if DEBUG_FLOW:
		print("[flow setup] clock bridge calibrated: camera clock is ",
				"%.3f" % (_cam_clock_offset_ns / 1.0e9), "s AHEAD of Godot's Time.get_ticks_usec().",
				"  Every frame's exposure time is converted with: cap_usec = (timestamp_ns - ",
				_cam_clock_offset_ns, ") / 1000")
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
		print("\n[flow #", frame_id, "] (1) FRAME ARRIVES: ", width, "x", height,
				", ", data.size(), " bytes (grayscale Y-plane, 1 byte/pixel)",
				", sensor timestamp_ns=", timestamp_ns)
		print("[flow #", frame_id, "] (2) EXPOSURE TIME on Godot clock: cap_usec=", cap_usec,
				" -> these pixels are ", "%.1f" % lag_ms, "ms old",
				"  [", "FALLBACK GUESS, timestamp rejected!" if used_fallback else "from the real sensor timestamp", "]")

	cap_usec -= int(POSE_LOOKUP_BIAS_MS * 1000.0)
	if traced:
		# (3) the stored head poses are predicted ~POSE_LOOKUP_BIAS_MS into the future, so the
		# lookup target is shifted back by the same amount to line the two timelines up.
		print("[flow #", frame_id, "] (3) POSE LOOKUP TARGET = exposure - POSE_LOOKUP_BIAS_MS(",
				POSE_LOOKUP_BIAS_MS, "ms) = ", cap_usec,
				"   (compensates OpenXR predicting head poses ahead of time)")

	if _cam_frame_count == 1 or _cam_frame_count % 300 == 0:
		print("AndroidCamera frame ", _cam_frame_count, ": ", width, "x", height,
				" sensor->delivery lag=", lag_ms, "ms",
				" clock=", "realtime" if _cam_ts_realtime else "monotonic",
				" fallbacks=", _cam_fallback_count)

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

	_dispatch_detection(img, cap_usec, frame_id)

####################################################################################################

# True if this frame's journey should be traced. The SAME id gives the SAME answer at every
# station, so all prints belonging to one frame appear together (see DEBUG_FLOW).
func _tracing(frame_id: int) -> bool:
	return DEBUG_FLOW and frame_id % DEBUG_FLOW_EVERY == 0
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
	if _android_cam_started and not _cam_ts_realtime:
		_lead_print_timer += _delta
		if _lead_print_timer >= 1.0:
			_lead_print_timer = 0.0
			if _xr_api == null:
				_xr_api = OpenXRAPIExtension.new()
			var pdt := _xr_api.get_predicted_display_time()
			if pdt != 0:
				var lead_ms := float(pdt - (now_usec * 1000 + _cam_clock_offset_ns)) / 1.0e6
				print("OpenXR pose-prediction lead_ms=", lead_ms)

	# (a) Apply the latest finished detection (main thread -> scene-tree writes are safe here).
	# The markers were detected in a frame captured at _result_capture_usec; look up the head pose
	# from THAT moment -- using the live pose would re-introduce the pipeline latency as swim.
	_detect_mutex.lock()
	var have_result := _worker_has_result
	var markers: Dictionary = _result_markers
	var capture_usec: int = _result_capture_usec
	var result_id: int = _result_frame_id
	_worker_has_result = false
	_detect_mutex.unlock()
	if have_result:
		var cam_xform := _head_pose_at(capture_usec)
		if _tracing(result_id):
			# (7) what the history is asked, (8) what the whole correction was worth: the head
			# pose used vs. the LIVE one -- that difference is exactly the swim we avoid.
			var live_xform := xr_camera.global_transform
			var drift_cm := live_xform.origin.distance_to(cam_xform.origin) * 100.0
			var turn_deg := rad_to_deg(cam_xform.basis.get_rotation_quaternion().angle_to(
					live_xform.basis.get_rotation_quaternion()))
			print("[flow #", result_id, "] (7) APPLYING RESULT: asking the pose history for the head pose at ",
					capture_usec, " (history holds ", _xr_cam_pose_history.size(), " samples, newest is ",
					"%.1f" % ((now_usec - int(_xr_cam_pose_history[-1][0])) / 1000.0), "ms old)")
			print("[flow #", result_id, "] (8) HEAD POSE AT CAPTURE pos=", cam_xform.origin,
					"  vs LIVE pose now=", live_xform.origin,
					" -> the head moved ", "%.1f" % drift_cm, "cm / turned ", "%.1f" % turn_deg,
					" deg since exposure. Using the LIVE pose instead would displace the patch by exactly that much (= the swim).")
		for id in markers:
			if marker_nodes.has(id):
				# markers[id] is the marker pose in CAMERA space; cam_xform is the head pose at
				# capture time. Bake to world space and freeze it there, so the head can move
				# between detections without dragging the patch along.
				marker_nodes[id].global_transform = cam_xform * markers[id]
				if _tracing(result_id):
					print("[flow #", result_id, "] (9) BAKED marker ", id,
							": camera-space pos=", markers[id].origin,
							"  ->  world pos=", marker_nodes[id].global_transform.origin,
							"  (frozen there until the next detection)")

	# (b) Desktop-only: pull the newest CameraServer frame via GPU->CPU readback. On Android the
	# plugin pushes frames through _on_android_camera_frame instead and cam_texture stays null.
	if cam_texture == null:
		return
	var img := cam_texture.get_image()
	if img == null:
		return
	#print("image format: ",img.get_format()) #format lookup table https://docs.godotengine.org/en/stable/classes/class_image.html#enum-image-format

	_tcp_send_timer += _delta
	if _tcp_send_timer >= TCP_SEND_INTERVAL:
		_tcp_send_timer = 0.0
		if stream_peer != null and stream_peer.get_status() == StreamPeerTCP.STATUS_CONNECTED:
			_send_frame_tcp(img)

	# No sensor timestamp on this path -- approximate the capture time as CAMERA_LATENCY_MS ago.
	_flow_frame_counter += 1
	if _tracing(_flow_frame_counter):
		print("\n[flow #", _flow_frame_counter, "] (1-3) DESKTOP PATH: frame read back from the GPU; no",
				" sensor timestamp exists here, so the capture time is GUESSED as now - CAMERA_LATENCY_MS(",
				CAMERA_LATENCY_MS, "ms)")
	_dispatch_detection(img, now_usec - int(CAMERA_LATENCY_MS * 1000.0), _flow_frame_counter)

# Hand a frame + its capture time to the worker; only the newest frame is kept (frame drop).
func _dispatch_detection(img: Image, capture_usec: int, frame_id: int) -> void:
	_detect_mutex.lock()
	var was_pending := _has_pending
	var dropped_id := _pending_frame_id
	_pending_image = img                                # overwrites any unconsumed frame -> frame drop
	_pending_capture_usec = capture_usec
	_pending_frame_id = frame_id
	_has_pending = true
	_detect_mutex.unlock()
	if not was_pending:
		_detect_sem.post()                                 # only wake once per pending frame (bounded sem)
	if _tracing(frame_id):
		print("[flow #", frame_id, "] (4) HANDED TO WORKER together with its capture time",
				("  [note: frame #%d was still waiting and got DROPPED -- newest frame wins]" % dropped_id) if was_pending else "  [slot was free, worker woken]")

# Worker thread: blocks on the semaphore, runs OpenCV detection off the main thread, and parks the
# result for _process to apply. Touches only `processor` and value types -- never the scene tree.
func _detection_loop() -> void:
	while true:
		_detect_sem.wait()
		if _detect_exit:
			return
		_detect_mutex.lock()
		var img: Image = _pending_image
		var capture_usec: int = _pending_capture_usec
		var frame_id: int = _pending_frame_id
		_has_pending = false
		_pending_image = null
		_detect_mutex.unlock()
		if img == null:
			continue
		# No conversion: the C++ side handles 1ch (Quest Y-plane), 3ch (RGB), and 4ch (RGBA).
		var t0 := Time.get_ticks_usec()
		var traced := _tracing(frame_id)
		if traced:
			print("[flow #", frame_id, "] (5) WORKER PICKED IT UP (background thread), frame is now ",
					"%.1f" % ((t0 - capture_usec) / 1000.0), "ms old. Starting OpenCV detection...")
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
		var lens_q_raw := Quaternion(-0.99519097805023, 0.00269138417207, 0.00294101587497, 0.09787271916866)
		var lens_rotation := (lens_q_raw * Quaternion(1, 0, 0, 0)).inverse() # if visibly worse, try appending .inverse()
		var lens_translation := Vector3(-0.03237725794315, -0.01770938560367, -0.06345107406378) # raw LENS_POSE_TRANSLATION

		var intrinsics := Vector4(fx, fy, cx, cy)
		# OpenCV distCoeffs (k1, k2, p1, p2, k3) for the Quest passthrough lens; pass an empty
		# PackedFloat64Array() for no distortion. These used to be hardcoded in the C++ side.
		var distortion := PackedFloat64Array([-0.00484306,  0.14036606,  0.00044449, -0.00108918, -0.29608385])
		# Combine the lens rotation + translation into one rigid pose (matches the C++ lens_pose param).
		var lens_pose := Transform3D(Basis(lens_rotation), lens_translation)
		var markers: Dictionary = processor.get_6dof_of_all_aruco_patches_from_godot_image(img, aruco_patch_size, image_downscale_factor, intrinsics, distortion, lens_pose)
		print("(worker thread) detect=", (Time.get_ticks_usec() - t0) / 1000.0, "ms  fps=", Engine.get_frames_per_second(), " FPSOfTracking:=", (1000/((Time.get_ticks_usec() - t0) / 1000.0)), " frame_age=", (t0 - capture_usec) / 1000.0, "ms")
		if traced:
			# The marker poses are relative to the CAMERA AT EXPOSURE TIME -- however long the
			# detection took, that fact does not age, which is why capture_usec travels along.
			print("[flow #", frame_id, "] (6) DETECTION DONE in ",
					"%.1f" % ((Time.get_ticks_usec() - t0) / 1000.0), "ms, found ", markers.size(),
					" marker(s) in CAMERA space. Parking result + its capture time for the main thread.")
		_detect_mutex.lock()
		_result_markers = markers
		_result_capture_usec = capture_usec
		_result_frame_id = frame_id
		_worker_has_result = true
		_detect_mutex.unlock()


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
		print("TCP put_data error: ", err)
	
func _connect_tcp() -> void:
	stream_peer = StreamPeerTCP.new()
	stream_peer.big_endian = true

	var err := stream_peer.connect_to_host(TCP_HOST, TCP_PORT)
	print("TCP connect_to_host err: ", err)

	if err != OK:
		print("TCP connect_to_host failed immediately: ", err)


func _poll_tcp(delta: float) -> void:
	if stream_peer == null:
		_connect_tcp()
		return

	stream_peer.poll()

	var status := stream_peer.get_status()

	if status != _last_tcp_status:
		print("TCP status changed: ", _last_tcp_status, " -> ", status)
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
			print("TCP reconnecting...")
			_connect_tcp()
#command to stop (once adb is added to PATH)
# adb shell am force-stop de.unigreifswald.opencvaruco
