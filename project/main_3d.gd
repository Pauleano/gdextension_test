extends Node3D

var processor: OpenCVProcessor
var marker_nodes: Dictionary = {}

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
# output slot (worker -> main), guarded by _detect_mutex
var _result_markers: Dictionary = {}
var _result_cam_xform: Transform3D
var _has_result := false


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
	for child in xr_camera.get_children():
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
		marker_nodes[id].reparent(xr_origin, false) #reparent(new_parent: Node, keep_global_transform: bool = false)

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
	if formats.size() > 10:
		feed.set_format(10, {})        # feed format:2 640x480 YUV_420_888 (for now, choice can be altered) (good ArUco res, light on CPU)
	elif formats.size() > 0:
		feed.set_format(0, {})

	feed.set_active(true)                               # start delivering frames
	cam_texture = CameraTexture.new()
	cam_texture.camera_feed_id = feed.get_id()
	cam_texture.which_feed = CameraServer.FEED_RGBA_IMAGE
	cam_preview.texture = cam_texture
	print("camera feeds: ", feed_count)

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
	var have_result := _has_result
	var markers: Dictionary = _result_markers
	var cam_xform: Transform3D = _result_cam_xform
	_has_result = false
	_detect_mutex.unlock()
	if have_result:
		for id in markers:
			if marker_nodes.has(id):
				# markers[id] is the marker pose in CAMERA space; cam_xform is the head pose at
				# capture time. Bake to world space and freeze it there, so the head can move
				# between detections without dragging the patch along.
				marker_nodes[id].global_transform = cam_xform * markers[id]

	# (b) Hand the newest camera frame to the worker. get_image() (the GPU->CPU readback) and the
	# head-pose snapshot must happen on the main thread; the worker only does the OpenCV work.
	var img := cam_texture.get_image()
	if img == null:
		return
	#print("image format: ",img.get_format()) #format lookup table https://docs.godotengine.org/en/stable/classes/class_image.html#enum-image-format
	
	
	
	_tcp_send_timer += _delta
	if _tcp_send_timer >= TCP_SEND_INTERVAL:
		_tcp_send_timer = 0.0
		if stream_peer != null and stream_peer.get_status() == StreamPeerTCP.STATUS_CONNECTED:
			_send_frame_tcp(img)
	
		
	var cap_cam_xform := xr_camera.global_transform     # head pose AT capture time, for world bake
	_detect_mutex.lock()
	var was_pending := _has_pending
	_pending_image = img                                # overwrites any unconsumed frame -> frame drop
	_pending_cam_xform = cap_cam_xform
	_has_pending = true
	_detect_mutex.unlock()
	if not was_pending:
		_detect_sem.post()                                 # only wake once per pending frame (bounded sem)

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
		_detect_mutex.unlock()
		if img == null:
			continue
		# No conversion: the C++ side handles 1ch (Quest Y-plane), 3ch (RGB), and 4ch (RGBA).
		var t0 := Time.get_ticks_usec()
		var image_downscale_factor=0.5 #1 is original image, 0.5 means half width and half height
		var aruco_patch_size=0.1 #in meters
		var fx=877.06583568*image_downscale_factor
		var fy=878.33004836*image_downscale_factor
		var cx=645.36226952*image_downscale_factor #approxiamte cx is fx/2
		var cy=642.24557861*image_downscale_factor #approxiamte cy is fy/2
		# Physical passthrough-camera offset from the head-tracked reference (XRCamera3D), from the
		# Quest's ACAMERA_LENS_POSE_ROTATION / _TRANSLATION. init_quest_intrinsics() prints these on
		# the C++ side -- paste the values you see in the device logs here. 
		var lens_rotation := Quaternion(-0.99519097805023,0.00269138417207, 0.00294101587497, 0.09787271916866)   #quaternion(x,y,z,w) from device logs  
		var lens_translation := Vector3(-0.03237725794315, -0.01770938560367, -0.06345107406378)       # Vector3(tx, ty, tz) from device logs
		
		var markers: Dictionary = processor.get_6dof_of_all_aruco_patches_from_godot_image(img, aruco_patch_size,image_downscale_factor,fx,fy,cx,cy,lens_rotation,lens_translation)
		print("(worker thread) detect=", (Time.get_ticks_usec() - t0) / 1000.0, "ms  fps=", Engine.get_frames_per_second(), " FPSOfTracking:=", (1000/((Time.get_ticks_usec() - t0) / 1000.0)) )
		_detect_mutex.lock()
		_result_markers = markers
		_result_cam_xform = cam_xform
		_has_result = true
		_detect_mutex.unlock()

func _exit_tree() -> void:
	# Wake the worker out of its wait() and join it, so the thread doesn't outlive the scene.
	if _detect_thread != null and _detect_thread.is_started():
		_detect_mutex.lock()
		_detect_exit = true
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
# adb shell am force-stop com.example.godotcpptemplate
