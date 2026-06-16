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

func _ready() -> void:
	processor = OpenCVProcessor.new()

	# detect all available aruco_patch nodes, to later set their position
	for child in $XROrigin3D/XRCamera3D.get_children():
		var n: String = child.name
		if n.begins_with("aruco_patch"):
			var id_str: String = n.substr(11)
			if id_str.is_valid_int():
				marker_nodes[id_str.to_int()] = child

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
		feed.set_format(2, {})        # 640x480 YUV_420_888 (good ArUco res, light on CPU)
	elif formats.size() > 0:
		feed.set_format(0, {})

	feed.set_active(true)                               # start delivering frames
	cam_texture = CameraTexture.new()
	cam_texture.camera_feed_id = feed.get_id()
	cam_texture.which_feed = CameraServer.FEED_RGBA_IMAGE
	cam_preview.texture = cam_texture
	print("camera feeds: ", feed_count)


func _process(_delta: float) -> void:
	# pull the frame Godot's CameraFeed already owns; no second webcam capture in OpenCV
	if cam_texture == null:
		return
	var img := cam_texture.get_image()
	if img == null:
		return
	# No conversion: the C++ side handles 1ch (Quest Y-plane), 3ch (RGB), and 4ch (RGBA)
	# from the raw image data. Converting R8->RGB8 here would zero G/B and darken the image.
	var markers: Dictionary = processor.get_6dof_of_all_aruco_patches_from_godot_image(img, 0.05)
	for id in markers:
		if marker_nodes.has(id):
			print("id: ",id," 6dofs: ",markers[id])
			marker_nodes[id].set_transform(markers[id])

#command to stop (once adb is added to PATH)
# adb shell am force-stop com.example.godotcpptemplate
