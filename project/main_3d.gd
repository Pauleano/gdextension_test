extends Node3D

var processor: OpenCVProcessor
var marker_nodes: Dictionary = {}

# keep a reference so the camera backend isn't freed
var camera_extension: CameraServerExtension
var cam_texture: CameraTexture
@onready var cam_preview: TextureRect = $CameraLayer/CameraPreview

func _ready() -> void:
	#processor = OpenCVProcessor.new()

	#for child in $Camera3D.get_children():
	#	var n: String = child.name
	#	if n.begins_with("aruco_patch"):
	#		var id_str: String = n.substr(11)
	#		if id_str.is_valid_int():
	#			marker_nodes[id_str.to_int()] = child

	#var output2: Transform3D = processor.get_6dof_of_aruco_patch_from_picture("res://assets/img_of_marker0_dict4x4_50.png")
	#print(output2)

	# --- live camera display via CameraServerExtension addon ---
	camera_extension = CameraServerExtension.new()      # keep reference alive
	# enable feed enumeration (4.5+), Since Godot v4.5, CameraServerExtension requires setting CameraServer.monitoring_feeds to true before instantiation to provide custom feeds.
	CameraServer.monitoring_feeds = true                
	CameraServer.camera_feeds_updated.connect(_on_camera_feeds_updated)
	_on_camera_feeds_updated()                          # in case a feed is already present
	

func _on_camera_feeds_updated() -> void:
	if cam_texture != null:
		return                                          # already initialised
	if CameraServer.get_feed_count() == 0:				#if no camerafeeds available
		return
	
	
	#for i in range(CameraServer.get_feed_count()):
	#	var feed = CameraServer.get_feed(i)
	#	print("Index:", i, " ID:", feed.get_id())
	
	var feed := CameraServer.get_feed(0)				#only have one laptop camera (index 0=use first available, but internal feed_id is "1")
	feed.set_active(true)                               # start delivering frames
	cam_texture = CameraTexture.new()
	cam_texture.camera_feed_id = feed.get_id()			
	
	#print("Bildformat:", feed.get_datatype())			#returns 1 (RGB),lookup-table: https://docs.godotengine.org/en/stable/classes/class_camerafeed.html#enum-camerafeed-feeddatatype
	cam_texture.which_feed = CameraServer.FEED_RGBA_IMAGE
	cam_preview.texture = cam_texture
	print("camera feeds: ", CameraServer.get_feed_count())


func _process(_delta: float) -> void:
	return  # TEMP: disabled so Godot's CameraFeed can own the webcam (Windows = single owner)
	# var markers: Dictionary = processor.get_6dof_of_all_aruco_patches_from_webcam(0.05)
	# for id in markers:
	# 	if marker_nodes.has(id):
	# 		marker_nodes[id].set_transform(markers[id])
