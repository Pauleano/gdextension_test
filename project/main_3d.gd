extends Node3D
#@onready var processor: OpenCVProcessor = $OpenCVProcessor
#@onready var marker: Node3D = $Camera3D/aruco_patch

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	# ": type" ist optional (strong typing)
	var processor: OpenCVProcessor = OpenCVProcessor.new()
	#"res://assets/Caution.png" does not work, because underlying cv:: function expect path starting from C:
	#using \ leads to error: "invalid escape in string"
	#var output: Vector2 = processor.get_image_size("C:/Users/Paul Geiger/Documents/gdextension_test/gdextension_test/project/assets/caution.png")
	#var output: Vector2 = processor.get_image_size("res://assets/Caution.png")
	#var output: Vector2 = processor.get_image_size("res://assets/img_of_marker0_dict4x4_50.png")
	#print(output)
	var output2: Transform3D =processor.get_6dof_of_aruco_patch_from_picture("res://assets/img_of_marker0_dict4x4_50.png")
	print(output2)
	print($aruco_patch.position)
	print($aruco_patch/MeshInstance3D.position)
	$aruco_patch.set_transform(output2)
	print($aruco_patch.position)
	print($aruco_patch/MeshInstance3D.position)
	
	#processor.marker_pose_found.connect(_on_pose)
	#processor.get_6dof_of_aruco_patch_from_picture("res://assets/img_of_marker0_dict4x4_50.png")


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(_delta: float) -> void:
	pass

#func _on_pose(pose: Transform3D) -> void:
#	marker.transform = pose
