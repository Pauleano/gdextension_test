extends Node3D

var processor: OpenCVProcessor
var marker_nodes: Dictionary = {}

func _ready() -> void:
	processor = OpenCVProcessor.new()

	for child in $Camera3D.get_children():
		var n: String = child.name
		if n.begins_with("aruco_patch"):
			var id_str: String = n.substr(11)
			if id_str.is_valid_int():
				marker_nodes[id_str.to_int()] = child

	var output2: Transform3D = processor.get_6dof_of_aruco_patch_from_picture("res://assets/img_of_marker0_dict4x4_50.png")
	print(output2)


func _process(_delta: float) -> void:
	var markers: Dictionary = processor.get_6dof_of_all_aruco_patches_from_webcam(0.05)
	for id in markers:
		if marker_nodes.has(id):
			marker_nodes[id].set_transform(markers[id])
