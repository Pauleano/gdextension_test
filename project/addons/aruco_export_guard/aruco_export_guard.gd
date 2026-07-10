@tool
extends EditorPlugin

const GuardScript := preload("res://addons/aruco_export_guard/aruco_export_check.gd")

var _guard: EditorExportPlugin


func _enter_tree() -> void:
	_guard = GuardScript.new()
	add_export_plugin(_guard)


func _exit_tree() -> void:
	remove_export_plugin(_guard)
	_guard = null
