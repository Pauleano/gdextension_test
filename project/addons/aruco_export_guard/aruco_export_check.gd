@tool
extends EditorExportPlugin

# Godot exports a BROKEN apk when a GDExtension library listed in the
# .gdextension is missing on disk: it packages a 0-byte .so and reports
# "Completed successfully" (EditorExportPlatformAndroid::save_apk_so only
# ERR_PRINTs). This guard fails the export instead.
#
# Android preset option -> godot-cpp arch suffix.
# Keep in sync with res://bin/opencv_aruco.gdextension.
const ABI_ARCHS := {
	"architectures/armeabi-v7a": "arm32",
	"architectures/arm64-v8a": "arm64",
	"architectures/x86": "x86_32",
	"architectures/x86_64": "x86_64",
}

# Non-empty => a required lib was missing: delete the artifact in _export_end()
# (the engine still writes the apk after an EXPORT_MESSAGE_ERROR; removing it
# makes the failure hard for CI/headless exports too, which always exit 0).
var _abort_path := ""


func _get_name() -> String:
	return "aruco_android_guard"


func _supports_platform(platform: EditorExportPlatform) -> bool:
	return platform.get_os_name() == "Android"


func _export_begin(features: PackedStringArray, is_debug: bool, path: String, _flags: int) -> void:
	_abort_path = ""
	var target := "template_debug" if is_debug else "template_release"
	var precision := ".double" if features.has("double") else ""
	var missing := PackedStringArray()
	for option: String in ABI_ARCHS:
		# get_option() is valid here: the engine assigns the preset to the
		# plugin right before calling _export_begin.
		if not bool(get_option(option)):
			continue
		var lib := "res://bin/android/libopencv_aruco.android.%s%s.%s.so" % [target, precision, ABI_ARCHS[option]]
		if not FileAccess.file_exists(lib):
			missing.append(lib)
	if missing.is_empty():
		return
	_abort_path = path
	var msg := "Missing OpenCV ArUco GDExtension librar%s for the enabled Android ABIs:\n  %s\nBuild first (see README):  scons platform=android target=%s" \
			% ["y" if missing.size() == 1 else "ies", "\n  ".join(missing), target]
	# EXPORT_MESSAGE_ERROR marks the export result dialog as red "Failed."
	get_export_platform().add_message(EditorExportPlatform.EXPORT_MESSAGE_ERROR, "OpenCV ArUco", msg)
	push_error(msg)  # also reach the Output panel / headless stderr


func _export_end() -> void:
	if _abort_path.is_empty():
		return
	if FileAccess.file_exists(_abort_path):
		DirAccess.remove_absolute(_abort_path)
		push_error("OpenCV ArUco export guard: removed incomplete artifact '%s'." % _abort_path)
	_abort_path = ""
