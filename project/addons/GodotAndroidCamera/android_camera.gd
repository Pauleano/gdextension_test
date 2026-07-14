class_name AndroidCamera extends Node

const _plugin_name: String = "GodotAndroidCamera"

## Pixel format of the frames delivered by [signal camera_frame].
enum OutputFormat {
	## Grayscale, 1 byte per pixel: the camera's native Y (luminance) plane, forwarded
	## without any conversion or readback. Ideal for computer vision (e.g. ArUco).
	LUMA = 0,
	## RGBA, 4 bytes per pixel: CameraX converts every frame from the camera's native
	## YUV on the CPU before delivery. Use for display; costs one conversion per frame.
	RGBA = 1,
}

var java_interface: JNISingleton

## Emitted for every camera frame.
## [param timestamp_ns] is the sensor timestamp (start of exposure) in nanoseconds.
## When the feed's "timestamp_source" is "realtime" it is on the same clock as
## [method get_current_timestamp_nanos] and Android sensor events, so detections can be
## matched exactly against a camera-pose stream. See [method get_clock_offset_nanos].
signal camera_frame(timestamp_ns: int, data: PackedByteArray, width: int, height: int)

func _initialize_java_interface() -> void:
	assert(Engine.has_singleton(_plugin_name))
	java_interface = Engine.get_singleton(_plugin_name)
	java_interface.connect("on_camera_frame", _on_camera_frame)

## Internal signal method.
func _on_camera_frame(timestamp_ns: int, data: PackedByteArray, width: int, height: int) -> void:
	if not is_instance_valid(java_interface):
		_initialize_java_interface()
	camera_frame.emit(timestamp_ns, data, width, height)

## Converts raw data from a frame into an ImageTexture.
## Pass the same [param output_format] the camera was started with.
static func raw_data_to_image(data: PackedByteArray, width: int, height: int,
		output_format: OutputFormat = OutputFormat.RGBA) -> ImageTexture:
	var format := Image.FORMAT_RGBA8 if output_format == OutputFormat.RGBA else Image.FORMAT_L8
	var image: Image = Image.create_from_data(width, height, false, format, data)
	return ImageTexture.create_from_image(image)

## Shows camera permissions pop-up. Required for other camera functions to work.
## Returns true if permissions accepted, false otherwise.
func request_camera_permissions() -> bool:
	if not is_instance_valid(java_interface):
		_initialize_java_interface()

	java_interface.requestCameraPermissions()
	return java_interface.allPermissionsGranted()

## Lists the available camera feeds. Returns a Dictionary keyed by camera id, e.g.:
##     { "0": {"facing": "back", "sensor_orientation": 90, "has_flash": true,
##             "timestamp_source": "realtime"}, "1": {...} }
## On Meta Quest 3/3S (Horizon OS v76+) the passthrough cameras additionally carry
##     "source": "passthrough" and "position": "left" / "right"
## so the two head-mounted cameras can be told apart.
## Pass a key as camera_id to [method start_camera] to choose that feed.
func get_available_cameras() -> Dictionary:
	if not is_instance_valid(java_interface):
		_initialize_java_interface()

	return java_interface.getAvailableCameras()

## Lists the formats supported by a feed ("" = default back camera). Returns:
##     { "resolutions": [{"width": int, "height": int, "max_fps": int}, ...],
##       "fps_ranges":  [{"lower": int, "upper": int}, ...] }
## resolutions are the sizes the camera can stream; fps_ranges are the auto-exposure
## target ranges accepted by [method start_camera]'s fps_lower/fps_upper.
func get_supported_formats(camera_id: String = "") -> Dictionary:
	if not is_instance_valid(java_interface):
		_initialize_java_interface()

	var raw: Dictionary = java_interface.getSupportedFormats(camera_id)
	var result := {"resolutions": [], "fps_ranges": []}
	var resolutions: PackedInt32Array = raw.get("resolutions", PackedInt32Array())
	for i in range(0, resolutions.size() - 2, 3):
		result.resolutions.append({
			"width": resolutions[i],
			"height": resolutions[i + 1],
			"max_fps": resolutions[i + 2],
		})
	var fps_ranges: PackedInt32Array = raw.get("fps_ranges", PackedInt32Array())
	for i in range(0, fps_ranges.size() - 1, 2):
		result.fps_ranges.append({"lower": fps_ranges[i], "upper": fps_ranges[i + 1]})
	return result

## Camera intrinsics for 6DoF pose estimation ("" = default back camera). Returns a
## Dictionary with (when the device provides them):
##     "intrinsics": PackedFloat32Array [fx, fy, cx, cy, skew] in active-array pixels,
##     "distortion": PackedFloat32Array of lens distortion coefficients (API 28+),
##     "active_array_width" / "active_array_height": int,
##     "estimated": true if intrinsics were derived from focal length and sensor size
##                  rather than factory calibration,
##     "lens_pose_translation" (float[3]) / "lens_pose_rotation" (quaternion float[4]):
##                  the lens pose relative to the device's pose reference — on Quest
##                  passthrough cameras, the camera's offset from the headset, for
##                  chaining marker -> camera -> head -> world transforms,
##     "lens_pose_reference": what the pose is relative to (see Android
##                  LENS_POSE_REFERENCE; API 28+).
## Scale fx/cx by frame_width / active_array_width and fy/cy by
## frame_height / active_array_height to get the matrix for the streamed resolution.
func get_camera_intrinsics(camera_id: String = "") -> Dictionary:
	if not is_instance_valid(java_interface):
		_initialize_java_interface()

	return java_interface.getCameraIntrinsics(camera_id)

## Current time in nanoseconds on the clock frame timestamps use (when the feed's
## "timestamp_source" is "realtime").
func get_current_timestamp_nanos() -> int:
	if not is_instance_valid(java_interface):
		_initialize_java_interface()

	return java_interface.getCurrentTimestampNanos()

## Offset between Godot's clock and the boottime clock (elapsedRealtimeNanos), such that
##     Time.get_ticks_usec() * 1000 + offset  ≈  frame timestamp_ns
## Valid for feeds whose "timestamp_source" is "realtime". Sample once (or occasionally)
## and use it to place camera frames on the same timeline as engine-side data such as a
## camera-pose stream. For "unknown" feeds see [method get_monotonic_clock_offset_nanos].
func get_clock_offset_nanos() -> int:
	if not is_instance_valid(java_interface):
		_initialize_java_interface()

	var godot_usec := Time.get_ticks_usec()
	var camera_ns: int = java_interface.getCurrentTimestampNanos()
	return camera_ns - godot_usec * 1000

## Same as [method get_clock_offset_nanos] but against CLOCK_MONOTONIC (System.nanoTime),
## the clock feeds with "timestamp_source": "unknown" typically stamp frames on. Godot's
## Time.get_ticks_usec() is itself monotonic, so this offset is exact and drift-free for
## such feeds, while the boottime offset would be wrong by the device's total doze time.
func get_monotonic_clock_offset_nanos() -> int:
	if not is_instance_valid(java_interface):
		_initialize_java_interface()

	var godot_usec := Time.get_ticks_usec()
	var camera_ns: int = java_interface.getCurrentMonotonicNanos()
	return camera_ns - godot_usec * 1000

## Starts streaming camera frames (via camera_frame signal) with given parameters.
## The resulting image will be the closest size to (desired_width, desired_height)
## available.
## [param camera_id]: a key of [method get_available_cameras], or "" for the default
## back camera.
## [param fps_lower]/[param fps_upper]: a range from [method get_supported_formats]'
## fps_ranges; leave both 0 to select the highest range automatically.
## [param output_format]: see [enum OutputFormat]; use LUMA for CV pipelines.
func start_camera(desired_width: int, desired_height: int, flash_on: bool = false,
		camera_id: String = "", fps_lower: int = 0, fps_upper: int = 0,
		output_format: OutputFormat = OutputFormat.RGBA) -> void:
	if not is_instance_valid(java_interface):
		_initialize_java_interface()

	java_interface.startCamera(camera_id, desired_width, desired_height,
			fps_lower, fps_upper, output_format, flash_on)

## Stops camera streaming.
func stop_camera() -> void:
	if not is_instance_valid(java_interface):
		_initialize_java_interface()

	java_interface.stopCamera()

## Returns an array of the sampling frequency range in the format [lower, upper] Hz.
## For most phones, this should be [60, 60] or [30, 60].
func get_sampling_frequency_range() -> PackedInt32Array:
	return java_interface.getSamplingFrequencyRange()
