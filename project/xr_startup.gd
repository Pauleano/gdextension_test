extends XROrigin3D

# Minimal OpenXR startup: find the interface, initialise it, and switch the
# viewport to XR. Works on both desktop (no headset -> flat fallback) and Quest.

var xr_interface: XRInterface


func _ready() -> void:
	print("available interfaces:",XRServer.get_interfaces())
	xr_interface = XRServer.find_interface("OpenXR")
	print("XRinterface initialised:",xr_interface.is_initialized())
	if xr_interface and xr_interface.is_initialized():
		print("OpenXR initialised successfully")
		# The XR compositor drives frame pacing; disable the desktop v-sync. (dont wait
		DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
		get_viewport().use_xr = true

		OpenXRMetaEnvironmentDepthExtension.start_environment_depth()
		# optional: keep your own hands from occluding the patches
		#OpenXRMetaEnvironmentDepthExtension.set_hand_removal_enabled(true)


		_enable_passthrough()
	else:
		print("OpenXR not initialised - running in flat (non-XR) mode")

# Meta passthrough (AR): composite the rendered scene over the real world by switching the
# XR environment blend mode to alpha-blend and clearing the viewport to transparent. Wherever
# the scene draws nothing, the real-world camera shows through; opaque meshes render on top.
func _enable_passthrough() -> void:
	var modes := xr_interface.get_supported_environment_blend_modes()
	if XRInterface.XR_ENV_BLEND_MODE_ALPHA_BLEND in modes:
		xr_interface.environment_blend_mode = XRInterface.XR_ENV_BLEND_MODE_ALPHA_BLEND
		get_viewport().transparent_bg = true
		print("passthrough enabled (alpha blend)")
	else:
		print("alpha-blend passthrough not supported; available modes: ", modes)
