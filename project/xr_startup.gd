extends XROrigin3D

# Minimal OpenXR startup: find the interface, initialise it, and switch the
# viewport to XR. Works on both desktop (no headset -> flat fallback) and Quest.

var xr_interface: XRInterface

func _ready() -> void:
	xr_interface = XRServer.find_interface("OpenXR")
	if xr_interface and xr_interface.is_initialized():
		print("OpenXR initialised successfully")
		# The XR compositor drives frame pacing; disable the desktop v-sync.
		DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
		get_viewport().use_xr = true
	else:
		print("OpenXR not initialised - running in flat (non-XR) mode")
