
#pragma once

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/open_xrapi_extension.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/transform3d.hpp>

#include "openxr_minimal.h"

using namespace godot;

//Head (VIEW space) pose at an ARBITRARY XrTime, straight from the OpenXR runtime.
//
//Godot only ever hands out the head pose for xrWaitFrame's predicted display time -- a
//FORECAST of where the head will be when the frame is scanned out. Pairing a camera frame with
//it means baking the runtime's prediction error into every marker pose, worst exactly during
//the fast head motion where it shows. xrLocateSpace with the frame's own capture time returns
//the runtime's fused, after-the-fact estimate for that instant instead: measured, not predicted.
//
//Nothing here links the OpenXR loader. Godot has already created the instance, the session and
//the play space; we only borrow them (OpenXRAPIExtension) and fetch the three entry points we
//need through xrGetInstanceProcAddr. See openxr_minimal.h for why there is no <openxr/openxr.h>.
//
//Main thread only, like everything else that talks to the XR runtime here.
class OpenXRHeadLocator : public Node {
	GDCLASS(OpenXRHeadLocator, Node)

private:
	//Created lazily rather than in the constructor: OpenXRAPIExtension is a thin wrapper around
	//the OpenXRAPI singleton, which may not exist yet (or at all) when this node is built.
	Ref<OpenXRAPIExtension> xr_api;

	PFN_xrCreateReferenceSpace xr_create_reference_space = nullptr;
	PFN_xrLocateSpace xr_locate_space = nullptr;
	PFN_xrDestroySpace xr_destroy_space = nullptr;
	bool procs_resolved = false;

	XrSpace view_space = XR_NULL_HANDLE;
	//The session view_space was created under. A space is a CHILD of its session: when the
	//session goes away the runtime destroys the space with it, and destroying it ourselves
	//afterwards is undefined behaviour. Comparing against the live session id is what lets us
	//tell "still ours" from "stale handle, forget it" after a session restart.
	uint64_t view_space_session = 0;
	//locate_head sits on the per-frame path, so a runtime that rejects every call must not print
	//at the camera rate; this counts failures so only every 128th is logged.
	uint64_t locate_fail_count = 0;

	//Resolve the entry points and (re)create the VIEW space if the session allows it right now.
	//Idempotent and cheap once warm (three pointer compares), so locate_head can just call it.
	bool ensure_space();

	//gate for the chatty output; errors print regardless. Static for the same reason as in
	//OpenCVProcessor: GDScript sets it once, for every instance, before anything is built.
	static bool debug_prints_enabled;

protected:
	static void _bind_methods();

public:
	OpenXRHeadLocator();
	~OpenXRHeadLocator();

	static void set_debug_prints_enabled(bool p_enabled);

	//True once the VIEW space exists, i.e. once locate_head can return anything. False before
	//the OpenXR session is up, and on a build/desktop run without OpenXR at all.
	bool is_ready();

	//Head pose at p_xr_time, expressed in Godot's OpenXR PLAY space -- NOT world space. The
	//caller still has to apply XROrigin3D's transform, XRServer.get_reference_frame() and the
	//world scale to get where XRCamera3D would be (see _play_space_to_world in main_3d.gd).
	//
	//p_xr_time is a raw XrTime in nanoseconds: on the Quest that is CLOCK_MONOTONIC, the same
	//clock the passthrough camera stamps its frames with, so a frame's sensor timestamp can go
	//in unconverted. A time in the past is the whole point; the runtime answers it from its
	//tracking history.
	//
	//Returns { "valid": bool, "tracked": bool, "transform": Transform3D, "flags": int,
	//          "result": int }:
	//  valid   -- orientation AND position may be read; false means ignore "transform" entirely
	//  tracked -- additionally being measured right now, rather than extrapolated through a
	//             tracking loss. valid && !tracked is a usable but degraded pose.
	//  flags   -- the raw XrSpaceLocationFlags, for logging
	//  result  -- the XrResult; negative on failure (e.g. the runtime refusing a time too far
	//             in the past), 0 on success
	Dictionary locate_head(int64_t p_xr_time);

	//Destroy the VIEW space while its session is still alive. Connect this to OpenXRInterface's
	//session_stopping signal; the destructor cannot be relied on for it, because by the time the
	//scene tree is torn down the session may already be gone.
	void release();
};
