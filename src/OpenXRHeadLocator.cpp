#include "OpenXRHeadLocator.h"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/basis.hpp>
#include <godot_cpp/variant/quaternion.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/variant/vector3.hpp>

using namespace godot;

bool OpenXRHeadLocator::debug_prints_enabled = false;

//Same output convention as OpenCVProcessor.cpp: "[opencv_aruco] [Klasse::funktion] text", damit
//alles dieser Extension in logcat ueber EIN grep zu finden ist. ACV_DBG haengt am Flag aus
//GDScript, ACV_ERR nicht.
#define ACV_DBG(...) \
	do { if (debug_prints_enabled) UtilityFunctions::print("[opencv_aruco] [OpenXRHeadLocator::", __func__, "] ", __VA_ARGS__); } while (0)
#define ACV_ERR(...) \
	UtilityFunctions::printerr("[opencv_aruco] [OpenXRHeadLocator::", __func__, "] ", __VA_ARGS__)

void OpenXRHeadLocator::_bind_methods() {
	ClassDB::bind_method(D_METHOD("is_ready"), &OpenXRHeadLocator::is_ready);
	ClassDB::bind_method(D_METHOD("locate_head", "xr_time"), &OpenXRHeadLocator::locate_head);
	ClassDB::bind_method(D_METHOD("release"), &OpenXRHeadLocator::release);
	//statisch, damit GDScript es -- wie bei OpenCVProcessor -- schon vor new() setzen kann
	ClassDB::bind_static_method("OpenXRHeadLocator", D_METHOD("set_debug_prints_enabled", "enabled"), &OpenXRHeadLocator::set_debug_prints_enabled);
}

void OpenXRHeadLocator::set_debug_prints_enabled(bool p_enabled) {
	debug_prints_enabled = p_enabled;
}

OpenXRHeadLocator::OpenXRHeadLocator() {}

OpenXRHeadLocator::~OpenXRHeadLocator() {
	//Belt and braces only: by scene-tree teardown the session is usually already gone, and
	//release() notices that and skips xrDestroySpace instead of calling it on a dead handle.
	//The real cleanup path is the session_stopping signal (see main_3d.gd).
	release();
}

bool OpenXRHeadLocator::ensure_space() {
	if (xr_api.is_null()) {
		xr_api.instantiate();
	}
	//Guards the whole no-OpenXR case (desktop run without a headset): every OpenXRAPIExtension
	//method null-checks the singleton, and is_initialized() is false when there is none.
	if (!xr_api->is_initialized()) {
		return false;
	}

	const uint64_t session = xr_api->get_session();
	if (session == 0) {
		return false;                          //session not created yet -- try again next call
	}

	if (view_space != XR_NULL_HANDLE) {
		if (view_space_session == session) {
			return true;                       //warm path: nothing to do
		}
		//Session was restarted. The old space died with the old session, so the handle is
		//merely dropped -- calling xrDestroySpace on it now would be undefined behaviour.
		ACV_DBG("session changed, dropping stale view space: old=", (int64_t)view_space_session, " new=", (int64_t)session);
		view_space = XR_NULL_HANDLE;
		view_space_session = 0;
	}

	if (!procs_resolved) {
		//All three are core OpenXR 1.0 entry points, so a null here means the instance itself is
		//not usable -- worth one loud line rather than a silent fallback.
		xr_create_reference_space = reinterpret_cast<PFN_xrCreateReferenceSpace>(
				static_cast<uintptr_t>(xr_api->get_instance_proc_addr("xrCreateReferenceSpace")));
		xr_locate_space = reinterpret_cast<PFN_xrLocateSpace>(
				static_cast<uintptr_t>(xr_api->get_instance_proc_addr("xrLocateSpace")));
		xr_destroy_space = reinterpret_cast<PFN_xrDestroySpace>(
				static_cast<uintptr_t>(xr_api->get_instance_proc_addr("xrDestroySpace")));
		procs_resolved = true;
		if (xr_create_reference_space == nullptr || xr_locate_space == nullptr || xr_destroy_space == nullptr) {
			ACV_ERR("xrGetInstanceProcAddr failed: create=", (uint64_t)(uintptr_t)xr_create_reference_space,
					" locate=", (uint64_t)(uintptr_t)xr_locate_space,
					" destroy=", (uint64_t)(uintptr_t)xr_destroy_space);
		} else {
			ACV_DBG("openxr entry points resolved");
		}
	}
	if (xr_create_reference_space == nullptr || xr_locate_space == nullptr) {
		return false;
	}

	//Our own VIEW space, because Godot exposes the play space but not its internal view space.
	//It is the same reference space Godot locates for the head, with an identity offset, so the
	//pose that comes back is directly comparable to XRCamera3D -- only the TIME differs, which
	//is the entire point of this class.
	XrReferenceSpaceCreateInfo create_info{};
	create_info.type = XR_TYPE_REFERENCE_SPACE_CREATE_INFO;
	create_info.next = nullptr;
	create_info.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
	create_info.poseInReferenceSpace.orientation.x = 0.0f;
	create_info.poseInReferenceSpace.orientation.y = 0.0f;
	create_info.poseInReferenceSpace.orientation.z = 0.0f;
	create_info.poseInReferenceSpace.orientation.w = 1.0f;   //identity quaternion, NOT all-zero
	create_info.poseInReferenceSpace.position.x = 0.0f;
	create_info.poseInReferenceSpace.position.y = 0.0f;
	create_info.poseInReferenceSpace.position.z = 0.0f;

	XrSpace space = XR_NULL_HANDLE;
	const XrResult result = xr_create_reference_space(
			reinterpret_cast<XrSession>(static_cast<uintptr_t>(session)), &create_info, &space);
	if (!XR_SUCCEEDED(result) || space == XR_NULL_HANDLE) {
		ACV_ERR("xrCreateReferenceSpace(VIEW) failed: result=", (int64_t)result,
				" (", xr_api->get_error_string(result), ")");
		return false;
	}

	view_space = space;
	view_space_session = session;
	locate_fail_count = 0;
	ACV_DBG("view space created: session=", (int64_t)session);
	return true;
}

bool OpenXRHeadLocator::is_ready() {
	return ensure_space();
}

Dictionary OpenXRHeadLocator::locate_head(int64_t p_xr_time) {
	Dictionary out;
	out["valid"] = false;
	out["tracked"] = false;
	out["transform"] = Transform3D();
	out["flags"] = 0;
	out["result"] = 0;

	if (!ensure_space()) {
		return out;
	}
	//Godot's play space is what XROrigin3D stands for; locating the head against it gives
	//exactly the pose XRCamera3D carries (up to the reference frame and world scale the caller
	//applies). 0 would mean the session exists but the play space does not yet.
	const uint64_t play_space = xr_api->get_play_space();
	if (play_space == 0) {
		return out;
	}

	XrSpaceLocation location{};
	location.type = XR_TYPE_SPACE_LOCATION;
	location.next = nullptr;

	const XrResult result = xr_locate_space(view_space,
			reinterpret_cast<XrSpace>(static_cast<uintptr_t>(play_space)),
			(XrTime)p_xr_time, &location);
	out["result"] = (int64_t)result;
	if (!XR_SUCCEEDED(result)) {
		//This sits on the per-frame path, so an unhappy runtime must not flood logcat at the
		//camera rate: the first failure and then every 128th carry the message.
		if ((locate_fail_count++ % 128) == 0) {
			ACV_ERR("xrLocateSpace failed: result=", (int64_t)result,
					" (", xr_api->get_error_string(result), ") xr_time=", p_xr_time,
					" failures=", (int64_t)locate_fail_count);
		}
		return out;
	}

	const XrSpaceLocationFlags flags = location.locationFlags;
	out["flags"] = (int64_t)flags;
	//Both halves must be VALID before `pose` may be read at all -- an invalid half is not
	//"identity", it is uninitialised as far as the spec is concerned.
	const bool valid = (flags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT)
			&& (flags & XR_SPACE_LOCATION_POSITION_VALID_BIT);
	out["valid"] = valid;
	out["tracked"] = (flags & XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT)
			&& (flags & XR_SPACE_LOCATION_POSITION_TRACKED_BIT);
	if (valid) {
		//XrPosef -> Transform3D by hand. OpenXR and Godot share the right-handed Y-up convention,
		//so this is a straight copy of 7 floats.
		//NOT OpenXRAPIExtension::transform_from_pose(): that binding takes a `const void *`, and
		//godot-cpp does not marshal the opaque pointer through -- measured on device it returns
		//IDENTITY for every pose, silently, with no error anywhere. That cost one full device
		//round trip to find, because an identity head pose looks exactly like a plausible-but-
		//wrong pose: it sits head-height away from the real one and never changes with time.
		//Do not "simplify" this back to the engine call.
		const Quaternion q(location.pose.orientation.x, location.pose.orientation.y,
				location.pose.orientation.z, location.pose.orientation.w);
		const Vector3 p(location.pose.position.x, location.pose.position.y, location.pose.position.z);
		out["transform"] = Transform3D(Basis(q), p);
	}
	return out;
}

void OpenXRHeadLocator::release() {
	if (view_space == XR_NULL_HANDLE) {
		return;
	}
	//Only destroy a space whose session is still the live one; otherwise the runtime has already
	//taken it down together with the session and we would be passing a dangling handle.
	const uint64_t live_session = (xr_api.is_valid() && xr_api->is_initialized()) ? xr_api->get_session() : 0;
	if (live_session != 0 && live_session == view_space_session && xr_destroy_space != nullptr) {
		const XrResult result = xr_destroy_space(view_space);
		if (!XR_SUCCEEDED(result)) {
			ACV_ERR("xrDestroySpace failed: result=", (int64_t)result);
		} else {
			ACV_DBG("view space destroyed");
		}
	} else {
		ACV_DBG("view space dropped without xrDestroySpace: session already gone");
	}
	view_space = XR_NULL_HANDLE;
	view_space_session = 0;
}
