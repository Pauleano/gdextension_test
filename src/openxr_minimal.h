#pragma once

//Minimal, self-contained transcription of the handful of OpenXR 1.0 declarations that
//OpenXRHeadLocator needs. Deliberately NOT the official <openxr/openxr.h>:
//
//  * We never LINK against the OpenXR loader. Godot has already created the instance and the
//    session; every entry point we use is fetched at runtime through
//    OpenXRAPIExtension::get_instance_proc_addr(). So all that is missing are struct layouts,
//    a few enum values and the function-pointer signatures -- no library, no import lib, no
//    android_native_app_glue.
//  * Vendoring the real header (~4700 lines, plus openxr_platform_defines.h) would mean a new
//    include path in SConstruct for BOTH platforms, or a new Conan dependency cross-compiled
//    for arm64. That is a lot of build surface for three structs.
//
//Everything below is fixed by the OpenXR 1.0 ABI and can therefore never change: the spec
//guarantees backwards compatibility for existing structures and enumerant values. The numeric
//literals are the spec's, spelled out here because there is no header to take them from:
//  XR_TYPE_REFERENCE_SPACE_CREATE_INFO = 37, XR_TYPE_SPACE_LOCATION = 42 (XrStructureType)
//  XR_REFERENCE_SPACE_TYPE_VIEW        = 1                              (XrReferenceSpaceType)
//A wrong value here does not corrupt anything -- the runtime rejects the call and returns an
//XrResult, which OpenXRHeadLocator logs. If this ever needs more of OpenXR than the three
//functions below, swap this file for the real SDK header rather than growing it.

#include <stdint.h>

//Calling convention, as in openxr_platform_defines.h. A no-op everywhere we build (__stdcall is
//ignored on x86_64), but wrong function-pointer types are exactly the kind of thing that only
//breaks on a platform you did not test.
#if defined(_WIN32)
#define XRAPI_PTR __stdcall
#else
#define XRAPI_PTR
#endif

//XR_DEFINE_HANDLE: a real pointer on 64-bit, a uint64 elsewhere. All our targets (arm64,
//x86_64) take the first branch; the second exists so a 32-bit build would still compile.
#if INTPTR_MAX == INT64_MAX
#define XR_DEFINE_HANDLE(object) typedef struct object##_T *object;
#define XR_NULL_HANDLE nullptr
#else
#define XR_DEFINE_HANDLE(object) typedef uint64_t object;
#define XR_NULL_HANDLE 0
#endif

XR_DEFINE_HANDLE(XrSession)
XR_DEFINE_HANDLE(XrSpace)

typedef int32_t XrResult;
typedef int64_t XrTime;         //nanoseconds; CLOCK_MONOTONIC on Android/Quest
typedef uint64_t XrFlags64;
typedef XrFlags64 XrSpaceLocationFlags;

//Every XrResult >= 0 is a success code (XR_SUCCESS is 0), every negative one an error.
#define XR_SUCCEEDED(result) ((result) >= 0)

//XrStructureType values used below.
#define XR_TYPE_REFERENCE_SPACE_CREATE_INFO 37
#define XR_TYPE_SPACE_LOCATION 42

//XrReferenceSpaceType. VIEW is the head: origin between the eyes, +Y up, -Z forward. Every
//runtime must support it, so it never needs an xrEnumerateReferenceSpaces check.
#define XR_REFERENCE_SPACE_TYPE_VIEW 1

//XrSpaceLocationFlagBits. VALID means the corresponding half of `pose` may be read at all;
//TRACKED additionally means it is currently being measured rather than extrapolated from the
//last known state (i.e. VALID && !TRACKED == tracking loss, pose is a guess).
#define XR_SPACE_LOCATION_ORIENTATION_VALID_BIT 0x00000001
#define XR_SPACE_LOCATION_POSITION_VALID_BIT 0x00000002
#define XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT 0x00000004
#define XR_SPACE_LOCATION_POSITION_TRACKED_BIT 0x00000008

typedef struct XrVector3f {
	float x;
	float y;
	float z;
} XrVector3f;

typedef struct XrQuaternionf {
	float x;
	float y;
	float z;
	float w;
} XrQuaternionf;

typedef struct XrPosef {
	XrQuaternionf orientation;
	XrVector3f position;
} XrPosef;

//`type` and `referenceSpaceType` are C enums in the real header, i.e. int-sized -- int32_t
//reproduces that layout exactly (the trailing padding after `type` on LP64 is implicit and
//identical either way).
typedef struct XrReferenceSpaceCreateInfo {
	int32_t type;
	const void *next;
	int32_t referenceSpaceType;
	XrPosef poseInReferenceSpace;
} XrReferenceSpaceCreateInfo;

typedef struct XrSpaceLocation {
	int32_t type;
	void *next;
	XrSpaceLocationFlags locationFlags;
	XrPosef pose;
} XrSpaceLocation;

typedef XrResult(XRAPI_PTR *PFN_xrCreateReferenceSpace)(XrSession session, const XrReferenceSpaceCreateInfo *createInfo, XrSpace *space);
typedef XrResult(XRAPI_PTR *PFN_xrLocateSpace)(XrSpace space, XrSpace baseSpace, XrTime time, XrSpaceLocation *location);
typedef XrResult(XRAPI_PTR *PFN_xrDestroySpace)(XrSpace space);
