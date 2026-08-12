# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Godot 4.7 C++ GDExtension (`opencv_aruco`) that statically links OpenCV (via Conan) to detect ArUco markers and compute their 6DoF poses, plus a Godot test project that renders a mesh on each detected marker. The main deployment target is a Meta Quest 3 (Android arm64, OpenXR passthrough AR); Windows desktop is the development platform.

## Build commands

There is no test suite or linter. Verification = build the extension, run the Godot project (desktop) or export + sideload the APK (Quest).

```powershell
# Windows native (DLL -> project/bin/windows/)
scons                                   # defaults to platform=windows target=template_debug

# Android arm64 for Quest 3 (.so -> project/bin/android/)
$env:CONAN_HOME = "C:\c"                # space-free Conan cache — REQUIRED on this machine
$env:ANDROID_HOME = "C:\asdk"           # junction to the real Android SDK — REQUIRED on this machine
scons platform=android arch=arm64 target=template_debug
```

- This machine's user path contains a space (`C:\Users\Paul Geiger`), which breaks Autotools deps and godot-cpp archiving — hence the two env vars above (README section 5b.4). Keep `ANDROID_HOME` consistent between builds; switching between junction and real path forces a full rebuild.
- First build compiles OpenCV from source through Conan (~10–20 min); later builds hit the Conan cache. Per-target Conan output lives in `conan_install/<platform>.<arch>.<target>/`; delete that folder or touch `conanfile.py` to force a dependency re-resolve.
- `SConstruct` bootstraps Conan itself (pip install + `conan profile detect`) — no manual Conan setup needed. For Android it uses the checked-in Jinja profile `profiles/android-arm64` (NDK path derived from `ANDROID_HOME`); `CONAN_HOST_PROFILE` overrides it.
- APK export happens in the Godot editor on Windows (Project > Export, preset "Meta Quest 3"). The editor does NOT build the Android `.so` — it must already exist in `project/bin/android/` before exporting. Sideload: `adb install -r <path>.apk`; stop the app: `adb shell am force-stop de.unigreifswald.opencvaruco`.

## Naming invariant

The library name `opencv_aruco` and entry symbol `opencv_aruco_library_init` must stay consistent across three places: `SConstruct` (`libname`), `project/bin/opencv_aruco.gdextension`, and `src/register_types.cpp`. godot-cpp composes filenames as `<lib?><name>.<platform>.<target>[.double].<arch><suffix>` — note `.double` comes BEFORE the arch, and Windows DLLs have no `lib` prefix while Linux/Android `.so`s do (the Android SCons branch forces `SHLIBPREFIX="lib"` because Windows hosts default to no prefix).

## Architecture

Data flows: Godot CameraServer frame (GPU→CPU readback) → GDScript `WorkerThreadPool` task → C++ OpenCV detection → world-space marker poses, baked in C++ with the head pose at the frame's capture time.

**C++ side (`src/`)** — `OpenCVProcessor` (a `Node`) exposes three detection entry points; the one that matters is `get_6dof_of_all_aruco_patches_from_godot_image(image, marker_sizes, default_marker_size, downscale, intrinsics, distortion, lens_pose, corners_out)`, which takes a frame the Godot CameraServer already owns (avoids a second capture). It runs `ArucoDetector` (DICT_4X4_50) + `solvePnP` and returns a Dictionary of marker id → `Transform3D`, pre-multiplied by the caller-supplied pose (GDScript passes head-pose-at-capture × lens pose, so results are world space; identity yields raw camera space — OpenCV→Godot basis change = 180° X-flip baked in). Object points are built per marker: `marker_sizes` is the id → side-length table, ids missing from it (or with a non-positive entry) fall back to `default_marker_size`, so differently sized markers solve correctly within one frame. `corners_out` is a debug-only out-parameter (Godot Dictionaries are shared references): it receives id → the 4 detected pixel corners, scaled back out of the downscaled detection frame, for the TCP overlay. On Android it also reads the Quest passthrough camera intrinsics via the NDK Camera2 API — the Quest's world-facing passthrough cameras are ids "50"/"51".

**GDScript side (`project/main_3d.gd`)** — patch nodes are created and freed at runtime, never authored in the scene: the first detection of a marker id spawns an `aruco_patch<id>` `Node3D` (one shared `BoxMesh`, scaled to that id's real marker size) under `XROrigin3D`, and an id absent from detection results for `PATCH_LOST_TIMEOUT_MS` (500 ms — a grace period, since detection drops markers for the odd frame) gets freed. The `marker_nodes` dictionary is the id → node registry and the thing that keeps ids unique: `_get_or_create_patch` returns the existing node for a known id, so no id can gain a second node or inherit another's. Both creation and pruning happen in `_apply_detection_result` on the main thread. Detection (~80 ms on Quest) runs as one-shot `WorkerThreadPool` tasks (at most one in flight, collected via `is_task_completed` polling); `get_image()` and all scene-tree writes stay on the main thread. Marker poses are baked to world space with the head pose at the frame's capture time (timestamped pose history, interpolated; capture time = now − `CAMERA_LATENCY_MS`, sampled at readback and passed into the C++ call combined with the lens pose — calibration values are `@export`ed, `_lens_pose` is derived once in `_ready`) — pairing old pixels with the live head pose makes patches "swim". `xr_startup.gd` initialises OpenXR and Meta passthrough (alpha blend + transparent viewport). A TCP frame-streamer (`tools/tcp_receiver.py`, port 7007 via `adb reverse`) exists for debugging: it sends the frame plus that detection's pixel corners as one packet and the receiver draws them with `cv2.aruco.drawDetectedMarkers`, which is why the send happens in `_poll_detection_task` (frame held in `_stream_img`) and not at readback time — the corners do not exist until the worker is done.

**Camera backends** — Android/Quest uses Godot's native CameraServer (4.5+). Windows desktop needs the `CameraServerExtension` addon, instantiated via `ClassDB.instantiate()` with an untyped var so the script still parses on Android where that class doesn't exist. Frames are PULLED via `CameraTexture.get_image()`, a GPU→CPU stall on the main thread, so `_process` reads back only when no detection task is in flight — declining to pull *is* the frame drop, which is why this branch has no pending-frame slot. This is the one place the `android-camera-plugin` branch deliberately differs: there frames are PUSHED from a CameraX plugin with a sensor timestamp and no readback, so it parks frames in a pending slot and derives the capture time from the timestamp instead of the fixed `CAMERA_LATENCY_MS` guess.

**Export guard (`project/addons/aruco_export_guard`)** — an `EditorExportPlugin` that fails the Android export if a required `.so` is missing for an enabled ABI. Without it Godot packages a 0-byte `.so` and reports success. It also deletes the broken APK in `_export_end()`. Keep its `ABI_ARCHS` map in sync with the `.gdextension` file.

## Platform gotchas (hard-won, do not regress)

- **Quest rendering must be Vulkan**: `rendering_method="mobile"` in `project.godot`. The GL renderer crashes on the first frame with "Failed to import swapchain IPC textures".
- **Android API level is pinned to 24** in `SConstruct` (godot-cpp defaults to 21) because OpenCV's videoio needs `libcamera2ndk`/`libmediandk`, which only exist in the NDK sysroot from API 24+. The Android branch also explicitly links `camera2ndk mediandk android log`.
- **C++ exceptions are enabled** (`disable_exceptions=False`) — OpenCV headers require them; godot-cpp disables them by default.
- **Windows CRT matching**: Conan's `compiler.runtime` is derived from godot-cpp's `use_static_cpp` (→ `/MT`). Never link `*d.lib` debug variants of third-party libs — godot-cpp's `template_debug` still uses the release CRT.
- **Quest hand-tracking**: a "controllers required" dialog on launch means the APK lacks the hand-tracking feature — enable Meta Hand Tracking (Optional) in the export preset.
- **Camera feed selection**: Quest exposes feed "1 | FRONT" plus the passthrough pair "50 | BACK"/"51 | BACK"; pick a "BACK" feed, and `CameraServer.monitoring_feeds = true` plus `feed.set_format(...)` must happen before `set_active(true)` or no frames arrive.
- NDK version `23.2.8568313` is pinned by godot-cpp; the Conan profile's `compiler.version=12` matches its clang. OpenXR Vendors plugin v5 lives in `project/addons/godotopenxrvendors`.

Comments in `SConstruct`, `conanfile.py`, and parts of the C++ are in German; that's the existing style, keep edits consistent with the file you're in.

The README documents the full Windows + WSL + Quest workflow in detail (tracks 1–4 plus the native-Windows alternative 5b); point users there rather than re-deriving setup steps.
