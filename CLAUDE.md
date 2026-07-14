# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Godot 4.6 C++ GDExtension (`opencv_aruco`) that statically links OpenCV (via Conan) to detect ArUco markers and compute their 6DoF poses, plus a Godot test project that renders a mesh on each detected marker. The main deployment target is a Meta Quest 3 (Android arm64, OpenXR passthrough AR); Windows desktop is the development platform.

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

Data flows: camera frame (GodotAndroidCamera plugin on Quest, CameraServer on desktop) → GDScript worker thread → C++ OpenCV detection → camera-space marker poses → baked to world space using the head pose at the frame's capture time.

**C++ side (`src/`)** — `OpenCVProcessor` (a `Node`) exposes three detection entry points; the one that matters is `get_6dof_of_all_aruco_patches_from_godot_image(image, marker_size, downscale, intrinsics, distortion, lens_pose)`, which takes a frame the Godot CameraServer already owns (avoids a second capture). It runs `ArucoDetector` (DICT_4X4_50) + `solvePnP` and returns a Dictionary of marker id → `Transform3D` in camera space (OpenCV→Godot basis change = 180° X-flip baked in). On Android it also reads the Quest passthrough camera intrinsics via the NDK Camera2 API — the Quest's world-facing passthrough cameras are ids "50"/"51".

**GDScript side (`project/main_3d.gd`)** — scene nodes named `aruco_patch<id>` under `XROrigin3D` are matched to marker ids. Detection (~80 ms on Quest) runs on a background `Thread` with a single frame slot (newest frame wins); `get_image()` and all scene-tree writes stay on the main thread. Marker poses are baked to world space with the head pose at the frame's capture time (timestamped pose history, interpolated; capture time = the plugin's sensor timestamp on Quest, now − `CAMERA_LATENCY_MS` on desktop) — pairing old pixels with the live head pose makes patches "swim". `xr_startup.gd` initialises OpenXR and Meta passthrough (alpha blend + transparent viewport). A TCP frame-streamer (`tools/tcp_receiver.py`, port 7007 via `adb reverse`) exists for debugging.

**Camera backends** — Android/Quest uses the `GodotAndroidCamera` Android plugin (`project/addons/GodotAndroidCamera`, a CameraX AAR): frames arrive as raw Y-plane bytes on the CPU (no GPU→CPU readback) together with the **sensor timestamp** of each frame, which is mapped onto Godot's clock (`get_clock_offset_nanos`) to look up the head pose at the true capture time. If the plugin singleton is missing from the APK, the script falls back to Godot's native CameraServer (4.5+) with the fixed `CAMERA_LATENCY_MS` guess. Windows desktop needs the `CameraServerExtension` addon, instantiated via `ClassDB.instantiate()` with an untyped var so the script still parses on Android where that class doesn't exist. Rebuilding the AAR: `./gradlew build` in `../godot-android-camera`, then copy `plugin/demo/addons/GodotAndroidCamera` over `project/addons/GodotAndroidCamera`.

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
