# Thermo-Nuclear Code Quality Review — `android-camera-plugin` (second pass)

**Scope:** everything on `android-camera-plugin` that is not on `cameraserver-readback`, as committed
at `0dc4137`. The working tree is clean apart from this file, so every citation below is against
committed code and every line number was checked at the time of writing — the previous pass of this
document had drifted from the tree it described, which is the one failure a review cannot afford.

**Verdict: closer, but still do not merge as-is.** The previous pass named nine structural problems;
seven are genuinely fixed, and the two node extractions plus the length-prefixed wire format left the
codebase materially better than a rearrangement would have. What remains is smaller but sharper: one
real data race that the last pass identified and the fix then *missed*, a debug default that shipped
in the product configuration (fixed in `6c5d155` while this document was being written), and a
cluster of missing models — three parallel dictionaries here, twelve loose accumulators there, ten
fields for one in-flight job — that are the reason several functions cannot be read without holding
the whole file in your head.

So the merge bar is now **one three-line change** ([B1](#b1--rebuild_distortion-still-races-the-worker-thread-and-the-fix-went-to-the-wrong-variable)),
plus whatever of §1–§4 is worth taking before the branch lands rather than after.

Measured facts this review rests on:

| file | total | comment | code | note |
|---|---|---|---|---|
| `project/open_cv_processor.gd` | **930** | 422 | 425 | was 1496 before the split |
| `project/detection_diagnostics.gd` | 507 | 237 | 224 | now committed |
| `project/tcp_debug_stream.gd` | 414 | 205 | 161 | now committed |
| `src/OpenCVProcessor.cpp` | 660 | 186 | 383 | |
| `src/OpenCVProcessor.h` | 240 | **148** | 66 | **2.24 comment lines per code line** |
| `src/OpenXRHeadLocator.cpp` | 209 | 34 | 153 | still the cleanest file in the diff |
| `tools/tcp_receiver.py` | 481 | — | — | 154-line module-level loop remains |
| `tools/plot_marker_pos.py` | 727 | — | — | |
| `tools/plot_reproj_log.py` | 415 | — | — | |
| `tools/plotlib.py` | 164 | — | — | new, shared |
| branch diff vs merge base | +6099 / −695 | | | code files only |

**Fixed since the last pass, and worth recording so nobody re-litigates it:** the 1496-line file
(split, and the split is now committed rather than one `git clean` from oblivion); the three
competing calibration sources and their A/B booleans (deleted, the winning numbers promoted to
defaults, the reasoning kept as a comment); `PATCH_LOST_TIMEOUT_MS` (back to 500 ms, comment and
`CLAUDE.md` agreeing); the two dead `solvePnP` entry points and `cv::VideoCapture`; the frame
protocol's five prose invariants (now one length field); the hand-rolled big-endian encoding; the
duplicated plot infrastructure; the untyped `Node` host references; the orphan `main_3d.tscn`; and
the image-overwrite hazard in the receiver.

---

## 0. Blockers

### B1 — `rebuild_distortion()` still races the worker thread, and the fix went to the wrong variable

This was the previous pass's B3. The remedy it prescribed — snapshot into a local before the OpenCV
call — **was applied, but only to the value where a torn read is harmless, and not to the one where
it is a use-after-free.**

[src/OpenCVProcessor.cpp:437-440](src/OpenCVProcessor.cpp#L437) now reads:

```cpp
// EINMAL in eine lokale Kopie lesen: der (Remote-)Inspector schreibt camera_intrinsics auf dem
// Hauptthread, waehrend diese Detektion im Worker laeuft -- vier getrennte Zugriffe koennten
// dann zwei Kalibrierungen mischen.
const Vector4 K_active = camera_intrinsics;
```

That is correct and the reasoning is right. `Vector4` is four floats; the worst case is one frame
computed from two calibrations. Now compare `distort_mat`, sixty lines further down:

```cpp
bool ok2 = cv::solvePnP(
    obj_pts, corners[i], Kamera_matrix, distort_mat,   // <- the member itself, not a copy
```

([src/OpenCVProcessor.cpp:504-508](src/OpenCVProcessor.cpp#L504)). `cv::Mat` is a refcounted handle.
`set_camera_distortion` ([:130](src/OpenCVProcessor.cpp#L130)) calls `rebuild_distortion()`, which
does `distort_mat = cv::Mat(n, 1, CV_32F)` ([:114](src/OpenCVProcessor.cpp#L114)) — dropping the last
reference to the old buffer and freeing it — on the **main** thread, while the worker is inside
`solvePnP` holding an `InputArray` view onto that same object. That is a read of freed memory, not a
mixed calibration, and the header at [:46-49](src/OpenCVProcessor.h#L46) explicitly promises the
opposite: *"A torn read costs one frame's markers, never consistency."* True for every member it
lists; not true for the one derived member it does not.

The window is narrow (an inspector edit landing inside a ~20 ms `solvePnP`), but editing calibration
live in the remote inspector is precisely the workflow this class was restructured to enable — the
header sells it twice. A crash that only happens while you are tuning is the worst possible schedule
for one.

**Remedy, unchanged and still three lines:** take the copy at the top of `detect_and_solve_all` and
of `project_marker_corners`, exactly as `K_active` already does — `const cv::Mat dist = distort_mat;`
— and pass `dist`. The copy is a refcount bump, so the buffer cannot be freed under the call. Do
`lens_pose` at the same time ([:408](src/OpenCVProcessor.cpp#L408),
[:565](src/OpenCVProcessor.cpp#L565)): a torn `Transform3D` is benign, but a snapshot is free and it
makes the rule uniform instead of a judgement call per member. Then the header's threading paragraph
becomes true as written.

---

### B2 — The shipped scene has the TCP debug streamer enabled

> **FIXED in `6c5d155`,** between this review being written and being committed: the `enabled = true`
> override is gone from the scene, so the node falls back to its own default. The finding is kept
> here because it had survived one review already, and because the shape of it — code and
> configuration disagreeing, with the comment being the half that gets believed — is worth
> recognising the next time a debug switch is flipped "just for this session".

At the time of review, [project/aruco_markers.tscn:25](project/aruco_markers.tscn#L25) read:

```
[node name="TcpDebugStream" type="Node" parent="OpenCVProcessor" unique_id=1900282746]
script = ExtResource("3_tcpds")
enabled = true
```

against the node's own default and its own stated reason
([project/tcp_debug_stream.gd:23-29](project/tcp_debug_stream.gd#L23)):

> *OFF by default, because a streamer nobody is listening to still costs a connect attempt every
> second for the whole session (and a log line with it), and on device that is the normal case.*

An APK exported from that scene attempts a `127.0.0.1:7007` connection once per second for the entire
session, and the moment anything answers it starts encoding and buffering every detected frame. The
code was right and the configuration contradicted it — which is worse than either alone, because the
next reader trusts the comment. It survived the previous review, which named it.

While in there: `handeye_capture` and `record_global_marker_pos` are correctly absent from the scene
(so they default off). The streamer is the only one that escaped.

---

## 1. Missed code judo — complexity a reframing deletes rather than rearranges

### J1 — Three parallel dictionaries keyed by marker id, and the coupling they force

[project/open_cv_processor.gd](project/open_cv_processor.gd) keeps one marker's state in three
separate dictionaries declared 215 lines apart:

| dict | line | holds |
|---|---|---|
| `marker_nodes` | [:15](project/open_cv_processor.gd#L15) | id → patch `Node3D` |
| `_marker_last_seen` | [:20](project/open_cv_processor.gd#L20) | id → last-seen µs |
| `_marker_poses` | [:230](project/open_cv_processor.gd#L230) | id → world pose |

Each carries its own comment explaining how its lifetime differs from the others', and those comments
are correct and hard-won — the poses and timestamps deliberately outlive the node. But "three
dictionaries with three different pruning rules" is the *implementation* of one idea, `Marker`, and
writing it as three is what produces the rest of this finding:

- `_apply_detection_result` writes all three in three statements
  ([:796-798](project/open_cv_processor.gd#L796)) with nothing tying them together;
- the prune loop iterates `marker_nodes.keys()` and must explain, in six lines of comment
  ([:826-831](project/open_cv_processor.gd#L826)), why it deliberately does *not* erase from the
  second one;
- and `_sync_marker_sizes` has to reach into a node's child by index to find the mesh:

```gdscript
var mesh_instance: MeshInstance3D = marker_nodes[id].get_child(0)
```

([:318](project/open_cv_processor.gd#L318)), with a comment naming the function 434 lines away that
put it there. Two functions coupled by child index is a design problem, not a nit: anyone adding a
second child to a patch — a label, a gizmo, an outline — silently changes what `_sync_marker_sizes`
scales.

**The judo move goes further than "make it a record".** Look at what `_sync_marker_sizes` is *for*:
it pushes a size onto a mesh that already exists, because the size might have changed in the
inspector since the node was built. It is called twice — once from `_ready`
([:348](project/open_cv_processor.gd#L348)), where its own comment admits there are no patches yet
and nothing can happen, and once per detection task from `_start_detection_task`
([:738](project/open_cv_processor.gd#L738)). But `_apply_detection_result` **already assigns
something to every live marker every detection** ([:797](project/open_cv_processor.gd#L797)). Assign
the scale there, beside the pose, and the whole function disappears: the `is_equal_approx` guard and
its explanation of float32 comparison, both call sites, the no-op call in `_ready`, the `get_child(0)`
reach, and the paragraph in `CLAUDE.md` describing when it may run. The cost is one `Vector3`
assignment per marker per detection instead of a compare-then-maybe-assign — at a dozen markers and
12 detections a second, nothing.

So: hold `{node, mesh, pose, last_seen}` per id (or keep the three dictionaries and just store the
`MeshInstance3D` instead of its parent), assign size beside pose, and delete `_sync_marker_sizes`
outright. That is roughly 25 lines of code and 20 of comment gone, and one cross-function convention
with it.

### J2 — Ten fields and a redundant boolean for "one job in flight"

[project/open_cv_processor.gd:142-157](project/open_cv_processor.gd#L142) declares the detection
worker's state as two five-field groups plus a task id:

```gdscript
var _pending_image: Image
var _pending_capture_usec := 0
var _pending_cam_xform := Transform3D.IDENTITY
var _has_pending := false
...
var _result_markers: Dictionary = {}
var _result_corners: Dictionary = {}
var _result_capture_usec := 0
var _result_cam_xform := Transform3D.IDENTITY
```

plus `_pending_frame_id` / `_result_frame_id` ([:193-194](project/open_cv_processor.gd#L193)) and
`_detecting_img` ([:200](project/open_cv_processor.gd#L200)). Twelve fields for two instances of the
same four-or-five-tuple, which then have to be copied field by field between slots
([:687-693](project/open_cv_processor.gd#L687), [:722-728](project/open_cv_processor.gd#L722),
[:874-878](project/open_cv_processor.gd#L874)) — three hand-written copy blocks that must stay in
sync, and a fourth shape (`_detecting_img`) that holds one field of the tuple separately because the
streamer needs it after the worker returns.

`_has_pending` is pure redundancy: it is true exactly when `_pending_image != null`, and
[:726-727](project/open_cv_processor.gd#L726) sets both together. A field whose value is derivable
from another field is a second source of truth for the same fact.

A tiny `DetectionJob` (image, capture_usec, cam_xform, frame_id, and on the way back markers +
corners) turns all three copy blocks into single assignments, makes `_has_pending` `_pending != null`,
and lets `_detecting_img` be `_running.image` — which is also a better statement of what it is. The
threading comments stay valid word for word; the memory barrier argument is about *when* the slot is
read, not how many variables it has.

### J3 — The flow trace is the last diagnostic still woven through the app script

`DEBUG_FLOW` puts `if traced: print(...)` at ten stations across five functions:
[:546](project/open_cv_processor.gd#L546), [:550](project/open_cv_processor.gd#L550),
[:557](project/open_cv_processor.gd#L557), [:586](project/open_cv_processor.gd#L586),
[:676](project/open_cv_processor.gd#L676), [:694](project/open_cv_processor.gd#L694),
[:700](project/open_cv_processor.gd#L700), [:729](project/open_cv_processor.gd#L729),
[:779](project/open_cv_processor.gd#L779), [:800](project/open_cv_processor.gd#L800),
[:848](project/open_cv_processor.gd#L848), [:869](project/open_cv_processor.gd#L869) — roughly 60
lines of multi-line format strings interleaved with the logic, several of them in the hot path.

The facility is good. The "same frame id at every station, so one frame's journey reads as a block"
design is the right idea and I would not lose it. But this branch has now performed the same
extraction twice, with good results both times, and stopped one facility short of finishing the job.
Two thirds of the remaining prose in `_on_android_camera_frame` is trace text. Either a
`_trace(frame_id, station, msg)` helper that no-ops on the flag (collapsing each site to one line) or
a `FlowTrace` child node in the style of the other two. Given that `DetectionDiagnostics` already
exists and already receives a per-frame `sample()` call, the trace stations are the natural third
tenant of it — the only obstacle is that they fire from inside functions the node does not see, which
a single `trace(id, station, text)` method solves.

Leaving this one in is now an inconsistency rather than a judgement call: the file's own structure
says diagnostics live in child nodes.

### J4 — The camera backend is still eight nullable fields and a mid-function early return

[project/open_cv_processor.gd:83-104](project/open_cv_processor.gd#L83) carries, for two mutually
exclusive backends: `camera_extension`, `cam_texture`, `android_camera`, `_android_cam_started`,
`_cam_clock_offset_ns`, `_cam_ts_realtime`, `_preview_texture`, `_xr_stamp_poses`. Which path is live
is decided by null-checking one of them in the middle of `_process`
([:663-664](project/open_cv_processor.gd#L663)):

```gdscript
if cam_texture == null:
    return
```

sitting between "poll the detection task" and "read the desktop camera". So `_process` means two
different things depending on a field assigned 200 lines away, and a reader has to know which
platform they are on to know which half of the function applies. This is the last "weird if in a
random place" in the file.

A `CameraSource` with two implementations (`AndroidPluginSource`, `CameraServerSource`), each owning
its own clock bridge and preview handling and emitting `(image, capture_usec, head_pose)`, removes
all eight fields from the host and leaves one path. That is the largest change this document
proposes and the one that would take the app script under 500 lines — but note that the branch has
already proved the pattern twice with the diagnostic nodes, so the risk is lower than it looks. At
minimum, make the decision explicit where it is made: `if _using_camera_server:` around the desktop
half, so the platform branch is visible rather than implied by a null.

### J5 — `tcp_receiver.py` still runs 154 lines at module level over five globals

`read_frame()` fixed the framing, which was the important half. What is left is the display loop:
[tools/tcp_receiver.py:315](tools/tcp_receiver.py#L315) opens a `while True:` that runs to
[:468](tools/tcp_receiver.py#L468) over `log_file`, `log_path`, `pos_file`, `pos_path` and
`frame_num` ([:308-313](tools/tcp_receiver.py#L308)), and does five things: the position-log edge
follow, the position-log write, the overlay drawing, the status line, the corner-log write, and the
keyboard handling. The socket itself is set up at import time
([:87-92](tools/tcp_receiver.py#L87)), which is why the module cannot be imported for a test without
side effects.

The two log facilities are the natural extraction and they are more alike than the previous review
credited — not identical (one follows a device edge, one a keypress; the row formats differ), but
both are "open a stamped file, write a header, append per frame, flush, close and announce". A
`StampedCsv` with `open_if(cond)` / `write(rows)` / `close()` collapses
[:341-354](tools/tcp_receiver.py#L341) and [:419-439](tools/tcp_receiver.py#L419) and the two
end-of-run blocks at [:470-478](tools/tcp_receiver.py#L470), and takes the four `*_file`/`*_path`
globals with it. What remains is a `main()` with a socket, a loop and two objects.

### J6 — `figure_diagnostics` is 141 lines and three unrelated figures

[tools/plot_reproj_log.py:198-339](tools/plot_reproj_log.py#L198) builds, in one function: the
gap-vs-displacement scatter with binned medians, the per-axis attribution with its least-squares fit
and printed diagnosis, and the cross-marker correlation panel with its own printout — switching
panel count on `has_axes` and addressing the last panel as `axes[-1]` because the index depends on
that switch.

It also contains the one piece of genuinely fragile control flow left in the tools:

```python
if len(ids) >= 2:
    ...
    common = sorted(set(ga) & set(gb))
    a = np.array([ga[k] for k in common])
    b = np.array([gb[k] for k in common])
if len(ids) >= 2 and len(common) > 0:
```

([:302-307](tools/plot_reproj_log.py#L302)). `common`, `a` and `b` exist only if the first branch
ran; the second condition is correct today *only* because `and` short-circuits in that order.
Reorder those two clauses — an entirely reasonable edit — and it is a `NameError` on every
single-marker log.

The decomposition applied to `plot_marker_pos.py` in the last pass (`report_*` functions, one figure
list) was not applied here, and this is the file that needed it more. Three builders —
`panel_strafe`, `panel_axis_fit`, `panel_cross_marker` — each returning its printout, composed by a
`figure_diagnostics` that decides which panels exist. The `common`/`a`/`b` problem disappears with
the scope.

---

## 2. Spaghetti and branching complexity

### S1 — `pose_source` is a display string used as control flow

[project/open_cv_processor.gd:566-585](project/open_cv_processor.gd#L566):

```gdscript
var pose_source := "history"
if use_xr_locate_space and _head_locator != null:
    ...
    pose_source = "xrLocateSpace" if loc.get("tracked", false) else "xrLocateSpace_untracked"
    ...
if pose_source == "history":
    cam_xform = _head_pose_at(lookup_usec)
```

A label whose only other job is the trace line at [:587](project/open_cv_processor.gd#L587) decides
which head-pose source runs. Renaming a string for the log, or adding a fourth source name, silently
changes which branch executes. Named in the last pass, unchanged. The fallback wants its own
`var located := false`; the string stays for the trace and becomes inert.

### S2 — `_on_android_camera_frame` is 80 lines with five jobs

[project/open_cv_processor.gd:527-607](project/open_cv_processor.gd#L527) does clock mapping with a
two-stage retry and fallback ([:531-541](project/open_cv_processor.gd#L531)), flow tracing, the head
pose lookup with its own fallback ([:565-585](project/open_cv_processor.gd#L565)), periodic
statistics ([:590-593](project/open_cv_processor.gd#L590)), preview-texture maintenance
([:600-605](project/open_cv_processor.gd#L600)) and dispatch. Three of the five are diagnostics.

The clock block — with its duplicated plausibility test, the same three-condition expression written
twice — is a self-contained `_capture_time_usec(timestamp_ns)`. The pose lookup is
`_head_pose_for(timestamp_ns, lookup_usec)`, and taking it out also fixes S1 for free, since a
function can `return` instead of leaving a label behind to be re-tested. That leaves a ~20-line
arrival handler that reads top to bottom.

### S3 — Twelve accumulators for one statistic, reset in two different places

[project/detection_diagnostics.gd:296-307](project/detection_diagnostics.gd#L296) declares
`_pdt_prev`, `_pdt_prev_now_usec`, `_pdt_deltas`, `_pdt_dupes`, `_pdt_advance_ns`,
`_pdt_wall_advance_usec`, `_pdt_delta_min_ns`, `_pdt_delta_max_ns`, `_lead_sum_ms`, `_lead_min_ms`,
`_lead_max_ms`, `_lead_print_timer` — all for the once-a-second `openxr pdt sync` line.

The window reset at [:402-406](project/detection_diagnostics.gd#L402) clears **five** of them. The
other four — the two min/max pairs — are re-seeded somewhere else entirely, by a sentinel test
`if _pdt_deltas == 0` at [:374](project/detection_diagnostics.gd#L374) that infers "this is a fresh
window" from a counter someone else zeroed. So the window's state is half-reset explicitly and
half-reset implicitly, and the two halves are 30 lines apart. That is exactly the non-atomic update
the standards call out: add a sixth accumulator and there are now two places to remember, one of
which does not look like a reset at all.

A `Window` helper with `add(delta_ns, lead_ms)` / `summary()` / `reset()` — or even just a plain
dictionary rebuilt wholesale, `_window = _fresh_window()` — makes the reset one statement that cannot
be partial, and the min/max seeding becomes ordinary first-sample handling inside `add`.

The same shape appears one section up: `_drift_ref`, `_drift_seed`, `_drift_rot`
([:186-192](project/detection_diagnostics.gd#L186)) are three id-keyed dictionaries for one per-marker
reference record, with `_drift_seed` erased in one place and the other two never — the same missing
model as [J1](#j1--three-parallel-dictionaries-keyed-by-marker-id-and-the-coupling-they-force), in a
different file.

### S4 — `_ready()` documents four things that are not there, and calls one no-op

[project/open_cv_processor.gd:339-351](project/open_cv_processor.gd#L339): *"Nothing to derive here
any more"*, *"No patch nodes to find"*, *"Nothing to size yet either"*, *"No worker setup needed"*.
These are notes-to-self from the refactor that produced them. They describe the history of the
function rather than the program, and a reader six months out has to verify each one before trusting
it — which is the opposite of what a comment is for.

Worse, the third one is attached to a call that it simultaneously admits does nothing:

```gdscript
# Nothing to size yet either (no patches exist), but the call keeps the "sizes reach the meshes"
# path in one place; it re-runs before every detection task from _start_detection_task.
_sync_marker_sizes()
```

A call kept for tidiness, over an empty dictionary, justified by a comment saying so. Delete the
call and the four comments; keep only the ordering note at
[:353-360](project/open_cv_processor.gd#L353), which carries a real constraint (the locator must
exist before frames arrive) and belongs on the line it guards. [J1](#j1--three-parallel-dictionaries-keyed-by-marker-id-and-the-coupling-they-force)
removes the call anyway.

### S5 — `xr_startup.gd` prints outside the project's own logging convention

Every other file routes output through `debug_prints_enabled` and the fixed
`[opencv_aruco] [tag::function]` prefix, and `CLAUDE.md` documents that prefix as *the* way to find
this app in logcat. [project/xr_startup.gd](project/xr_startup.gd) prints unconditionally and with no
prefix at [:10](project/xr_startup.gd#L10), [:13](project/xr_startup.gd#L13),
[:25](project/xr_startup.gd#L25), [:35](project/xr_startup.gd#L35) and
[:37](project/xr_startup.gd#L37) — including a dump of every XR interface on every launch. So the
one file that reports whether passthrough came up is the one file `adb logcat | grep opencv_aruco`
cannot see, and it talks whether or not debug output is on.

Small, but it is a shared convention with one defector, and the defector is on the startup path where
you actually want the line. Route them through the same prefix; gate the two informational ones and
leave the failure branches ungated (as `ACV_ERR` already is on the C++ side).

The file also still says it initialises OpenXR ([:3](project/xr_startup.gd#L3)) when the engine does
— see [§5](#5-stale-comments-and-documentation-drift) — and carries three lines of commented-out
depth-extension code at [:18-20](project/xr_startup.gd#L18).

---

## 3. Boundary, type and contract problems

### T1 — `channels` inferred by division, with an unguarded `else` that reads out of bounds

[src/OpenCVProcessor.cpp:643-655](src/OpenCVProcessor.cpp#L643):

```cpp
int channels = (width * height > 0) ? (int)(data.size() / (width * height)) : 0;
...
if (channels == 1) { ... }
else if (channels == 4) { ... }
else { cv::Mat rgb(height, width, CV_8UC3, (void *)data.ptr()); ... }
```

Anything that is not 1 or 4 is *assumed* to be 3. A two-byte-per-pixel format (`FORMAT_RG8`,
`FORMAT_RH`, a 16-bit depth frame) builds a `CV_8UC3` header over a buffer holding two thirds of the
bytes `cvtColor` will read — an out-of-bounds read, not an error. A padded or strided plane whose
`data.size()` is not an exact multiple mis-derives the count silently.

Unreachable today (the two live feeds are `FORMAT_L8` and `FORMAT_RGBA8`), which is why it is here
and not in the blockers — but `Image` already carries the answer and the guess is strictly worse than
the fact. Switch on `image->get_format()`: `FORMAT_L8`/`FORMAT_R8` → gray, `FORMAT_RGB8` → RGB,
`FORMAT_RGBA8` → RGBA, anything else → `ACV_ERR` and an empty result. This is the "generic magic
hiding a simple data shape" rule exactly — the format enum *is* the shape, and dividing byte counts
to re-derive it discards information that was handed to us.

### T2 — `corners_out` is an out-parameter passed by value

[src/OpenCVProcessor.cpp:617](src/OpenCVProcessor.cpp#L617):

```cpp
Dictionary OpenCVProcessor::detect_markers(const Ref<Image> &image, const Transform3D &head_pose, Dictionary corners_out)
```

It works — Godot `Dictionary` is a shared reference — but the signature says the opposite of what
happens, and the header spends four lines explaining that
([src/OpenCVProcessor.h:208-209](src/OpenCVProcessor.h#L208)). The private
`detect_and_solve_all` takes `Dictionary &` ([:397](src/OpenCVProcessor.cpp#L397)), so the two halves
of one operation state opposite conventions eleven lines apart in the same header
([:152](src/OpenCVProcessor.h#L152) vs [:210](src/OpenCVProcessor.h#L210)).

Return one `Dictionary { "poses": …, "corners": … }` and let the single caller unpack it — the caller
already builds a fresh dictionary per detection ([project/open_cv_processor.gd:860](project/open_cv_processor.gd#L860))
and immediately parks both in slots, so unpacking costs nothing. Failing that, `const Dictionary &`
with a name that says shared. A contract that depends on a Godot implementation detail *which the
signature contradicts* is a trap for the next person who "fixes" the missing `&`.

### T3 — Dead state carried on the class

[src/OpenCVProcessor.h:155-159](src/OpenCVProcessor.h#L155):

```cpp
cv::Mat K_cam50;
cv::Mat K_cam51;
int current_camera_id = 50;
cv::Mat D;
bool intrinsics_ready = false;
```

Verified against the whole tree: `intrinsics_ready` is never read or written anywhere.
`current_camera_id` is referenced only from a commented-out block
([:450-457](src/OpenCVProcessor.cpp#L450)). `K_cam50`, `K_cam51` and `D` are written by
`dump_quest_camera_metadata` ([:314](src/OpenCVProcessor.cpp#L314),
[:318](src/OpenCVProcessor.cpp#L318), [:333](src/OpenCVProcessor.cpp#L333)) and never read — the
function's own doc comment says so in as many words ([:256](src/OpenCVProcessor.cpp#L256)). A dump
that logs should log; storing what nothing reads makes five members look like configuration.

Also dead: the `marker_pose_found` signal, declared at
[src/OpenCVProcessor.cpp:46](src/OpenCVProcessor.cpp#L46) and emitted nowhere in the repository. And
two commented-out blocks — [:450-457](src/OpenCVProcessor.cpp#L450) (the camera-selection
alternative) and [:625-635](src/OpenCVProcessor.cpp#L625) (an RGBA conversion the live code below
already performs).

### T4 — `ACameraManager` leaked on the one early exit

[src/OpenCVProcessor.cpp:266-269](src/OpenCVProcessor.cpp#L266):

```cpp
if (ACameraManager_getCameraIdList(mgr, &idList) != ACAMERA_OK) {
    ACV_ERR("ACameraManager_getCameraIdList failed");
    return;                       // <- mgr never deleted
}
```

One-shot and diagnostic-only, so the impact is a few hundred bytes once per launch — but it is the
only manual resource in the file and it has exactly one early exit, so the fix is one line and the
argument for leaving it is nil. The three `continue`s further down
([:278](src/OpenCVProcessor.cpp#L278), [:283](src/OpenCVProcessor.cpp#L283)) are currently safe
because they sit before `meta` is allocated — safety by position, one inserted statement away from
not being safe. A small RAII guard would make both properties structural.

### T5 — `set_image_downscale_factor` does not clamp; the use site does

[src/OpenCVProcessor.cpp:136](src/OpenCVProcessor.cpp#L136) stores the raw value;
[:416](src/OpenCVProcessor.cpp#L416) clamps it to `[0.05, 1.0]` per frame. Defensible in isolation
— and the comment correctly explains that scaling the intrinsics by the *clamped* value is the point
— but it is inconsistent with `camera_distortion` and the lens pose, whose setters derive
immediately, and it means the getter can report `0.01` while the detector runs `0.05`. Clamp in the
setter and the property, the derived intrinsics and the log line all agree by construction; one
fewer number that means two things depending on who asks.

### T6 — `_start_android_camera` proceeds with an empty camera id

[project/open_cv_processor.gd:487-491](project/open_cv_processor.gd#L487): if neither the passthrough
search nor the `back`-facing fallback matches, `cam_id` stays `""` and is passed straight to
`start_camera(...)` at [:513](project/open_cv_processor.gd#L513) with `_android_cam_started` already
`true` ([:477](project/open_cv_processor.gd#L477)). No log line, no retry, no error. The failure
presents as an app that runs perfectly and never detects anything — the most expensive kind to
diagnose on a headset. A `push_error` and an early return turns a silent no-frames deploy into one
line in logcat.

### T7 — Two dictionary shapes from one function

`locate_head` returns a five-key `Dictionary`
([src/OpenXRHeadLocator.cpp:126-188](src/OpenXRHeadLocator.cpp#L126)) — reasonable, since a bound
GDExtension method has no better way to hand a struct to GDScript, and every key is always present,
which is the property that makes it safe.

`_locate_delta` does not preserve that property
([project/detection_diagnostics.gd:412-427](project/detection_diagnostics.gd#L412)): the failure path
returns `{valid, result, flags}` and the success path returns `{valid, tracked, flags, pos_mm,
rot_deg, ps_origin}`. Two different shapes, distinguishable only by reading `valid` first, and
`_fmt_locate` ([:490](project/detection_diagnostics.gd#L490)) is the branch that knows it. The caller
then indexes `at_pdt["ps_origin"]` at [:483](project/detection_diagnostics.gd#L483) guarded by a
`valid` test written three lines earlier. Return all keys always, with zeros for the ones that have
no meaning — the same discipline `locate_head` itself follows, one layer down.

---

## 4. File size and decomposition

### D1 — `src/OpenCVProcessor.h`: 148 comment lines to 66 code lines

Unchanged at 2.24:1, and the prose is still genuinely good — the lens-pose derivation, the
`LENS_POSE_REFERENCE == GYROSCOPE` warning, the "learned the hard way" note about the stale literals,
and now the paragraph explaining why the A/B switch was deleted rather than kept. That last one is a
real improvement: it records a *decision* where a switch used to sit.

But a header is still the wrong container. It is what every translation unit includes and what every
reader pages through to find the interface, and the interface here is 66 lines that would fit on one
screen. Move the calibration essays to `docs/calibration.md` (or the top of the `.cpp`, which is
already where the Camera2 and basis-change discussions live) and leave a two-line pointer at each
property. The essays gain a place where they can be read as essays, with headings and an order of
their own.

### D2 — `OpenCVProcessor` is still four classes in a trench coat

At 660 + 240 lines it holds the detection pipeline, the calibration store and its derived values, an
Android Camera2 metadata dumper, and a debug reprojection utility. The dead entry points are gone,
which was the biggest of the five, and `detect_and_solve_all` is genuinely the only change of basis
left — that is real progress.

The remaining split that pays for itself is the Camera2 dump: `dump_quest_camera_metadata` is 122
lines ([:259-381](src/OpenCVProcessor.cpp#L259)), the only `#ifdef __ANDROID__` block, the only NDK
dependency, the sole reason the API-24 pin exists, and — once [T3](#t3--dead-state-carried-on-the-class)
removes `K_cam50`/`K_cam51`/`D` — it shares no state whatsoever with the rest of the class. A free
function in `src/QuestCameraMetadata.cpp` that logs and returns nothing. That also puts the platform
constraint and the code that needs it in the same file, which is where the next person will look for
it.

`detect_and_solve_all` itself is 137 lines ([:397-534](src/OpenCVProcessor.cpp#L397)) with six
`cv::getTickCount()` profiling sites threaded through it. The profiling is worth keeping — it is how
the ArucoNano swap was justified — but it would read better as a small scoped timer than as six
manual pairs.

### D3 — `CLAUDE.md` is 30 KB, and [line 44](CLAUDE.md#L44) is one paragraph of ~2,500 words

Unchanged, and now slightly worse: this pass added the wire-format and node-typing explanations to
the same paragraph, because there was nowhere else for them to go. That paragraph now covers the
patch lifecycle, the worker thread, the pose history, pdt stamping, `POSE_LOOKUP_TRIM_MS`,
`use_xr_locate_space`, the public marker API, `xr_startup.gd`, `DetectionDiagnostics`,
`TcpDebugStream`, the frame protocol and the plot tools.

A file loaded into context every session that nobody can scan is not documentation, it is ballast —
and it is now actively resisting its own maintenance, since every addition makes the next one harder
to place. Split into subsections with headings, one per component, so a change to `TcpDebugStream`
has an obvious home. This is cheap and it is the highest-leverage documentation change available.

---

## 5. Stale comments and documentation drift

Fewer than last time, and the two worst (the intrinsics that no longer matched, the timeout that had
silently gone 10×) are gone. What is left:

| location | says | reality |
|---|---|---|
| [project/xr_startup.gd:3](project/xr_startup.gd#L3) | "find the interface, **initialise it**, and switch the viewport" | it does not initialise; the engine does, before the main scene loads. `CLAUDE.md` states this correctly, the file does not |
| [tools/requirements.txt:2](tools/requirements.txt#L2) | "tcp_receiver writes `tools/images/`, plot_reproj_log reads it back" | writes `tools/saved_from_stream/`; the reader is `cameraCalibration.py`, not the plot script |
| [tools/requirements.txt:16](tools/requirements.txt#L16) | "matplotlib — `plot_reproj_log.py` only" | `plot_marker_pos.py` and `plotlib.py` need it too |
| [tools/handeye_solve.py:3](tools/handeye_solve.py#L3) | samples "written by `open_cv_processor.gd`'s `handeye_capture`" | written by `detection_diagnostics.gd` since the split |
| [tools/plot_reproj_log.py:35](tools/plot_reproj_log.py#L35) | "MUST match `camera_intrinsics` in `src/OpenCVProcessor.h`" | it *does* match today — but it is still four hand-copied constants with no enforcement, and the last time they diverged nothing said so |

That last row is the one worth acting on rather than just correcting. The invariant is real and the
enforcement is a comment. The intrinsics could travel in the reproj-log CSV header — the device
already knows them, the receiver already writes a header line, and then the plot uses the camera the
capture actually ran with instead of a copy that happens to agree. That deletes the invariant instead
of restating it.

---

## 6. Introduced by the last cleanup pass

Reviewing the changes made earlier in this session on the same terms as everything else.

- **`_NOTE_CORNERS` has four entries and two callers.**
  [tools/plotlib.py:135-140](tools/plotlib.py#L135) defines all four corners of a rectangle; only
  `"upper left"` (the default) and `"lower right"` are ever passed. That is speculative generality of
  exactly the kind flagged elsewhere in this document — two dead map entries added on the assumption
  that a future figure will want them. Keep the two that are used.
- **`process_marker` takes a parameter it can derive.**
  [tools/plot_marker_pos.py:658](tools/plot_marker_pos.py#L658) takes `sel_count` alongside `one`,
  but `sel_count` is `len(one["frame"])`. Six parameters where five would do, and one of them is a
  second statement of a fact the first already carries.
- **`axis_panel` has eight parameters.**
  [tools/plot_marker_pos.py:119](tools/plot_marker_pos.py#L119) —
  `(ax, frame, values, color, linestyle, label, unit, zero_line)`. This document's previous pass
  rejected a merged `figure_four_panel` partly on parameter count; the panel helper that replaced it
  is not far off. It is defensible (every argument is genuinely per-panel and there are four call
  sites, versus two for the figure version) but it is not obviously past the line, and `color` +
  `linestyle` are always `C_AXES[i]` + `S_AXES[i]` at three of the four sites — passing the index
  would collapse two arguments into one.
- **`_debug()` is still written twice.**
  [project/tcp_debug_stream.gd:111](project/tcp_debug_stream.gd#L111) and
  [project/detection_diagnostics.gd:59](project/detection_diagnostics.gd#L59) are now identical
  one-line forwards to the same C++ static. Harmless — that was the point of the static — but two
  copies of `return OpenCVProcessor.get_debug_prints_enabled()` is one copy too many; call it
  directly at the four use sites, or accept it and stop calling it a helper.
- **`read_frame` signals failure two ways.** `None` for "the stream is over or unrecoverable",
  `FrameError` for "this frame is bad but the stream is fine"
  ([tools/tcp_receiver.py:234-291](tools/tcp_receiver.py#L234)). Documented and deliberate, and the
  distinction is real, but a caller must handle both a return value and an exception to be correct.
  Acceptable as-is; worth knowing it is a two-channel contract.

---

## 7. What is good, and should not be touched

- **The two node extractions.** Three touch points, arguments handed over rather than read back, the
  whole facility removable by deleting a node. That is the right shape, executed cleanly, and it is
  the model the remaining extractions ([J3](#j3--the-flow-trace-is-the-last-diagnostic-still-woven-through-the-app-script),
  [J4](#j4--the-camera-backend-is-still-eight-nullable-fields-and-a-mid-function-early-return)) should copy.
- **The length-prefixed frame format.** Replacing an invariant that five comments defended with a
  field that makes violating it detectable is the single best structural change on the branch. The
  `seek(0)` patch-in — rather than computing the size from the marker counts — is the detail that
  makes it airtight, because it leaves no second description of the layout to disagree with the
  writer.
- **The cross-frame reprojection design.** Recognising that same-frame reprojection cancels
  algebraically, and building the latch and the travelling baselines around that, is genuinely subtle
  test design. The `REPROJ_BASELINE_MS` justification with its `fx · turn · error / range` arithmetic
  is how a constant should be chosen.
- **The `transform_from_pose()` warning** at
  [src/OpenXRHeadLocator.cpp:174-181](src/OpenXRHeadLocator.cpp#L174). A documented negative result
  that saves someone a device round trip. Keep it verbatim, including the "do not simplify this back".
- **`OpenXRHeadLocator` as a whole.** Session-lifetime handling via `view_space_session`, the
  rate-limited failure log, `ensure_space()` as the single warm/cold path, every dictionary key always
  present. 209 lines that do one thing with no leaks and no branching sprawl. Still the cleanest file
  in the diff.
- **`get_marker_size()` as the single id → metric-scale resolution**, used by `solvePnP`, by the
  rendered mesh and by the reprojection overlay. One function, so the three cannot drift.
- **Recording the deleted A/B switch as a paragraph of reasoning** at
  [src/OpenCVProcessor.h:57-68](src/OpenCVProcessor.h#L57). A measurement whose result is now in the
  code and whose apparatus is gone — which is what should have happened to it, and the comment says
  why the alternative was rejected. That is the pattern for every future "temporary" flag here.

---

## 8. Suggested order of work

**Before merge:**

1. Snapshot `distort_mat` (and `lens_pose`) into locals in `detect_and_solve_all` and
   `project_marker_corners`. Three lines; the header's threading paragraph becomes true.
   ([B1](#b1--rebuild_distortion-still-races-the-worker-thread-and-the-fix-went-to-the-wrong-variable))
2. ~~Turn `TcpDebugStream.enabled` off in the scene.~~ **DONE** (`6c5d155`).
   ([B2](#b2--the-shipped-scene-has-the-tcp-debug-streamer-enabled))

**High value, low risk:**

3. Switch `detect_markers` on `image->get_format()`.
   ([T1](#t1--channels-inferred-by-division-with-an-unguarded-else-that-reads-out-of-bounds))
4. Delete the five dead members, the dead signal and the two commented-out blocks.
   ([T3](#t3--dead-state-carried-on-the-class))
5. `push_error` + return on an empty `cam_id`; `ACameraManager_delete` on the failure path; clamp the
   downscale in its setter. ([T6](#t6--_start_android_camera-proceeds-with-an-empty-camera-id),
   [T4](#t4--acameramanager-leaked-on-the-one-early-exit), [T5](#t5--set_image_downscale_factor-does-not-clamp-the-use-site-does))
6. Fix the five stale comments in [§5](#5-stale-comments-and-documentation-drift), and put the
   intrinsics in the reproj-log CSV header so the fifth cannot recur.
7. Trim the four dead entries and parameters noted in
   [§6](#6-introduced-by-the-last-cleanup-pass).

**Structural, worth scheduling:**

8. The `Marker` record: store the mesh, assign size beside pose, delete `_sync_marker_sizes` and the
   `get_child(0)` coupling. ([J1](#j1--three-parallel-dictionaries-keyed-by-marker-id-and-the-coupling-they-force),
   [S4](#s4--_ready-documents-four-things-that-are-not-there-and-calls-one-no-op))
9. The `DetectionJob` record; drop `_has_pending`.
   ([J2](#j2--ten-fields-and-a-redundant-boolean-for-one-job-in-flight))
10. Extract `_capture_time_usec` and `_head_pose_for` out of `_on_android_camera_frame`; the second
    one removes the string-as-control-flow with it.
    ([S2](#s2--_on_android_camera_frame-is-80-lines-with-five-jobs), [S1](#s1--pose_source-is-a-display-string-used-as-control-flow))
11. Window object for the pdt accumulators, so the reset cannot be partial.
    ([S3](#s3--twelve-accumulators-for-one-statistic-reset-in-two-different-places))
12. `figure_diagnostics` into three panel builders; `StampedCsv` in the receiver.
    ([J6](#j6--figure_diagnostics-is-141-lines-and-three-unrelated-figures), [J5](#j5--toolstcp_receiverpy-still-runs-154-lines-at-module-level-over-five-globals))
13. Move the flow trace into a node; then the `CameraSource` split.
    ([J3](#j3--the-flow-trace-is-the-last-diagnostic-still-woven-through-the-app-script), [J4](#j4--the-camera-backend-is-still-eight-nullable-fields-and-a-mid-function-early-return))
14. Move the calibration essays out of the header; section `CLAUDE.md`.
    ([D1](#d1--srcopencvprocessorh-148-comment-lines-to-66-code-lines), [D3](#d3--claudemd-is-30-kb-and-line-44-is-one-paragraph-of-2500-words))
15. Pull the Camera2 dump into its own translation unit.
    ([D2](#d2--opencvprocessor-is-still-four-classes-in-a-trench-coat))
