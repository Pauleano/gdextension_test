
#pragma once

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/variant/quaternion.hpp>
#include <godot_cpp/variant/vector2.hpp>
#include <godot_cpp/variant/vector3.hpp>
#include <godot_cpp/variant/vector4.hpp>
#include <godot_cpp/variant/packed_float64_array.hpp>
#include <godot_cpp/variant/packed_vector2_array.hpp>
#include <godot_cpp/variant/transform3d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/core.hpp>
#include <memory>

using namespace godot;

//ArucoNano is forward-declared, deliberately NOT included: aruco_nano.h defines three statics
//without `inline` (visitedAwareTracingContour/getBorderErrors/thres255Adaptive), so including it
//here would emit them in every translation unit pulling this header in -- register_types.cpp does
//-- and the link fails on duplicate symbols. The header is included in OpenCVProcessor.cpp only,
//which is why `detector` is held by pointer rather than by value. The destructor lives in the
//.cpp, where the type is complete.
namespace aruco_nano { class ArucoDetector; }

class OpenCVProcessor : public Node {
    GDCLASS(OpenCVProcessor, Node)

private:
    //kept open across calls; opening DSHOW takes 1-3s, so reusing is essential for per-frame use
    cv::VideoCapture cap;
    //built once in ctor and reused; dictionary+params are baked in at construction
    std::unique_ptr<aruco_nano::ArucoDetector> detector;

    // ============================ Kalibrierung (Inspector-Properties) ============================
    // These used to be @export vars in the GDScript and were handed to the detection call as seven
    // arguments on every frame. They live here now: the C++ side is their only consumer, the call
    // collapses to detect_markers(image, head_pose), and the invariants below stop spanning the
    // language boundary.
    //
    // PRECISION: Vector4/Quaternion/Vector3 components are real_t = float32 in a single-precision
    // build, so those hold ~7 significant digits no matter what step the inspector uses -- the
    // intrinsics below already truncate when stored. That is far under the calibration's own noise,
    // but it is why the inspector will not show every digit written here. Only camera_distortion
    // (PackedFloat64Array) and the plain floats keep full double precision.
    //
    // THREADING: the detection task reads these from a WorkerThreadPool thread while the (remote)
    // inspector writes them on the main thread, unlocked -- exactly as the GDScript vars were read
    // before. A torn read costs one frame's markers, never consistency, because every one of them
    // is re-read at the top of each detection.

    // Intrinsics (fx, fy, cx, cy) in pixels for the NATIVE 640x480 frame; image_downscale_factor is
    // applied to them at use time, so NEVER bake it in here.
    // Calibrated with tools/cameraCalibration.py from TCP-streamed frames captured through the
    // GodotAndroidCamera (CameraX) path, i.e. the pipeline that actually runs. That provenance is
    // the point: the previous set came from the CameraServer readback path and agreed with this one
    // to 1.5px and 0.2% on the focal lengths, which is what ruled out CameraX picking a different
    // sensor crop -- a different field of view would have moved fx by percent, not by 0.08%. A
    // calibration is only valid for the capture path that produced its images, so RE-SHOOT these
    // after any change to the resolution, the output format, or the camera backend.
    Vector4 camera_intrinsics = Vector4(435.37335635, 435.96983202, 320.84589009, 241.55014114);
    // OpenCV distCoeffs (k1, k2, p1, p2, k3) for the Quest passthrough lens; an EMPTY array means
    // "no distortion". Filled in the ctor, since a PackedFloat64Array has no inline initialiser.
    PackedFloat64Array camera_distortion;
    // Detection resolution knob: 1.0 = native frame, 0.5 = half width AND half height, i.e. a
    // quarter of the pixels -> markedly cheaper detection, at the price of small or distant markers
    // dropping below the resolution the detector needs. It scales the frame AND camera_intrinsics
    // together inside detect_and_solve_all; that pairing is the whole reason it lives beside them.
    float image_downscale_factor = 1.0f;

    // Passthrough camera expressed in the OpenXR VIEW frame -- MEASURED, not read off the device.
    // Stored in the Quest's raw ACAMERA_LENS_POSE_* convention only because rebuild_lens_pose()
    // decodes that convention: the raw quaternion is ~168.8deg about X = the Android
    // sensor->camera-optical 180deg X-flip PLUS the camera's real ~11deg pitch, and since the
    // marker pose out of solvePnP already contains that same 180deg flip (the negate-Y/Z change of
    // basis in detect_and_solve_all), the rebuild multiplies by Quaternion(1,0,0,0) (=180deg about
    // X) to cancel it and keep ONLY the physical mounting tilt.
    // Translation is in the sensor frame (X right, Y up, Z toward viewer), which matches Godot
    // camera axes -> no sign flips.
    // These two are ONE calibration and must be replaced as a pair -- a rotation from one solve
    // beside a translation from another describes no camera that exists.
    //
    // DO NOT "fix" these by pasting in what init_quest_intrinsics() logs at startup, however
    // authoritative that dump looks. It is gyro-referenced: LENS_POSE_REFERENCE == GYROSCOPE means
    // the metadata describes the camera relative to the IMU, while the head pose it gets combined
    // with is the VIEW pose. Those frames differ by the IMU's mounting rotation, which NEITHER api
    // exposes -- OpenXR has no IMU reference space, Camera2 never mentions the view -- so the
    // difference cannot be looked up and has to be measured. Running the raw dump is what produced
    // a ~11mm offset that no pose-path change could touch (present with the head completely still,
    // on both the history and the xrLocateSpace path) while still swinging the patch around as the
    // head turned, the error being conjugated by the head transform.
    //
    // Solved by tools/handeye_solve.py from 500 captured samples of marker 0 (see handeye_capture
    // in the GDScript): H_i * L * M_i collapses to one world pose to 1.2mm median / 3.5mm p90, at a
    // rotation-observability ratio of 2.8 -- the solve warns above ~20, where the marker stayed too
    // central in frame for one rotation axis to be pinned down. Re-measure with that tool rather
    // than editing by eye.
    // CAUTION, learned the hard way: until 2026-08-12 these two literals were still the RAW
    // gyro-referenced dump, unchanged since 2026-07-07, while this comment already described them
    // as measured. Nothing about a wrong lens pose announces itself, and a comment claiming the
    // calibration was done is enough to stop anyone re-checking it -- which is exactly what kept a
    // ~0.9deg offset (15mm at 1m, 5mm at 40cm) alive across three branches and two Godot versions.
    // What the solve changed tells the rest of the story: the ~11.2deg X pitch came back
    // essentially untouched, and virtually the whole 0.45deg correction landed in YAW and ROLL --
    // precisely the two components the raw quaternion cannot express, its axis being pure X to
    // within 0.2deg. Translation moved only 4.7mm, matching the measured offset scaling with
    // distance (a translation error would not have).
    // Note what the same solve says about ORIENTATION: the marker's world rotation still scatters
    // 2.2deg median / 5.2deg p90, and that is NOT calibration error -- it is solvePnP's inherent
    // noise on a single small planar marker, and it stays no matter how good this pose is. A patch
    // that sits in the right place but visibly wobbles in orientation is that, and the fix is
    // temporal averaging or a multi-marker board, not another lens-pose hunt.
    Quaternion lens_rotation_raw = Quaternion(-0.99520788537121, -0.00260202523029, 0.00286401182712, 0.09770512676372);
    Vector3 lens_translation = Vector3(-0.03586537368457, -0.01756000674934, -0.06026289442816);

    // Fallback physical side length in meters for every marker id WITHOUT an aruco_patch_sizes
    // entry. Sets the pose's metric scale in solvePnP AND (through get_marker_size) the rendered
    // cuboid's edge length, so the two can never disagree.
    float aruco_patch_size = 0.1f;
    // Ground-truth lookup table: index = marker id, value = physical side length in meters.
    // 0 = unset -> that id falls back to aruco_patch_size (as do all ids past the end of the array),
    // so an untouched table behaves exactly like the single-size setup before it existed.
    PackedFloat64Array aruco_patch_sizes;

    // Derived, never edited: rebuilt by the setters above so an inspector edit takes effect on the
    // very next frame. Deriving them once at startup was the old behaviour and meant a lens-pose
    // tweak in the remote inspector of a running deploy did nothing at all.
    cv::Mat distort_mat;                       // camera_distortion as a CV_32F column
    Transform3D lens_pose;                     // decoded from lens_rotation_raw/lens_translation
    void rebuild_distortion();
    void rebuild_lens_pose();

    //shared detect+solvePnP pipeline; frame is grayscale. used by the godot-image path.
    //Everything it needs beyond the pixels is a member above: per-id marker sizes, intrinsics,
    //distortion, downscale and the lens pose. head_pose is the ONLY per-frame input -- the head
    //pose at the frame's capture time, which only the caller can know.
    //corners_out is filled with id -> PackedVector2Array of the 4 detected pixel corners (see the
    //definition for the exact contract); it is an out-parameter, never read.
    Dictionary detect_and_solve_all(const cv::Mat &frame, const Transform3D &head_pose, Dictionary &corners_out);

    //for intrinsics
    cv::Mat K_cam50; //intrinsics matrix
    cv::Mat K_cam51;
    int current_camera_id =50; //use first back camera and 50 is listed before 51, so camera 50 is always used
    cv::Mat D; //distortions
    bool intrinsics_ready = false;
    void init_quest_intrinsics();

    //gate for all debug output (ACV_DBG in the .cpp); errors (ACV_ERR) ignore it.
    //static on purpose: this class is a node AUTHORED IN THE SCENE, so PackedScene constructs it
    //before it applies a single property and long before any _ready -- an instance property could
    //not be set early enough by anyone. GDScript sets the static, then calls initialize().
    static bool debug_prints_enabled;

    //initialize() darf nur einmal wirken; siehe dort.
    bool initialized = false;

protected:
    static void _bind_methods();

public:
    OpenCVProcessor();
    ~OpenCVProcessor();

    //call from GDScript as OpenCVProcessor.set_debug_prints_enabled(true) BEFORE initialize()
    static void set_debug_prints_enabled(bool p_enabled);

    //everything the constructor used to do that prints or touches the device, deferred so the
    //debug flag above can still gate it. Called from _ready in open_cv_processor.gd; idempotent.
    void initialize();

    //function die wir in godot aufrufen wollen
    Dictionary get_6dof_of_all_aruco_patches_from_picture(const String &file_path);
    Dictionary get_6dof_of_all_aruco_patches_from_webcam(const float &marker_size);

    //THE detection entry point. Takes a frame the caller already owns (the CameraX plugin's Y-plane
    //on Quest, a CameraServer frame on desktop) plus the head pose at that frame's CAPTURE time,
    //and returns id -> Transform3D in WORLD space: head_pose * lens_pose is pre-multiplied onto
    //every solvePnP result, so the caller has nothing left to apply. An identity head_pose yields
    //raw camera-space poses. Everything else it needs is a property above.
    //corners_out is an OUT-parameter: Godot Dictionaries are shared references, so the caller passes
    //an (empty) Dictionary and gets the detected pixel corners written into its own instance.
    Dictionary detect_markers(const Ref<Image> &image, const Transform3D &head_pose, Dictionary corners_out);

    //The ONE place a marker id becomes a physical size: table entry when present and positive,
    //aruco_patch_size otherwise. Bound so the GDScript can scale the rendered patch with the exact
    //number solvePnP used -- one function, so the pose and the mesh cannot drift apart.
    float get_marker_size(int id) const;

    //Read-only view of the decoded lens pose, for the hand-eye capture in the GDScript: it has to
    //invert head_pose * lens_pose to recover the camera-space marker pose from a world-space one.
    Transform3D get_lens_pose() const;

    //Debug counterpart to detect_markers: projects WORLD-space marker poses back into pixel space
    //through the same intrinsics, distortion and lens pose the detection used. Only meaningful
    //ACROSS frames -- see the implementation for why the same-frame version cancels algebraically.
    Dictionary project_marker_corners(Dictionary world_poses, const Transform3D &head_pose) const;

    void set_camera_intrinsics(const Vector4 &p_value);
    Vector4 get_camera_intrinsics() const;
    void set_camera_distortion(const PackedFloat64Array &p_value);
    PackedFloat64Array get_camera_distortion() const;
    void set_image_downscale_factor(float p_value);
    float get_image_downscale_factor() const;
    void set_lens_rotation_raw(const Quaternion &p_value);
    Quaternion get_lens_rotation_raw() const;
    void set_lens_translation(const Vector3 &p_value);
    Vector3 get_lens_translation() const;
    void set_aruco_patch_size(float p_value);
    float get_aruco_patch_size() const;
    void set_aruco_patch_sizes(const PackedFloat64Array &p_value);
    PackedFloat64Array get_aruco_patch_sizes() const;
};
