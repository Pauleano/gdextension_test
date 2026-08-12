
#pragma once

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/variant/quaternion.hpp>
#include <godot_cpp/variant/vector3.hpp>
#include <godot_cpp/variant/vector2.hpp>
#include <godot_cpp/variant/vector4.hpp>
#include <godot_cpp/variant/packed_float64_array.hpp>
#include <godot_cpp/variant/packed_vector2_array.hpp>
#include <godot_cpp/variant/transform3d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/core.hpp>

using namespace godot;

class OpenCVProcessor : public Node {
    GDCLASS(OpenCVProcessor, Node)

private:
    //kept open across calls; opening DSHOW takes 1-3s, so reusing is essential for per-frame use
    cv::VideoCapture cap;
    //built once in ctor and reused; dictionary+params are baked in at construction
    cv::aruco::ArucoDetector detector;

    //shared detect+solvePnP pipeline; frame is BGR. used by the godot-image path.
    //marker_sizes maps marker id -> physical side length (m); ids without an entry (or with a
    //non-positive one) use default_marker_size, so differently sized patches solve correctly.
    //downscale (0<d<=1) shrinks the frame before detectMarkers -- the dominant cost on Quest.
    //corners_out is filled with id -> PackedVector2Array of the 4 detected pixel corners (see the
    //definition for the exact contract); it is an out-parameter, never read.
    Dictionary detect_and_solve_all(const cv::Mat &frame, const Dictionary &marker_sizes, float default_marker_size, float downscale,const float &fx, const float &fy,const float &cx,const float &cy,const cv::Mat &distort,const Transform3D &lens_pose, Dictionary &corners_out);

    //for intrinsics
    cv::Mat K_cam50; //intrinsics matrix
    cv::Mat K_cam51;
    int current_camera_id =50; //use first back camera and 50 is listed before 51, so camera 50 is always used
    cv::Mat D; //distortions
    bool intrinsics_ready = false;
    void init_quest_intrinsics();

    //gate for all debug output (ACV_DBG in the .cpp); errors (ACV_ERR) ignore it.
    //static on purpose: the ctor already prints, so GDScript must be able to set this
    //BEFORE OpenCVProcessor.new() -- an instance property would be set too late.
    static bool debug_prints_enabled;

protected:
    static void _bind_methods();

public:
    OpenCVProcessor();
    ~OpenCVProcessor();

    //call from GDScript as OpenCVProcessor.set_debug_prints_enabled(true) before new()
    static void set_debug_prints_enabled(bool p_enabled);

    //function die wir in godot aufrufen wollen
    Dictionary get_6dof_of_all_aruco_patches_from_picture(const String &file_path);
    Dictionary get_6dof_of_all_aruco_patches_from_webcam(const float &marker_size);
    //takes a frame the Godot CameraServer already owns (avoids a second DSHOW capture on Windows)
    //corners_out is an OUT-parameter: Godot Dictionaries are shared references, so the caller passes
    //an (empty) Dictionary and gets the detected pixel corners written into its own instance.
    Dictionary get_6dof_of_all_aruco_patches_from_godot_image(const Ref<Image> &image, const Dictionary &marker_sizes, const float &default_marker_size, const float &downscale, const Vector4 &intrinsics, const PackedFloat64Array &distortion, const Transform3D &lens_pose, Dictionary corners_out);
};
