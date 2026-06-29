
#pragma once

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/image.hpp>
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
    //downscale (0<d<=1) shrinks the frame before detectMarkers -- the dominant cost on Quest.
    Dictionary detect_and_solve_all(const cv::Mat &frame, float marker_size, float downscale);

    //for intrinsics
    cv::Mat K; //intrinsics matrix
    cv::Mat D; //distortions
    bool intrinsics_ready = false;
    void init_quest_intrinsics();

protected:
    static void _bind_methods();

public:
    OpenCVProcessor();
    ~OpenCVProcessor();

    //function die wir in godot aufrufen wollen
    Dictionary get_6dof_of_all_aruco_patches_from_picture(const String &file_path);
    Dictionary get_6dof_of_all_aruco_patches_from_webcam(const float &marker_size);
    //takes a frame the Godot CameraServer already owns (avoids a second DSHOW capture on Windows)
    Dictionary get_6dof_of_all_aruco_patches_from_godot_image(const Ref<Image> &image, const float &marker_size, const float &downscale);
};
