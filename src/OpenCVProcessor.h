
#pragma once

#include <godot_cpp/classes/node.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

using namespace godot;

class OpenCVProcessor : public Node {
    GDCLASS(OpenCVProcessor, Node)

private:
    //kept open across calls; opening DSHOW takes 1-3s, so reusing is essential for per-frame use
    cv::VideoCapture cap;
    //built once in ctor and reused; dictionary+params are baked in at construction
    cv::aruco::ArucoDetector detector;

protected:
    static void _bind_methods();

public:
    OpenCVProcessor();
    ~OpenCVProcessor();

    //function die wir in godot aufrufen wollen
    Vector2 get_image_size(const String &file_path);
    //Transform3D get_6dof_of_aruco_path_using_webcam();
    Transform3D get_6dof_of_aruco_patch_from_picture(const String &file_path);
    Transform3D get_6dof_of_aruco_patch_from_webcam(const float &marker_size);
    Dictionary get_6dof_of_all_aruco_patches_from_picture(const String &file_path);
    Dictionary get_6dof_of_all_aruco_patches_from_webcam(const float &marker_size);
};
