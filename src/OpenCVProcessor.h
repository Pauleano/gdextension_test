
#pragma once

#include <godot_cpp/classes/node.hpp>

using namespace godot;

class OpenCVProcessor : public Node {
    GDCLASS(OpenCVProcessor, Node)

protected:
    static void _bind_methods();

public:
    OpenCVProcessor();
    ~OpenCVProcessor();

    //function die wir in godot aufrufen wollen
    Vector2 get_image_size(const String &file_path);
    //Transform3D get_6dof_of_aruco_path_using_webcam();
    Transform3D get_6dof_of_aruco_patch_from_picture(const String &file_path);
};
