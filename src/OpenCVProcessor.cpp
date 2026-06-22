#include "OpenCVProcessor.h" //include custom header file
#include <godot_cpp/variant/utility_functions.hpp> 
#include <opencv2/opencv.hpp> //solvepnp should be inside of calib3d module which is inside opencv
#include <godot_cpp/classes/project_settings.hpp> //to convert path starting from res: to global path
//#include <godot_cpp/variant/transform3d.hpp> wird momentan nicht benötigt, da es von wo anders anscheinend schon reingezogen wird
//#include <opencv2/objdetect/aruco_detector.hpp> falls error "incomplete type" auftauchen sollten 

using namespace godot;

void OpenCVProcessor::_bind_methods() {
    ClassDB::bind_method(D_METHOD("get_6dof_of_all_aruco_patches_from_picture", "res_path"), &OpenCVProcessor::get_6dof_of_all_aruco_patches_from_picture);
    ClassDB::bind_method(D_METHOD("get_6dof_of_all_aruco_patches_from_webcam", "marker_size"), &OpenCVProcessor::get_6dof_of_all_aruco_patches_from_webcam);
    ClassDB::bind_method(D_METHOD("get_6dof_of_all_aruco_patches_from_godot_image", "image", "marker_size"), &OpenCVProcessor::get_6dof_of_all_aruco_patches_from_godot_image);

    ADD_SIGNAL(MethodInfo("marker_pose_found", PropertyInfo(Variant::TRANSFORM3D, "pose")));
}

OpenCVProcessor::OpenCVProcessor() {
    cv::aruco::DetectorParameters params;
    params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    detector = cv::aruco::ArucoDetector(
        cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50),
        params);
}
OpenCVProcessor::~OpenCVProcessor() {}

Dictionary OpenCVProcessor::get_6dof_of_all_aruco_patches_from_picture(const String &res_path) {
    Dictionary result;

    String global_path = ProjectSettings::get_singleton()->globalize_path(res_path);
    std::string path_std = global_path.utf8().get_data();
    cv::Mat image = cv::imread(path_std);
    if (image.empty()) {
        UtilityFunctions::printerr("Could not load image: ", global_path);
        return result;
    }

    float half = 0.05f / 2.0f;
    std::vector<cv::Point3f> obj_pts = {
        {-half,  half, 0.0f},
        { half,  half, 0.0f},
        { half, -half, 0.0f},
        {-half, -half, 0.0f}
    };

    float w = static_cast<float>(image.cols);
    float h = static_cast<float>(image.rows);

    cv::Mat Kamera_matrix = (cv::Mat_<float>(3, 3) <<
        w, 0, w/2.0f,
        0, w, h/2.0f,
        0, 0, 1);
    cv::Mat distort = cv::Mat::zeros(5, 1, CV_32F);

    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<int> ids;
    detector.detectMarkers(image, corners, ids);

    if (ids.empty()) {
        UtilityFunctions::printerr("Kein Marker erkannt");
        return result;
    }

    for (size_t i = 0; i < ids.size(); ++i) {
        cv::Mat rvec, tvec;
        bool ok2 = cv::solvePnP(
            obj_pts, corners[i], Kamera_matrix, distort,
            rvec, tvec,
            false,
            cv::SOLVEPNP_IPPE_SQUARE);
        if (!ok2) {
            continue;
        }

        cv::Mat rot_matrix;
        cv::Rodrigues(rvec, rot_matrix);

        Basis basis(
            Vector3(rot_matrix.at<double>(0,0), -rot_matrix.at<double>(1,0), -rot_matrix.at<double>(2,0)),
            Vector3(-rot_matrix.at<double>(0,1), rot_matrix.at<double>(1,1), rot_matrix.at<double>(2,1)),
            Vector3(-rot_matrix.at<double>(0,2), rot_matrix.at<double>(1,2), rot_matrix.at<double>(2,2))
        );

        Vector3 origin(
            ( tvec.at<double>(0)),
            (-tvec.at<double>(1)),
            (-tvec.at<double>(2))
        );

        result[ids[i]] = Transform3D(basis, origin);
    }

    return result;
}

//shared detect+solvePnP pipeline; frame may be gray (1ch) or BGR (3ch). uses the passed marker_size.
//same approximate intrinsics (fx=fy=width, no distortion) and OpenCV->Godot change of basis
//as the picture/webcam variants above.
Dictionary OpenCVProcessor::detect_and_solve_all(const cv::Mat &frame, float marker_size) {
    Dictionary result;

    float half = marker_size / 2.0f;
    std::vector<cv::Point3f> obj_pts = {
        {-half,  half, 0.0f},
        { half,  half, 0.0f},
        { half, -half, 0.0f},
        {-half, -half, 0.0f}
    };

    float w = static_cast<float>(frame.cols);
    float h = static_cast<float>(frame.rows);

    cv::Mat Kamera_matrix = (cv::Mat_<float>(3, 3) <<
        w, 0, w/2.0f,
        0, w, h/2.0f,
        0, 0, 1);
    cv::Mat distort = cv::Mat::zeros(5, 1, CV_32F);

    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<int> ids;
    // Profiling: split the per-frame cost into detectMarkers vs. solvePnP so we can see which one
    // dominates on the Quest. cv::getTickCount/Frequency -> milliseconds.
    double tick_freq = cv::getTickFrequency();
    int64_t t_detect = cv::getTickCount();
    detector.detectMarkers(frame, corners, ids);
    double detect_ms = (cv::getTickCount() - t_detect) / tick_freq * 1000.0;
    UtilityFunctions::print("detectMarkers=", detect_ms, "ms");

    if (ids.empty()) {
        UtilityFunctions::printerr("Kein Marker erkannt");
        return result;
    }

    double solve_ms = 0.0;                               // accumulated solvePnP time over all markers
    for (size_t i = 0; i < ids.size(); ++i) {
        cv::Mat rvec, tvec;
        int64_t t_solve = cv::getTickCount();
        bool ok2 = cv::solvePnP(
            obj_pts, corners[i], Kamera_matrix, distort,
            rvec, tvec,
            false,
            cv::SOLVEPNP_IPPE_SQUARE);
        solve_ms += (cv::getTickCount() - t_solve) / tick_freq * 1000.0;
        if (!ok2) {
            continue;
        }

        cv::Mat rot_matrix;
        cv::Rodrigues(rvec, rot_matrix);

        Basis basis(
            Vector3(rot_matrix.at<double>(0,0), -rot_matrix.at<double>(1,0), -rot_matrix.at<double>(2,0)),
            Vector3(-rot_matrix.at<double>(0,1), rot_matrix.at<double>(1,1), rot_matrix.at<double>(2,1)),
            Vector3(-rot_matrix.at<double>(0,2), rot_matrix.at<double>(1,2), rot_matrix.at<double>(2,2))
        );

        Vector3 origin(
            ( tvec.at<double>(0)),
            (-tvec.at<double>(1)),
            (-tvec.at<double>(2))
        );

        result[ids[i]] = Transform3D(basis, origin);
    }

    UtilityFunctions::print("solvePnP total=", solve_ms, "ms  (", (int)ids.size(), " markers)");
    return result;
}

//is given a frame the Godot CameraServer/CameraFeed already owns ()
Dictionary OpenCVProcessor::get_6dof_of_all_aruco_patches_from_godot_image(const Ref<Image> &image, const float &marker_size) {
    Dictionary result;

    if (image.is_null() || image->is_empty()) {
        UtilityFunctions::printerr("Empty image from CameraFeed");
        return result;
    }

    //for meta quest 3 (expected format=rgba8):
    //Ref<Image> img= image;
    //if (img->get_format() != Image::FORMAT_RGBA8) {
    //    img = image->duplicate();
    //    img->convert(Image::FORMAT_RGBA8);
    //}
    //int width = img->get_width();
    //int height = img->get_height();
    //PackedByteArray data = img->get_data();
    //cv::Mat rgba(height, width, CV_8UC4, (void *)data.ptr());
    //cv::cvtColor(rgba, gray, cv::COLOR_RGBA2GRAY);

    // Channel count varies by platform/feed: desktop CameraServer gives RGB8 (3ch), while the
    // Quest passthrough feed (YUV_420_888) hands us the Y/luminance plane as R8 (1ch) -- which
    // is already grayscale. Pick the conversion from the actual bytes-per-pixel.
    int width = image->get_width();
    int height = image->get_height();
    PackedByteArray data = image->get_data();        // data owns image pixel data
    int channels = (width * height > 0) ? (int)(data.size() / (width * height)) : 0;

    cv::Mat gray;                                    // grayscale matrix detectMarkers wants
    if (channels == 1) {
        // single-channel (Quest passthrough Y-plane) is already grayscale -- use directly
        gray = cv::Mat(height, width, CV_8UC1, (void *)data.ptr());
    } else if (channels == 4) {
        cv::Mat rgba(height, width, CV_8UC4, (void *)data.ptr());
        cv::cvtColor(rgba, gray, cv::COLOR_RGBA2GRAY);
    } else {
        cv::Mat rgb(height, width, CV_8UC3, (void *)data.ptr());
        cv::cvtColor(rgb, gray, cv::COLOR_RGB2GRAY);
    }

    return detect_and_solve_all(gray, marker_size);
}

Dictionary OpenCVProcessor::get_6dof_of_all_aruco_patches_from_webcam(const float &marker_size) {
    Dictionary result;

    if (!cap.isOpened()) {
        cap.open(0, cv::CAP_DSHOW);
        if (!cap.isOpened()) {
            UtilityFunctions::printerr("Could not open webcam");
            return result;
        }
    }

    cv::Mat frame;
    if (!cap.read(frame) || frame.empty()) {
        UtilityFunctions::printerr("Could not read frame from webcam");
        return result;
    }

    float half = marker_size / 2.0f;
    std::vector<cv::Point3f> obj_pts = {
        {-half,  half, 0.0f},
        { half,  half, 0.0f},
        { half, -half, 0.0f},
        {-half, -half, 0.0f}
    };

    float w = static_cast<float>(frame.cols);
    float h = static_cast<float>(frame.rows);

    cv::Mat Kamera_matrix = (cv::Mat_<float>(3, 3) <<
        w, 0, w/2.0f,
        0, w, h/2.0f,
        0, 0, 1);
    cv::Mat distort = cv::Mat::zeros(5, 1, CV_32F);

    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<int> ids;
    detector.detectMarkers(frame, corners, ids);

    if (ids.empty()) {
        UtilityFunctions::printerr("Kein Marker erkannt");
        return result;
    }

    for (size_t i = 0; i < ids.size(); ++i) {
        cv::Mat rvec, tvec;
        bool ok2 = cv::solvePnP(
            obj_pts, corners[i], Kamera_matrix, distort,
            rvec, tvec,
            false,
            cv::SOLVEPNP_IPPE_SQUARE);
        if (!ok2) {
            continue;
        }

        cv::Mat rot_matrix;
        cv::Rodrigues(rvec, rot_matrix);

        Basis basis(
            Vector3(rot_matrix.at<double>(0,0), -rot_matrix.at<double>(1,0), -rot_matrix.at<double>(2,0)),
            Vector3(-rot_matrix.at<double>(0,1), rot_matrix.at<double>(1,1), rot_matrix.at<double>(2,1)),
            Vector3(-rot_matrix.at<double>(0,2), rot_matrix.at<double>(1,2), rot_matrix.at<double>(2,2))
        );

        Vector3 origin(
            ( tvec.at<double>(0)),
            (-tvec.at<double>(1)),
            (-tvec.at<double>(2))
        );

        result[ids[i]] = Transform3D(basis, origin);
    }

    return result;
}