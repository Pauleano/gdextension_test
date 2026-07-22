#include "OpenCVProcessor.h" //include custom header file
#include <godot_cpp/variant/utility_functions.hpp> 
#include <opencv2/opencv.hpp> //solvepnp should be inside of calib3d module which is inside opencv
#include <godot_cpp/classes/project_settings.hpp> //to convert path starting from res: to global path
#include <algorithm> //std::clamp for the downscale parameter
//#include <godot_cpp/variant/transform3d.hpp> wird momentan nicht benötigt, da es von wo anders anscheinend schon reingezogen wird
//#include <opencv2/objdetect/aruco_detector.hpp> falls error "incomplete type" auftauchen sollten 
//for intrinsics
#ifdef __ANDROID__
#include <camera/NdkCameraManager.h>
#include <camera/NdkCameraMetadata.h>
#include <camera/NdkCameraMetadataTags.h>
#include <android/log.h>
#endif
using namespace godot;

bool OpenCVProcessor::debug_prints_enabled = false;

//Einheitliches Format fuer alle Ausgaben: "[opencv_aruco] [OpenCVProcessor::funktion] text".
//Das feste Praefix macht sie in logcat greifbar (adb logcat | grep opencv_aruco), wo sie sonst
//zwischen Engine- und OpenXR-Ausgaben untergehen. __func__ statt handgeschriebenem Namen, damit
//die Angabe beim Umbenennen nicht veraltet.
//ACV_DBG haengt am debug_prints_enabled-Flag (aus GDScript, siehe main_3d.gd); die Argumente werden
//nur ausgewertet, wenn das Flag gesetzt ist -- wichtig fuer die Ausgaben im Detektions-Pfad.
//ACV_ERR ist NICHT abschaltbar: echte Fehler sollen auch ohne Debug-Flag im Log stehen.
#define ACV_DBG(...) \
    do { if (debug_prints_enabled) UtilityFunctions::print("[opencv_aruco] [OpenCVProcessor::", __func__, "] ", __VA_ARGS__); } while (0)
#define ACV_ERR(...) \
    UtilityFunctions::printerr("[opencv_aruco] [OpenCVProcessor::", __func__, "] ", __VA_ARGS__)

void OpenCVProcessor::_bind_methods() {
    ClassDB::bind_method(D_METHOD("get_6dof_of_all_aruco_patches_from_picture", "res_path"), &OpenCVProcessor::get_6dof_of_all_aruco_patches_from_picture);
    ClassDB::bind_method(D_METHOD("get_6dof_of_all_aruco_patches_from_webcam", "marker_size"), &OpenCVProcessor::get_6dof_of_all_aruco_patches_from_webcam);
    ClassDB::bind_method(D_METHOD("get_6dof_of_all_aruco_patches_from_godot_image", "image", "marker_size", "downscale", "intrinsics", "distortion", "lens_pose"), &OpenCVProcessor::get_6dof_of_all_aruco_patches_from_godot_image);
    //statisch gebunden, damit es vor new() aufrufbar ist (der Konstruktor gibt selbst schon aus)
    ClassDB::bind_static_method("OpenCVProcessor", D_METHOD("set_debug_prints_enabled", "enabled"), &OpenCVProcessor::set_debug_prints_enabled);

    ADD_SIGNAL(MethodInfo("marker_pose_found", PropertyInfo(Variant::TRANSFORM3D, "pose")));
}

void OpenCVProcessor::set_debug_prints_enabled(bool p_enabled) {
    debug_prints_enabled = p_enabled;
}

OpenCVProcessor::OpenCVProcessor() {

    ACV_DBG("opencv build information:\n", String(cv::getBuildInformation().c_str())); // Full build configuration.

    cv::aruco::DetectorParameters params;
    
    //NONE is cheapest; CONTOUR is a cheap middle ground; SUBPIX is the most accurate but iterative (slow on Quest).
    params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    detector = cv::aruco::ArucoDetector(
        cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50),
        params);
    
    init_quest_intrinsics();
}
OpenCVProcessor::~OpenCVProcessor() {}


void OpenCVProcessor::init_quest_intrinsics() {

    ACV_DBG("called");
    #ifdef __ANDROID__
    ACameraManager *mgr = ACameraManager_create();

    ACameraIdList *idList = nullptr;
    if (ACameraManager_getCameraIdList(mgr, &idList) != ACAMERA_OK) {
        ACV_ERR("ACameraManager_getCameraIdList failed");
        return;
    }

    for (int i = 0; i < idList->numCameras; i++) {

        const char *id = idList->cameraIds[i];

        // only care about Quest tracking cams
        std::string sid(id);
        if (sid != "50" && sid != "51")
            continue;

        ACameraMetadata *meta = nullptr;

        if (ACameraManager_getCameraCharacteristics(mgr, id, &meta) != ACAMERA_OK)
            continue;

        //Wert-Initialisierung: die count/type-Felder von dist werden unten auch dann ausgegeben,
        //wenn getConstEntry fehlschlaegt -- ohne {} waere das ein Lesen aus uninitialisiertem Speicher.
        ACameraMetadata_const_entry intr{};
        ACameraMetadata_const_entry dist{};
        ACameraMetadata_const_entry entry{};

        // -------------------------
        // INTRINSIC MATRIX
        // -------------------------

        if (ACameraMetadata_getConstEntry(
                meta,
                ACAMERA_LENS_INTRINSIC_CALIBRATION,
                &intr) == ACAMERA_OK && intr.count == 5)
        {
            float fx = intr.data.f[0];
            float fy = intr.data.f[1];
            float cx = intr.data.f[2];
            float cy = intr.data.f[3];
            ACV_DBG("camera ", sid.c_str(), " intrinsics: fx=", fx, " fy=", fy, " cx=", cx, " cy=", cy);

            //need to rescale the 1280x1280 camera intrinsics (quest3 resolution) to the 640x480 (resolution used in gdscript)
            
            cv::Mat K = (cv::Mat_<float>(3,3) <<
                fx, 0,  cx,
                0,  fy, cy,
                0,  0,  1
            );
            if (sid == "50") {
                K_cam50 = K;
            }

            if (sid == "51") {
                K_cam51 = K;
            }
            ACV_DBG("quest intrinsics loaded: camera=", sid.c_str());
        }

        // -------------------------
        // DISTORTION
        // -------------------------
        camera_status_t test= ACameraMetadata_getConstEntry(
                meta,
                ACAMERA_LENS_DISTORTION,
                &dist);
        
        if (test == ACAMERA_OK && dist.count >= 5)
        {
            D = cv::Mat(1, dist.count, CV_32F);

            for (int j = 0; j < dist.count; j++){
                D.at<float>(j) = dist.data.f[j];

            }
            ACV_DBG("quest distortions loaded: camera=", sid.c_str());
        }
        //status=-10004 heisst: keine Distortion-Koeffizienten verfuegbar
        ACV_DBG("distortion entry: status=", test, " count=", dist.count, " type=", dist.type);

        if (ACameraMetadata_getConstEntry(
            meta,
            ACAMERA_LENS_POSE_TRANSLATION,
            &entry) == ACAMERA_OK) {
            float tx = entry.data.f[0];
            float ty = entry.data.f[1];
            float tz = entry.data.f[2];
            ACV_DBG("lens_pose_translation: x=", tx, " y=", ty, " z=", tz);
        }

        if (ACameraMetadata_getConstEntry(
                meta,
                ACAMERA_LENS_POSE_ROTATION,
                &entry) == ACAMERA_OK) {
            float qx = entry.data.f[0];
            float qy = entry.data.f[1];
            float qz = entry.data.f[2];
            float qw = entry.data.f[3];
            ACV_DBG("lens_pose_rotation: x=", qx, " y=", qy, " z=", qz, " w=", qw);
        }

    if (ACameraMetadata_getConstEntry(
            meta,
            ACAMERA_LENS_POSE_REFERENCE,
            &entry) == ACAMERA_OK) {
        int pose_ref = entry.data.i32[0];
        ACV_DBG("lens_pose_reference=", pose_ref);
    }



        ACameraMetadata_free(meta);
    }

    ACameraManager_deleteCameraIdList(idList);
    ACameraManager_delete(mgr);
    #endif
}

Dictionary OpenCVProcessor::get_6dof_of_all_aruco_patches_from_picture(const String &res_path) {
    Dictionary result;

    String global_path = ProjectSettings::get_singleton()->globalize_path(res_path);
    std::string path_std = global_path.utf8().get_data();
    cv::Mat image = cv::imread(path_std);
    if (image.empty()) {
        ACV_ERR("could not load image: ", global_path);
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
        //kein Fehler, sondern der Normalfall sobald kein Marker im Bild ist -> nur Debug-Ausgabe
        ACV_DBG("kein Marker erkannt");
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
Dictionary OpenCVProcessor::detect_and_solve_all(const cv::Mat &frame, float marker_size, float downscale,const float &fx, const float &fy,const float &cx,const float &cy,const cv::Mat &distort,const Transform3D &lens_pose) {
    Dictionary result;

    // Physical passthrough-camera offset from the head-tracked reference (XRCamera3D), taken from
    // the Quest's ACAMERA_LENS_POSE_ROTATION / _TRANSLATION and passed in from GDScript. solvePnP
    // returns the marker pose in the physical CAMERA frame, so we pre-multiply by this lens pose to
    // re-express each marker relative to the head reference; GDScript then bakes it to world space
    // with the head transform sampled at capture time. Identity rotation + zero translation makes
    // this a no-op, so the pose is unchanged until real values are supplied.
    // NOTE: if markers land in the wrong place, the axis convention between the Android lens pose
    // and Godot is the knob -- try the conjugate quaternion / flipped translation signs (easiest to
    // do where lens_pose is constructed in GDScript). An identity Transform3D makes this a no-op.

    double tick_freq = cv::getTickFrequency();

    // Downscale before detection: the adaptive-threshold + contour stage scales with pixel count,
    // so 0.5 = ~4x fewer pixels. The fake intrinsics (fx=fy=width) scale with the image too, so the
    // metric pose stays correct; only corner localisation gets coarser. Pass 1.0f to disable.
    // Clamp to a sane range so a bad value from GDScript can't blow up cv::resize.
    const float DETECT_DOWNSCALE = std::clamp(downscale, 0.05f, 1.0f);
    int64_t t_resize = cv::getTickCount();
    cv::Mat det_frame;
    if (DETECT_DOWNSCALE != 1.0f) {
        cv::resize(frame, det_frame, cv::Size(), DETECT_DOWNSCALE, DETECT_DOWNSCALE, cv::INTER_AREA);
    } else {
        det_frame = frame;
    }
    double resize_ms = (cv::getTickCount() - t_resize) / tick_freq * 1000.0;

    float half = marker_size / 2.0f;
    std::vector<cv::Point3f> obj_pts = {
        {-half,  half, 0.0f},
        { half,  half, 0.0f},
        { half, -half, 0.0f},
        {-half, -half, 0.0f}
    };

    // Intrinsics derived from the (downscaled) detection frame, so corners + K share one pixel space.
    
    float w = static_cast<float>(det_frame.cols);
    float h = static_cast<float>(det_frame.rows);
    ACV_DBG("detect frame: width=", w, " height=", h);

    cv::Mat Kamera_matrix = (cv::Mat_<float>(3, 3) <<
        fx, 0, cx,
        0, fy, cy,
        0, 0, 1);
    /*
    cv::Mat Kamera_matrix;

    if (current_camera_id == 50)
        Kamera_matrix = K_cam50;
    else
        Kamera_matrix = K_cam51;
    */
    // Distortion coefficients are supplied by the caller and passed straight to solvePnP.
    // An empty Mat is valid and means "no distortion".

    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<int> ids;
    // Profiling: split the per-frame cost so we can see which stage dominates on the Quest.
    int64_t t_detect = cv::getTickCount();
    detector.detectMarkers(det_frame, corners, ids);
    double detect_ms = (cv::getTickCount() - t_detect) / tick_freq * 1000.0;
    ACV_DBG("resize_ms=", resize_ms, " detect_ms=", detect_ms);

    if (ids.empty()) {
        //kein Fehler, sondern der Normalfall sobald kein Marker im Bild ist -> nur Debug-Ausgabe
        ACV_DBG("kein Marker erkannt");
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

        result[ids[i]] = lens_pose * Transform3D(basis, origin);
    }

    ACV_DBG("solvepnp_ms=", solve_ms, " markers=", (int)ids.size());
    return result;
}

//is given a frame the Godot CameraServer/CameraFeed already owns ()
Dictionary OpenCVProcessor::get_6dof_of_all_aruco_patches_from_godot_image(const Ref<Image> &image, const float &marker_size, const float &downscale, const Vector4 &intrinsics, const PackedFloat64Array &distortion, const Transform3D &lens_pose) {
    Dictionary result;

    if (image.is_null() || image->is_empty()) {
        ACV_ERR("empty image from CameraFeed");
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

    // Unpack the packed Godot inputs: intrinsics = (fx, fy, cx, cy); distortion = OpenCV distCoeffs.
    float fx = (float)intrinsics.x;
    float fy = (float)intrinsics.y;
    float cx = (float)intrinsics.z;
    float cy = (float)intrinsics.w;

    // Copy the distortion vector into a column Mat (CV_32F, to match the camera matrix). An empty
    // array leaves `distort` empty, which solvePnP treats as zero distortion.
    cv::Mat distort;
    int dist_n = (int)distortion.size();
    if (dist_n > 0) {
        distort = cv::Mat(dist_n, 1, CV_32F);
        const double *dp = distortion.ptr();
        for (int j = 0; j < dist_n; ++j) {
            distort.at<float>(j) = (float)dp[j];
        }
    }

    return detect_and_solve_all(gray, marker_size, downscale, fx, fy, cx, cy, distort, lens_pose);
}

Dictionary OpenCVProcessor::get_6dof_of_all_aruco_patches_from_webcam(const float &marker_size) {
    Dictionary result;

    if (!cap.isOpened()) {
        cap.open(0, cv::CAP_DSHOW);
        if (!cap.isOpened()) {
            ACV_ERR("could not open webcam");
            return result;
        }
    }

    cv::Mat frame;
    if (!cap.read(frame) || frame.empty()) {
        ACV_ERR("could not read frame from webcam");
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
        //kein Fehler, sondern der Normalfall sobald kein Marker im Bild ist -> nur Debug-Ausgabe
        ACV_DBG("kein Marker erkannt");
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