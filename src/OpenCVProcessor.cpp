#include "OpenCVProcessor.h" //include custom header file
#include <godot_cpp/variant/utility_functions.hpp> 
#include <opencv2/opencv.hpp> //solvepnp should be inside of calib3d module which is inside opencv
#include <godot_cpp/classes/project_settings.hpp> //to convert path starting from res: to global path
//#include <godot_cpp/variant/transform3d.hpp> wird momentan nicht benötigt, da es von wo anders anscheinend schon reingezogen wird
//#include <opencv2/objdetect/aruco_detector.hpp> falls error "incomplete type" auftauchen sollten 

using namespace godot;

void OpenCVProcessor::_bind_methods() {
    ClassDB::bind_method(D_METHOD("get_image_size", "res_path"), &OpenCVProcessor::get_image_size);
    ClassDB::bind_method(D_METHOD("get_6dof_of_aruco_patch_from_picture", "res_path"), &OpenCVProcessor::get_6dof_of_aruco_patch_from_picture);
    
    ADD_SIGNAL(MethodInfo("marker_pose_found", PropertyInfo(Variant::TRANSFORM3D, "pose")));
}

OpenCVProcessor::OpenCVProcessor() {}
OpenCVProcessor::~OpenCVProcessor() {}

Vector2 OpenCVProcessor::get_image_size(const String &res_path) {
   
    String global_path = ProjectSettings::get_singleton()->globalize_path(res_path);
    //godot string in std::string konvertieren für opencv
    std::string path_std = global_path.utf8().get_data();

    //opencv aufrufen: bild laden
    cv::Mat image = cv::imread(path_std);
    
    //prüfen ob bild geladen
    if (image.empty()) {
        UtilityFunctions::printerr("OpenCV Fehler: Konnte Bild nicht laden unter res:// Pfad: ", res_path, " | Gloabaler Pfad:", global_path);
        return Vector2(-1, -1); // Fehlerindikator
    }

    // 4. Dimensionen auslesen und an Godot zurückgeben
    int width = image.cols;
    int height = image.rows;
    
    UtilityFunctions::print("OpenCV hat das Bild erfolgreich geladen!");
    
    return Vector2(width, height);
}

Transform3D OpenCVProcessor::get_6dof_of_aruco_patch_from_picture(const String &res_path) {
    //res_path to global_path
    String global_path = ProjectSettings::get_singleton()->globalize_path(res_path);
    std::string path_std = global_path.utf8().get_data();
    //open image-file (expects png-file)
    cv::Mat image = cv::imread(path_std);
    if (image.empty()) {
        UtilityFunctions::printerr("Could not load image: ", global_path);
        return Transform3D();  // identity transform as error sentinel
    }

    //get used dictionary and get (default) 
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    auto params = cv::aruco::DetectorParameters();
    params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    
    //initialize detector 
    auto detector = cv::aruco::ArucoDetector(dictionary,params);

    //initialise markers corners (oben links, oben rechts, unten rechts, unten links)
    //in this order because we use flag=ippe_square
    float half = 0.05f / 2.0f;
    std::vector<cv::Point3f> obj_pts = {
        {-half,  half, 0.0f},
        { half,  half, 0.0f},
        { half, -half, 0.0f},
        {-half, -half, 0.0f}
    };

    //get bild größe (x=breite, y=höhe)
    float w = static_cast<float>(image.cols);
    float h = static_cast<float>(image.rows);

    //initialising K (camera intrinsics matrix) and distortion vector
    //both are approximations to be replace with exact values
    cv::Mat Kamera_matrix = (cv::Mat_<float>(3, 3) <<
        w, 0, w/2.0f,
        0, w, h/2.0f,
        0, 0, 1);
    cv::Mat distort = cv::Mat::zeros(5, 1, CV_32F);   

    //initialise outputs
    //vector that contains the four corners (each of type Point2f, a float 2d point) of each identified aruco patch, stored in a vector
    std::vector<std::vector<cv::Point2f>> corners;
    //vector which contains the ids of the identified trackers
    std::vector<int> ids;
   
    detector.detectMarkers(image, corners, ids);

    //falls kein marker identifiziert wurde
    if (ids.empty()) {
    UtilityFunctions::printerr("Kein Marker erkannt");
    return Transform3D();
    }

    //simplifizierende annahme; nur ein tracker
    //outputs initialisieren
    cv::Mat rvec,tvec;
    //mit solvePnP rvec und tvec bestimmen
    bool ok2 = cv::solvePnP(
        obj_pts, corners[0], Kamera_matrix, distort,
        rvec, tvec,
        false,  //dont use extrinsic guess (there are no rvec, tvec given as estimations of solutions)
        cv::SOLVEPNP_IPPE_SQUARE); //because we created the obj_pts matrix

    if (!ok2) {
        return Transform3D(); 
    }

    //convert rvec und tvec to Transform3D
    cv::Mat rot_matrix;
    cv::Rodrigues(rvec,rot_matrix);

// Basis-Konvertierung: Y und Z der OpenCV-Matrix flippen (M=diag(1,-1,-1)=M^-1, basiswechsel durch M*rot_mat*M^-1)
    Basis basis(
        Vector3( rot_matrix.at<double>(0,0), -rot_matrix.at<double>(0,1), -rot_matrix.at<double>(0,2)),
        Vector3(-rot_matrix.at<double>(1,0),  rot_matrix.at<double>(1,1),  rot_matrix.at<double>(1,2)),
        Vector3(-rot_matrix.at<double>(2,0),  rot_matrix.at<double>(2,1),  rot_matrix.at<double>(2,2))
    );

    Vector3 origin(
        static_cast<float>( tvec.at<double>(0)),
        static_cast<float>(-tvec.at<double>(1)),
        static_cast<float>(-tvec.at<double>(2))
    );

    return Transform3D(basis, origin);

}
