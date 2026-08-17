#include "OpenCVProcessor.h" //include custom header file
#include "aruco_nano.h" //header-only ArucoNano detector; see the forward decl in OpenCVProcessor.h
#include <godot_cpp/variant/utility_functions.hpp>
#include <opencv2/opencv.hpp> //solvepnp should be inside of calib3d module which is inside opencv
#include <godot_cpp/classes/project_settings.hpp> //to convert path starting from res: to global path
#include <godot_cpp/classes/engine.hpp> //is_editor_hint(): der Knoten liegt in der Szene, der Editor baut ihn mit
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
//ACV_DBG haengt am debug_prints_enabled-Flag (aus GDScript, siehe open_cv_processor.gd); die Argumente werden
//nur ausgewertet, wenn das Flag gesetzt ist -- wichtig fuer die Ausgaben im Detektions-Pfad.
//ACV_ERR ist NICHT abschaltbar: echte Fehler sollen auch ohne Debug-Flag im Log stehen.
#define ACV_DBG(...) \
    do { if (debug_prints_enabled) UtilityFunctions::print("[opencv_aruco] [OpenCVProcessor::", __func__, "] ", __VA_ARGS__); } while (0)
#define ACV_ERR(...) \
    UtilityFunctions::printerr("[opencv_aruco] [OpenCVProcessor::", __func__, "] ", __VA_ARGS__)

void OpenCVProcessor::_bind_methods() {
    ClassDB::bind_method(D_METHOD("get_6dof_of_all_aruco_patches_from_picture", "res_path"), &OpenCVProcessor::get_6dof_of_all_aruco_patches_from_picture);
    ClassDB::bind_method(D_METHOD("get_6dof_of_all_aruco_patches_from_webcam", "marker_size"), &OpenCVProcessor::get_6dof_of_all_aruco_patches_from_webcam);
    ClassDB::bind_method(D_METHOD("detect_markers", "image", "head_pose", "corners_out"), &OpenCVProcessor::detect_markers);
    ClassDB::bind_method(D_METHOD("get_marker_size", "id"), &OpenCVProcessor::get_marker_size);
    ClassDB::bind_method(D_METHOD("get_lens_pose"), &OpenCVProcessor::get_lens_pose);
    ClassDB::bind_method(D_METHOD("project_marker_corners", "world_poses", "head_pose"), &OpenCVProcessor::project_marker_corners);
    //statisch gebunden, damit GDScript es setzen kann, BEVOR initialize() laeuft -- der Knoten
    //selbst wird schon beim Instanziieren der Szene gebaut, also lange vor jedem Property und
    //jedem _ready (siehe initialize()).
    ClassDB::bind_static_method("OpenCVProcessor", D_METHOD("set_debug_prints_enabled", "enabled"), &OpenCVProcessor::set_debug_prints_enabled);
    ClassDB::bind_method(D_METHOD("initialize"), &OpenCVProcessor::initialize);

    ADD_SIGNAL(MethodInfo("marker_pose_found", PropertyInfo(Variant::TRANSFORM3D, "pose")));

    // ---- Kalibrierungs-Properties (frueher @export im GDScript) ----
    // Jede Property bekommt einen EXPLIZITEN Range-Hint, allein wegen der Schrittweite: ohne Hint
    // rastet der Inspector auf 0.001, und ein dort editierter Verzeichnungskoeffizient wird auf drei
    // Nachkommastellen gerundet -- was beide tangentialen Terme glatt auf null setzt. Die
    // Schrittweite, nicht der Wertebereich, ist hier der Punkt; "or_greater"/"or_less" heben die
    // Grenzen wieder auf, "hide_slider" laesst nur das Zahlenfeld stehen.
    ClassDB::bind_method(D_METHOD("set_camera_intrinsics", "value"), &OpenCVProcessor::set_camera_intrinsics);
    ClassDB::bind_method(D_METHOD("get_camera_intrinsics"), &OpenCVProcessor::get_camera_intrinsics);
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR4, "camera_intrinsics", PROPERTY_HINT_RANGE,
                     "0,2000,0.00001,or_greater,hide_slider"),
            "set_camera_intrinsics", "get_camera_intrinsics");

    ClassDB::bind_method(D_METHOD("set_camera_distortion", "value"), &OpenCVProcessor::set_camera_distortion);
    ClassDB::bind_method(D_METHOD("get_camera_distortion"), &OpenCVProcessor::get_camera_distortion);
    // Bei Array-Properties traegt der Elementtyp den Hint: "<Typ>/<Hint>:<Hint-String>" -- dieselbe
    // Kodierung, die @export_range auf einem Array[float] erzeugt.
    ADD_PROPERTY(PropertyInfo(Variant::PACKED_FLOAT64_ARRAY, "camera_distortion", PROPERTY_HINT_TYPE_STRING,
                     String::num_int64(Variant::FLOAT) + "/" + String::num_int64(PROPERTY_HINT_RANGE) +
                             ":-1,1,0.00000001,or_greater,or_less,hide_slider"),
            "set_camera_distortion", "get_camera_distortion");

    ClassDB::bind_method(D_METHOD("set_image_downscale_factor", "value"), &OpenCVProcessor::set_image_downscale_factor);
    ClassDB::bind_method(D_METHOD("get_image_downscale_factor"), &OpenCVProcessor::get_image_downscale_factor);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "image_downscale_factor", PROPERTY_HINT_RANGE, "0.1,1.0,0.05"),
            "set_image_downscale_factor", "get_image_downscale_factor");

    ClassDB::bind_method(D_METHOD("set_lens_rotation_raw", "value"), &OpenCVProcessor::set_lens_rotation_raw);
    ClassDB::bind_method(D_METHOD("get_lens_rotation_raw"), &OpenCVProcessor::get_lens_rotation_raw);
    ADD_PROPERTY(PropertyInfo(Variant::QUATERNION, "lens_rotation_raw", PROPERTY_HINT_RANGE,
                     "-1,1,0.00000001,or_greater,or_less,hide_slider"),
            "set_lens_rotation_raw", "get_lens_rotation_raw");

    ClassDB::bind_method(D_METHOD("set_lens_translation", "value"), &OpenCVProcessor::set_lens_translation);
    ClassDB::bind_method(D_METHOD("get_lens_translation"), &OpenCVProcessor::get_lens_translation);
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "lens_translation", PROPERTY_HINT_RANGE,
                     "-1,1,0.00000001,or_greater,or_less,hide_slider,suffix:m"),
            "set_lens_translation", "get_lens_translation");

    ClassDB::bind_method(D_METHOD("set_aruco_patch_size", "value"), &OpenCVProcessor::set_aruco_patch_size);
    ClassDB::bind_method(D_METHOD("get_aruco_patch_size"), &OpenCVProcessor::get_aruco_patch_size);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "aruco_patch_size", PROPERTY_HINT_RANGE,
                     "0.01,0.3,0.000001,or_greater,suffix:m"),
            "set_aruco_patch_size", "get_aruco_patch_size");

    ClassDB::bind_method(D_METHOD("set_aruco_patch_sizes", "value"), &OpenCVProcessor::set_aruco_patch_sizes);
    ClassDB::bind_method(D_METHOD("get_aruco_patch_sizes"), &OpenCVProcessor::get_aruco_patch_sizes);
    ADD_PROPERTY(PropertyInfo(Variant::PACKED_FLOAT64_ARRAY, "aruco_patch_sizes", PROPERTY_HINT_TYPE_STRING,
                     String::num_int64(Variant::FLOAT) + "/" + String::num_int64(PROPERTY_HINT_RANGE) +
                             ":0.01,0.3,0.000001,or_greater,suffix:m"),
            "set_aruco_patch_sizes", "get_aruco_patch_sizes");
}

// ================================ Kalibrierung: Zugriff + Ableitung ================================
// Die Setter bauen die abgeleiteten Groessen SOFORT neu. Genau das war vorher nicht so: das GDScript
// hat die Linsenpose einmal in _ready gerechnet, also blieb eine Korrektur im Remote-Inspector eines
// laufenden Deploys wirkungslos, bis die App neu startete.

void OpenCVProcessor::rebuild_distortion() {
    // Spalten-Mat in CV_32F, passend zur Kameramatrix. Ein leeres Array laesst distort_mat leer --
    // das ist gueltig und heisst fuer solvePnP "keine Verzeichnung".
    const int n = (int)camera_distortion.size();
    if (n <= 0) {
        distort_mat = cv::Mat();
        return;
    }
    distort_mat = cv::Mat(n, 1, CV_32F);
    const double *dp = camera_distortion.ptr();
    for (int j = 0; j < n; ++j) {
        distort_mat.at<float>(j) = (float)dp[j];
    }
}

void OpenCVProcessor::rebuild_lens_pose() {
    // Siehe die ausfuehrliche Herleitung an lens_rotation_raw im Header: die Multiplikation mit
    // Quaternion(1,0,0,0) (=180deg um X) hebt genau den Flip auf, den detect_and_solve_all beim
    // Basiswechsel OpenCV->Godot schon eingebaut hat, und laesst nur die physische Neigung stehen.
    lens_pose = Transform3D(Basis((lens_rotation_raw * Quaternion(1, 0, 0, 0)).inverse()), lens_translation);
}

void OpenCVProcessor::set_camera_intrinsics(const Vector4 &p_value) { camera_intrinsics = p_value; }
Vector4 OpenCVProcessor::get_camera_intrinsics() const { return camera_intrinsics; }

void OpenCVProcessor::set_camera_distortion(const PackedFloat64Array &p_value) {
    camera_distortion = p_value;
    rebuild_distortion();
}
PackedFloat64Array OpenCVProcessor::get_camera_distortion() const { return camera_distortion; }

void OpenCVProcessor::set_image_downscale_factor(float p_value) { image_downscale_factor = p_value; }
float OpenCVProcessor::get_image_downscale_factor() const { return image_downscale_factor; }

void OpenCVProcessor::set_lens_rotation_raw(const Quaternion &p_value) {
    lens_rotation_raw = p_value;
    rebuild_lens_pose();
}
Quaternion OpenCVProcessor::get_lens_rotation_raw() const { return lens_rotation_raw; }

void OpenCVProcessor::set_lens_translation(const Vector3 &p_value) {
    lens_translation = p_value;
    rebuild_lens_pose();
}
Vector3 OpenCVProcessor::get_lens_translation() const { return lens_translation; }

void OpenCVProcessor::set_aruco_patch_size(float p_value) { aruco_patch_size = p_value; }
float OpenCVProcessor::get_aruco_patch_size() const { return aruco_patch_size; }

void OpenCVProcessor::set_aruco_patch_sizes(const PackedFloat64Array &p_value) { aruco_patch_sizes = p_value; }
PackedFloat64Array OpenCVProcessor::get_aruco_patch_sizes() const { return aruco_patch_sizes; }

Transform3D OpenCVProcessor::get_lens_pose() const { return lens_pose; }

//Nicht-positive Eintraege zaehlen als "kein Eintrag", damit eine kaputte Tabellenzeile solvePnP nie
//ein entartetes Quadrat unterschiebt -- und damit eine 0 im Inspector schlicht "unbelegt" heisst.
float OpenCVProcessor::get_marker_size(int id) const {
    if (id >= 0 && id < (int)aruco_patch_sizes.size()) {
        const double s = aruco_patch_sizes[id];
        if (s > 0.0) {
            return (float)s;
        }
    }
    return aruco_patch_size;
}

void OpenCVProcessor::set_debug_prints_enabled(bool p_enabled) {
    debug_prints_enabled = p_enabled;
}

//Der Konstruktor MUSS still bleiben. Seit die Klasse ein in der Szene angelegter Knoten ist (statt
//per OpenCVProcessor.new() aus _ready erzeugt), laeuft er beim Instanziieren der Szene: PackedScene
//baut den Knoten und setzt ERST DANACH die Properties, und ueber uns haengt kein Skript mehr, das
//das Flag frueher setzen koennte. Alles, was ausgibt oder auf Geraete zugreift, steht darum in
//initialize() -- das GDScript eine Zeile nach set_debug_prints_enabled() aufruft.
//Der Detektor selbst bleibt hier: er gibt nichts aus und muss vor dem ersten Aufruf stehen.
OpenCVProcessor::OpenCVProcessor() {

    //ArucoNano replaces cv::aruco::ArucoDetector: same detectMarkers signature, but its own
    //"visited aware" contour tracer plus direct sub-pixel sampling. There is no
    //cornerRefinementMethod any more -- sub-pixel refinement is built in, not an opt-in stage.
    //
    //ARUCO_MIP_36h12 instead of the old DICT_4X4_50: 36 bits with a minimum Hamming distance of
    //12 between any two codes, against 16 bits and a distance of 4. ArucoNano defaults
    //errorCorrectionRate to 0 (OpenCV uses 0.6, which its author considers a false-positive
    //hazard), so there is no correction to lean on -- and a distance of 4 leaves no margin for
    //two flipped bits. That mismatch, not the detector itself, is what made 4X4_50 flaky here.
    //Cost: 6x6 bits plus the border is 8 modules across against 6, so a marker needs ~33% more
    //pixels to identify, i.e. ~25% less working distance at the same printed size.
    //
    //The vector constructor is REQUIRED here. The single-dictionary overload does
    //_params.dicts.push_back(dict) ON TOP of the default, which with this dictionary would
    //leave us searching {ARUCO_MIP_36h12, ARUCO_MIP_36h12} -- every candidate identified twice.
    //The vector overload assigns instead.
    detector = std::make_unique<aruco_nano::ArucoDetector>(
        std::vector<cv::aruco::Dictionary>{ cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_MIP_36h12) });

    //Defaults der Packed-Arrays: die haben keinen Inline-Initialisierer im Header. Godot liest die
    //Property-Defaults, indem es die Klasse einmal instanziiert -- was hier steht, ist also genau
    //das, was der Inspector als "unveraendert" ansieht.
    camera_distortion = PackedFloat64Array();
    camera_distortion.push_back(-0.00993192);
    camera_distortion.push_back(0.11168738);
    camera_distortion.push_back(0.00062258);
    camera_distortion.push_back(0.00113467);
    camera_distortion.push_back(-0.23033717);
    aruco_patch_sizes = PackedFloat64Array();
    for (int i = 0; i < 10; ++i) {
        aruco_patch_sizes.push_back(0.05);
    }

    //Die abgeleiteten Groessen einmal aufbauen, damit sie auch dann stimmen, wenn die Szene die
    //Properties gar nicht ueberschreibt (dann laeuft kein Setter).
    rebuild_distortion();
    rebuild_lens_pose();
}
OpenCVProcessor::~OpenCVProcessor() {}

//Der redselige Teil des frueheren Konstruktors, aus GDScript aufgerufen, NACHDEM
//set_debug_prints_enabled() gelaufen ist -- nur so laesst sich der Build-Info-Dump ueberhaupt noch
//abschalten (siehe Konstruktor). Idempotent, damit ein zweiter Aufruf (Szene neu geladen, Skript
//umgebaut) nicht ein zweites Mal die Kamera-Metadaten liest.
void OpenCVProcessor::initialize() {
    //Im Editor nur zurueckhalten: die Szene enthaelt den Knoten, also baut der Editor ihn bei jedem
    //Oeffnen mit. Der Build-Info-Dump gehoert nicht ins Editor-Ausgabefenster, und die Quest-
    //Kameras gibt es auf dem Entwicklungsrechner ohnehin nicht. VOR dem initialized-Flag, damit ein
    //Abbruch im Editor den einmaligen Lauf nicht verbraucht.
    if (Engine::get_singleton()->is_editor_hint()) {
        return;
    }
    if (initialized) {
        return;
    }
    initialized = true;

    ACV_DBG("opencv build information:\n", String(cv::getBuildInformation().c_str())); // Full build configuration.

    init_quest_intrinsics();
}


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
    detector->detectMarkers(image, corners, ids);

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

//shared detect+solvePnP pipeline; frame may be gray (1ch) or BGR (3ch). Everything it needs beyond
//the pixels and the head pose is a property (see the header): per-id marker sizes, intrinsics,
//distortion, downscale, lens pose. Same OpenCV->Godot change of basis as the picture/webcam
//variants above.
Dictionary OpenCVProcessor::detect_and_solve_all(const cv::Mat &frame, const Transform3D &head_pose, Dictionary &corners_out) {
    Dictionary result;

    // The two transforms that hold for EVERY marker, combined once: solvePnP returns the marker
    // pose in the physical CAMERA frame, head_pose is where the head was when the shutter opened
    // (only the caller can know that), and lens_pose is the fixed camera-to-head offset. So the
    // returned poses are already WORLD space and the caller has nothing left to apply. An identity
    // head_pose makes this the lens pose alone and yields near-camera-space poses.
    // NOTE: if markers land in the wrong place, the axis convention between the Android lens pose
    // and Godot is the knob -- try the conjugate quaternion / flipped translation signs on
    // lens_rotation_raw / lens_translation.
    const Transform3D cam_to_world = head_pose * lens_pose;

    double tick_freq = cv::getTickFrequency();

    // Downscale before detection: the adaptive-threshold + contour stage scales with pixel count,
    // so 0.5 = ~4x fewer pixels. The fake intrinsics (fx=fy=width) scale with the image too, so the
    // metric pose stays correct; only corner localisation gets coarser. Pass 1.0f to disable.
    // Clamp to a sane range so a bad value from the inspector can't blow up cv::resize.
    const float DETECT_DOWNSCALE = std::clamp(image_downscale_factor, 0.05f, 1.0f);
    int64_t t_resize = cv::getTickCount();
    cv::Mat det_frame;
    if (DETECT_DOWNSCALE != 1.0f) {
        cv::resize(frame, det_frame, cv::Size(), DETECT_DOWNSCALE, DETECT_DOWNSCALE, cv::INTER_AREA);
    } else {
        det_frame = frame;
    }
    double resize_ms = (cv::getTickCount() - t_resize) / tick_freq * 1000.0;

    // Intrinsics derived from the (downscaled) detection frame, so corners + K share one pixel space.
    
    float w = static_cast<float>(det_frame.cols);
    float h = static_cast<float>(det_frame.rows);
    ACV_DBG("detect frame: width=", w, " height=", h);

    // camera_intrinsics is calibrated for the NATIVE frame, so all four components scale with the
    // image -- and with the CLAMPED factor used for the resize above, not the raw property, so the
    // two can never disagree. (GDScript used to scale them with the unclamped value and hand the
    // result in; an out-of-range factor silently produced a camera matrix for a frame size that was
    // never rendered.)
    const float fx = (float)camera_intrinsics.x * DETECT_DOWNSCALE;
    const float fy = (float)camera_intrinsics.y * DETECT_DOWNSCALE;
    const float cx = (float)camera_intrinsics.z * DETECT_DOWNSCALE;
    const float cy = (float)camera_intrinsics.w * DETECT_DOWNSCALE;

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
    detector->detectMarkers(det_frame, corners, ids);
    double detect_ms = (cv::getTickCount() - t_detect) / tick_freq * 1000.0;
    ACV_DBG("resize_ms=", resize_ms, " detect_ms=", detect_ms);

    if (ids.empty()) {
        //kein Fehler, sondern der Normalfall sobald kein Marker im Bild ist -> nur Debug-Ausgabe
        ACV_DBG("kein Marker erkannt");
        return result;
    }

    // Pixel corners of everything detectMarkers found, handed back for the debug TCP overlay
    // (tools/tcp_receiver.py feeds them to cv2.aruco.drawDetectedMarkers on the laptop, so the
    // stream shows the borders THIS detector saw instead of a second, independent detection).
    // Divided by DETECT_DOWNSCALE, i.e. mapped back out of the detection frame into the ORIGINAL
    // frame's pixel space -- that is the image the streamer sends, and it carries no downscale.
    // Filled BEFORE the solve loop and for every detected marker, including any whose solvePnP
    // fails below: the overlay is meant to show what the detector saw, not what survived the pose
    // solve. Two markers sharing an id collapse into one entry, exactly as they do in `result`.
    for (size_t i = 0; i < ids.size(); ++i) {
        PackedVector2Array pts;
        for (int c = 0; c < 4; ++c) {
            pts.push_back(Vector2(corners[i][c].x / DETECT_DOWNSCALE, corners[i][c].y / DETECT_DOWNSCALE));
        }
        corners_out[ids[i]] = pts;
    }

    double solve_ms = 0.0;                               // accumulated solvePnP time over all markers
    for (size_t i = 0; i < ids.size(); ++i) {
        // Per-id physical size, through the SAME function the GDScript uses to scale the rendered
        // patch -- so the pose's metric scale and the box drawn over the marker cannot disagree.
        const float half = get_marker_size(ids[i]) / 2.0f;
        const std::vector<cv::Point3f> obj_pts = {
            {-half,  half, 0.0f},
            { half,  half, 0.0f},
            { half, -half, 0.0f},
            {-half, -half, 0.0f}
        };
        cv::Mat rvec, tvec;
        int64_t t_solve = cv::getTickCount();
        bool ok2 = cv::solvePnP(
            obj_pts, corners[i], Kamera_matrix, distort_mat,
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

        result[ids[i]] = cam_to_world * Transform3D(basis, origin);
    }

    ACV_DBG("solvepnp_ms=", solve_ms, " markers=", (int)ids.size());
    return result;
}

//Debug-Overlay: projiziert WELT-Posen zurueck ins Pixelbild, durch dieselben Intrinsics, dieselbe
//Verzeichnung und dieselbe lens_pose wie die Detektion.
//
//Only useful ACROSS frames. Reprojecting the poses that came out of THIS frame with THIS frame's
//head pose cancels exactly -- detect_and_solve_all multiplied by head_pose * lens_pose and this
//divides by it again, so the square would land on the marker no matter how wrong lens_pose is.
//Fed the PREVIOUS frame's poses instead, it isolates lens_pose and the head-pose timing in PIXELS:
//a wrong lens pose makes a marker's world pose depend on the head pose it was solved at, so the
//drawn square walks off the marker as the head turns between the two frames. That is the AX = XB
//residual made visible, with passthrough reprojection and the eye projection out of the loop --
//which is what the naked-eye comparison in the headset cannot separate.
Dictionary OpenCVProcessor::project_marker_corners(Dictionary world_poses, const Transform3D &head_pose) const {
    Dictionary out;
    if (world_poses.is_empty()) {
        return out;
    }

    //NATIVE-Frame-Intrinsics, bewusst NICHT mit image_downscale_factor skaliert: der Downscale
    //existiert nur, um die Detektion billiger zu machen. Diese Pixel werden gegen die Ecken
    //verglichen, die detect_and_solve_all oben schon wieder aus dem Detektionsframe herausgeteilt
    //hat -- beide leben also im Pixelraum des ORIGINALBILDS, das der Streamer verschickt.
    cv::Mat K = (cv::Mat_<double>(3, 3) <<
        camera_intrinsics.x, 0.0, camera_intrinsics.z,
        0.0, camera_intrinsics.y, camera_intrinsics.w,
        0.0, 0.0, 1.0);

    const Transform3D world_to_cam = (head_pose * lens_pose).affine_inverse();

    const Array ids = world_poses.keys();
    for (int i = 0; i < ids.size(); ++i) {
        const int id = (int)ids[i];
        const Transform3D cam_pose = world_to_cam * (Transform3D)world_poses[ids[i]];

        //Macht den OpenCV->Godot-Basiswechsel rueckgaengig, der in jeder zurueckgegebenen Pose
        //steckt: F = diag(1, -1, -1), also T_cv = F * T_godot * F, und F ist zu sich selbst invers.
        //b[r][c] ist hier ZEILENweise (godot-cpp Basis haelt rows[3]), waehrend der Hinweg die Basis
        //aus SPALTEN gebaut hat -- daher sieht das Vorzeichenmuster transponiert aus.
        const Basis &b = cam_pose.basis;
        cv::Mat rot = (cv::Mat_<double>(3, 3) <<
             b[0][0], -b[0][1], -b[0][2],
            -b[1][0],  b[1][1],  b[1][2],
            -b[2][0],  b[2][1],  b[2][2]);
        const Vector3 &o = cam_pose.origin;
        cv::Mat tvec = (cv::Mat_<double>(3, 1) << (double)o.x, (double)-o.y, (double)-o.z);

        //Hinter der Kamera liefert projectPoints weiterhin Zahlen -- gespiegelten Unsinn. Ein
        //Marker, den dieser Frame gar nicht sehen kann, faellt hier raus statt als Geisterquadrat
        //irgendwo im Bild zu landen.
        if (tvec.at<double>(2) <= 1e-4) {
            continue;
        }

        cv::Mat rvec;
        cv::Rodrigues(rot, rvec);

        //Dieselben Objektpunkte und dieselbe Groessenquelle wie im solvePnP oben.
        const float half = get_marker_size(id) / 2.0f;
        const std::vector<cv::Point3f> obj_pts = {
            {-half,  half, 0.0f},
            { half,  half, 0.0f},
            { half, -half, 0.0f},
            {-half, -half, 0.0f}
        };
        std::vector<cv::Point2f> img_pts;
        cv::projectPoints(obj_pts, rvec, tvec, K, distort_mat, img_pts);

        PackedVector2Array pts;
        for (size_t c = 0; c < img_pts.size(); ++c) {
            pts.push_back(Vector2(img_pts[c].x, img_pts[c].y));
        }
        out[id] = pts;
    }
    return out;
}

//is given a frame the caller already owns (CameraX plugin on Quest, CameraServer on desktop) plus
//the head pose at that frame's capture time -- the one input that changes per frame and that only
//the caller can know. Everything else is a property.
Dictionary OpenCVProcessor::detect_markers(const Ref<Image> &image, const Transform3D &head_pose, Dictionary corners_out) {
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

    // Intrinsics, distortion, marker sizes and the lens pose all come from the properties now --
    // the distortion Mat in particular is built once in its setter instead of being rebuilt out of
    // a PackedFloat64Array on every single frame.
    return detect_and_solve_all(gray, head_pose, corners_out);
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
    detector->detectMarkers(frame, corners, ids);

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