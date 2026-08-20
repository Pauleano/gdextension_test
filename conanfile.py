from conan import ConanFile


class GdextensionOpenCV(ConanFile):
    """OpenCV als Conan-Dependency: aus den Quellen gebaut und statisch gelinkt.

    Der eigentliche `conan install` wird vollautomatisch aus der SConstruct
    heraus aufgerufen (siehe conan_support.py) -- auf der Zielplattform genuegt
    daher ein einziger `scons`-Aufruf.
    """

    settings = "os", "compiler", "build_type", "arch"
    generators = "SConsDeps"

    def requirements(self):
        # Neue cv::aruco::ArucoDetector-API (objdetect) -> OpenCV >= 4.7
        self.requires("opencv/4.10.0")

    default_options = {
        # Statisch in die Shared-Lib (GDExtension) linken -> PIC noetig.
        "opencv/*:shared": False,
        "opencv/*:fPIC": True,
        # Module, die OpenCVProcessor wirklich nutzt
        "opencv/*:objdetect": True,   # cv::aruco::ArucoDetector
        "opencv/*:calib3d": True,     # solvePnP / Rodrigues
        # ACHTUNG, seit 2026-08 nicht mehr zutreffend begruendet: imgcodecs stand hier fuer imread,
        # videoio fuer VideoCapture -- beide Aufrufer (get_6dof_of_all_aruco_patches_from_picture /
        # ..._from_webcam) sind geloescht, und im ganzen src/ ruft nichts mehr imread oder
        # VideoCapture auf. Beide koennen vermutlich auf False; das spart Buildzeit und nimmt dem
        # Android-.so das DT_NEEDED auf libmediandk (das ueber videoios cpp_info hereinkommt, siehe
        # SConstruct). ABSICHTLICH NOCH TRUE: ein Umlegen erzwingt einen kompletten
        # OpenCV-From-Source-Rebuild (~10-20 min), also eine bewusste Entscheidung, kein
        # Nebeneffekt eines Aufraeumcommits.
        "opencv/*:imgcodecs": True,   # frueher: imread
        "opencv/*:videoio": True,     # frueher: VideoCapture (Webcam, macOS -> AVFoundation)
        # Ungenutzte / schwere Module + grosse externe Deps abschalten,
        # damit der From-Source-Build kleiner und schneller wird.
        "opencv/*:dnn": False,
        "opencv/*:gapi": False,
        "opencv/*:ml": False,
        "opencv/*:photo": False,
        "opencv/*:stitching": False,
        "opencv/*:video": False,
        "opencv/*:highgui": False,
        "opencv/*:with_ffmpeg": False,
    }
