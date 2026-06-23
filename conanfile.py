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
        "opencv/*:parallel": "openmp",   # <-- threading backend for parallel_for_ (detectMarkers)
        # Module, die OpenCVProcessor wirklich nutzt
        "opencv/*:objdetect": True,   # cv::aruco::ArucoDetector
        "opencv/*:calib3d": True,     # solvePnP / Rodrigues
        "opencv/*:imgcodecs": True,   # imread
        "opencv/*:videoio": True,     # VideoCapture (Webcam, macOS -> AVFoundation)
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
