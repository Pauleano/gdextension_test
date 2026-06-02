#!/usr/bin/env python
import os
import sys

from methods import print_error


libname = "snopek_tut"
projectdir = "project"

localEnv = Environment(tools=["default"], PLATFORM="")

# Build profiles can be used to decrease compile times.
# You can either specify "disabled_classes", OR
# explicitly specify "enabled_classes" which disables all other classes.
# Modify the example file as needed and uncomment the line below or
# manually specify the build_profile parameter when running SCons.

# localEnv["build_profile"] = "build_profile.json"

customs = ["custom.py"]
customs = [os.path.abspath(path) for path in customs]

opts = Variables(customs, ARGUMENTS)
opts.Update(localEnv)

Help(opts.GenerateHelpText(localEnv))

env = localEnv.Clone()

if not (os.path.isdir("godot-cpp") and os.listdir("godot-cpp")):
    print_error("""godot-cpp is not available within this folder, as Git submodules haven't been initialized.
Run the following command to download godot-cpp:

    git submodule update --init --recursive""")
    sys.exit(1)

env = SConscript("godot-cpp/SConstruct", {"env": env, "customs": customs})

opencv_header_files = [
    "opencv/build/install/include",
]

opencv_library_files = {
    'windows': [
        'opencv_world4140.lib',
    ],
    'macos': [
        'libopencv_core.dylib',
        'libopencv_imgcodecs.dylib',
        'libopencv_imgproc.dylib',
        'libopencv_videoio.dylib',
        'libopencv_objdetect.dylib',
        'libopencv_video.dylib',
        'libopencv_tracking.dylib'
    ],
    'linux': [
        'libopencv_core.so',
        'libopencv_imgcodecs.so',
        'libopencv_imgproc.so',
        'libopencv_videoio.so',
        'libopencv_objdetect.so',
        'libopencv_video.so',
        'libopencv_tracking.so'
    ]
}

opencv_library_path = {
    'windows': ['opencv/build/install/x64/vc17/lib'],
    'macos':   ['opencv/build/install/lib'],
    'linux':   ['opencv/build/install/lib'],
}

env.Append(CPPPATH=opencv_header_files)
env.Append(LIBPATH=opencv_library_path[env["platform"]])
env.Append(LIBS=opencv_library_files[env["platform"]])


sources = Glob("src/*.cpp")

# Create SharedLibrary

if env["platform"] == "macos":
    library = env.SharedLibrary(
        "demo/bin/godotopencvextension.{}.{}.framework/godotopencvextension.{}.{}".format(
            env["platform"], env["target"], env["platform"], env["target"]
        ),
        source=sources,
    )
else:
    library = env.SharedLibrary(
        "{}/bin/{}/{}{}{}".format(
            projectdir, env["platform"], libname, env["suffix"], env["SHLIBSUFFIX"]
        ),
        source=sources,
    )

opencv_runtime = []
if env["platform"] == "windows":
    opencv_runtime = env.Install(
        "{}/bin/{}".format(projectdir, env["platform"]),
        "opencv/build/install/x64/vc17/bin/opencv_world4140.dll",
    )

Default(library, opencv_runtime)
