#!/usr/bin/env python
import os
import platform
import shutil
import subprocess
import sys
import sysconfig

from methods import print_error


libname = "opencv_aruco"
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

# Camera2 NDK (libcamera2ndk/libmediandk, needed by OpenCV's videoio Android backend) only
# exists in the NDK sysroot from API 24+. godot-cpp defaults android_api_level to 21, where
# linking fails with "-lcamera2ndk". Pin it to 24 for android builds (matches the Conan
# android profile's os.api_level). Explicit android_api_level=... on the CLI still wins.
if ARGUMENTS.get("platform") == "android":
    ARGUMENTS.setdefault("android_api_level", "24")

opts = Variables(customs, ARGUMENTS)
opts.Update(localEnv)

Help(opts.GenerateHelpText(localEnv))

env = localEnv.Clone()

# macOS: standardmaessig fuer die Host-Architektur bauen statt godot-cpps Default
# "universal". Die OpenCV-Libs baut Conan fuer EINE Architektur (Host); ein universal
# Extension-Binary wuerde dagegen nicht linken. Ein explizites `arch=...` auf der
# Kommandozeile hat weiterhin Vorrang.
if sys.platform == "darwin" and "arch" not in ARGUMENTS:
    env["arch"] = "arm64" if platform.machine() == "arm64" else "x86_64"

# OpenCV-Header (flann etc.) nutzen C++-Exceptions; godot-cpp deaktiviert sie per
# Default (-fno-exceptions). Fuer dieses Projekt also Exceptions anlassen. Ein
# explizites `disable_exceptions=...` auf der Kommandozeile hat weiterhin Vorrang.
env["disable_exceptions"] = False

if not (os.path.isdir("godot-cpp") and os.listdir("godot-cpp")):
    print_error("""godot-cpp is not available within this folder, as Git submodules haven't been initialized.
Run the following command to download godot-cpp:

    git submodule update --init --recursive""")
    sys.exit(1)

env = SConscript("godot-cpp/SConstruct", {"env": env, "customs": customs})

sources = Glob("src/*.cpp")

# godot-cpp Architektur -> Conan settings.arch
conan_arch = {
    "arm64": "armv8", "x86_64": "x86_64", "x86_32": "x86", "arm32": "armv7",
}.get(env.get("arch"))


def _user_conan_path():
    """Pfad zum conan-Executable im User-Scripts-Verzeichnis (dort, wo
    `pip install --user` Scripts ablegt). Plattformabhaengig: Windows nutzt
    `Scripts\\conan.exe` inkl. Python-Versions-Unterordner, POSIX `.local/bin/conan`."""
    scheme = "nt_user" if os.name == "nt" else "posix_user"
    scripts = sysconfig.get_path("scripts", scheme)
    exe = "conan.exe" if os.name == "nt" else "conan"
    return os.path.join(scripts, exe)


def ensure_conan():
    """conan finden, bei Bedarf via pip installieren, Default-Profil sicherstellen.
    Plattformunabhaengig -- nur das eigentliche `conan install <ziel>` steht je
    Plattform-Branch (dort dupliziert)."""
    conan = shutil.which("conan") or _user_conan_path()
    if not os.path.isfile(conan):
        print("conan nicht gefunden -- installiere via pip ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--user", "conan>=2.0"], check=True)
        conan = _user_conan_path()
        if not os.path.isfile(conan):
            print_error("conan installiert, aber nicht gefunden unter: " + conan)
            sys.exit(1)
    if subprocess.run([conan, "profile", "path", "default"], capture_output=True).returncode != 0:
        subprocess.run([conan, "profile", "detect"], check=True)
    return conan


def conan_outdated(env, conan_out):
    """True, wenn fuer dieses Ziel noch kein aktuelles SConscript_conandeps vorliegt
    (erster Lauf oder conanfile.py geaendert). Bei `scons -c` nie bauen."""
    if env.GetOption("clean"):
        return False
    conandeps_file = os.path.join(conan_out, "SConscript_conandeps")
    return not (os.path.isfile(conandeps_file)
                and os.path.getmtime(conandeps_file) >= os.path.getmtime("conanfile.py"))


def merge_conan_deps(env, conan_out):
    """Conan-Dependency-Flags ins env mergen (plattformunabhaengig)."""
    conandeps_file = os.path.join(conan_out, "SConscript_conandeps")
    if os.path.isfile(conandeps_file):
        flags = SConscript(conandeps_file)["conandeps"]
        env.MergeFlags(flags)
        # godot-cpp linkt mit -undefined,dynamic_lookup -> $FRAMEWORKS aus MergeFlags
        # fallen weg; daher explizit als -framework auf die Linkzeile (macOS videoio).
        for framework_name in flags.get("FRAMEWORKS", []):
            env.Append(LINKFLAGS=["-framework", framework_name])


# Jede Plattform explizit: der plattformabhaengige `conan install <ziel>` steht hier
# (dupliziert). conan-Bootstrap (ensure_conan), Frische-Check (conan_outdated) und
# Flag-Merge (merge_conan_deps) sind ausgelagert, weil plattformunabhaengig.
# Ausgabe pro Ziel getrennt -> mehrere Plattformen aus EINEM Checkout.
if env["platform"] == "windows":
    conan_out = os.path.join("conan_install", "windows.{}.{}".format(env.get("arch") or "host", env["target"]))
    if conan_outdated(env, conan_out):
        settings = ["-s", "build_type=Release", "-s", "compiler.cppstd=17", "-s", "os=Windows"]
        if conan_arch:
            settings += ["-s", "arch=" + conan_arch]
        if not env.get("use_mingw", False):
            # C++-Runtime an godot-cpp angleichen (use_static_cpp -> /MT), sonst MD/MT-Mismatch.
            settings += ["-s", "compiler.runtime=" + ("static" if env.get("use_static_cpp", True) else "dynamic")]
        subprocess.run([ensure_conan(), "install", ".", "--output-folder", conan_out,
                        "--build=missing"] + settings, check=True)
    merge_conan_deps(env, conan_out)
    library = env.SharedLibrary(  # opencv_aruco.windows.<target>[.double].<arch>.dll (kein "lib")
        "{}/bin/windows/{}{}{}".format(projectdir, libname, env["suffix"], env["SHLIBSUFFIX"]),
        source=sources,
    )

elif env["platform"] == "linux":
    conan_out = os.path.join("conan_install", "linux.{}.{}".format(env.get("arch") or "host", env["target"]))
    if conan_outdated(env, conan_out):
        settings = ["-s", "build_type=Release", "-s", "compiler.cppstd=17", "-s", "os=Linux"]
        if conan_arch:
            settings += ["-s", "arch=" + conan_arch]
        subprocess.run([ensure_conan(), "install", ".", "--output-folder", conan_out,
                        "--build=missing"] + settings, check=True)
    merge_conan_deps(env, conan_out)
    library = env.SharedLibrary(  # libopencv_aruco.linux.<target>[.double].<arch>.so
        "{}/bin/linux/{}{}{}".format(projectdir, libname, env["suffix"], env["SHLIBSUFFIX"]),
        source=sources,
    )

elif env["platform"] == "android":
    conan_out = os.path.join("conan_install", "android.{}.{}".format(env.get("arch") or "host", env["target"]))
    if conan_outdated(env, conan_out):
        # Cross-Build: braucht ein Conan-Toolchain-Profil (NDK). Default ist das im Repo
        # eingecheckte profiles/android-arm64 (Jinja-Template: NDK-Pfad kommt zur Buildzeit
        # aus ANDROID_HOME, funktioniert daher auf Windows- wie Linux/WSL-Hosts).
        # CONAN_HOST_PROFILE ueberschreibt weiterhin (Profilname im Conan-Home oder Pfad).
        host_profile = os.environ.get("CONAN_HOST_PROFILE") or os.path.join("profiles", "android-arm64")
        subprocess.run([ensure_conan(), "install", ".", "--output-folder", conan_out,
                        "--build=missing", "-pr:h", host_profile], check=True)
    merge_conan_deps(env, conan_out)
    # OpenCV's videoio Android backend references the Camera2 NDK (libcamera2ndk/libmediandk);
    # without these the .so fails to load on device with "undefined symbol ACameraManager_create".
    env.Append(LIBS=["camera2ndk", "mediandk", "android", "log"])
    library = env.SharedLibrary(  # libopencv_aruco.android.<target>[.double].<arch>.so
        "{}/bin/android/{}{}{}".format(projectdir, libname, env["suffix"], env["SHLIBSUFFIX"]),
        source=sources,
        # Windows-Host: SHLIBPREFIX ist per Default "" -> ohne dieses "lib" hiesse die
        # Datei opencv_aruco.android...so und wuerde von der .gdextension (die den
        # lib-Praefix erwartet) nicht gefunden. Linux/macOS-Host setzt "lib" ohnehin.
        SHLIBPREFIX="lib",
    )

elif env["platform"] == "macos":
    conan_out = os.path.join("conan_install", "macos.{}.{}".format(env.get("arch") or "host", env["target"]))
    if conan_outdated(env, conan_out):
        settings = ["-s", "build_type=Release", "-s", "compiler.cppstd=17", "-s", "os=Macos"]
        if conan_arch:
            settings += ["-s", "arch=" + conan_arch]
        subprocess.run([ensure_conan(), "install", ".", "--output-folder", conan_out,
                        "--build=missing"] + settings, check=True)
    merge_conan_deps(env, conan_out)
    # minimales .framework; SHLIBPREFIX="" -> Binary-Name == Framework-Name (Godot-Anforderung).
    framework = "{}.macos.{}".format(libname, env["target"])
    library = env.SharedLibrary(
        "{}/bin/macos/{}.framework/{}".format(projectdir, framework, framework),
        source=sources,
        SHLIBPREFIX="",
    )

else:
    print_error(
        "Nicht unterstuetzte Plattform '{}'. Unterstuetzt: windows, linux, android, macos.".format(
            env["platform"]
        )
    )
    sys.exit(1)

Default(library)
