import os
import subprocess
import sys

import setuptools
from setuptools.command.build_ext import build_ext

class CMakeExtension(setuptools.Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class BuildExt(build_ext):
    def run(self):
        self._check_build_environment()
        for ext in self.extensions:
            self.build_extension(ext)

    def _check_build_environment(self):
        """Check for required build tools: CMake, C++ compiler, and python dev."""
        try:
            subprocess.check_call(["cmake", "--version"])
        except OSError as e:
            ext_names = ", ".join(e.name for e in self.extensions)
            raise RuntimeError(
                "CMake must be installed to build" +
                f"the following extensions: {ext_names}") from e
        print("Found CMake")

        cxx = "clang++"
        if os.environ.get("CXX") is not None:
            cxx = os.environ.get("CXX")
        try:
            subprocess.check_call([cxx, "--version"])
        except OSError as e:
            ext_names = ", ".join(e.name for e in self.extensions)
            raise RuntimeError(
                "A C++ compiler that supports c++17 must be installed to build the "
                + "following extensions: {}".format(ext_names)
                + ". We recommend: Clang version >= 7.0.0."
            ) from e
        print("Found C++ compiler: {}".format(cxx))

    def build_extension(self, ext):
        extension_dir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cxx = "clang++"
        if os.environ.get("CXX") is not None:
            cxx = os.environ.get("CXX")
        env = os.environ.copy()
        cmake_args = [
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DCMAKE_CXX_COMPILER={cxx}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extension_dir}",
        ]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)

        subprocess.check_call(
            ["make", "poker-rnad", f"-j{os.cpu_count()}"],
            cwd=self.build_temp,
            env=env
        )


setuptools.setup(
    name="poker_rnad",
    version="0.0",
    license="MIT License",
    python_requires=">=3.12",
    ext_modules=[CMakeExtension("poker-rnad", sourcedir=".")],
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
    packages=setuptools.find_packages(include=["poker-rnad"]))
