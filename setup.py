#!/usr/bin/env python
#
#  Copyright 2016 EIWA S.A.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following disclaimer
#    in the documentation and/or other materials provided with the
#    distribution.
#  * Neither the name of the  nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

"""
setup.py file for RAW library
"""

import sys

# Super-ugly hack to allow us to build a sdist without having numpy
if not (len(sys.argv) > 1 and sys.argv[1] in ('sdist', '--name', '--version')):
    import numpy
    NUMPY_INCLUDE_DIRS = [numpy.get_include()]
else:
    NUMPY_INCLUDE_DIRS = []

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from distutils import spawn
from distutils.version import LooseVersion
import subprocess, re

__author__ = [  "Juan Carrano <jc@eiwa.ag>"]
__version__ = "0.1.1"

def get_swig_executable():
    "Get SWIG executable"

    # Find SWIG executable
    swig_executable = None
    swig_minimum_version = "3.0.3"
    for executable in ["swig", "swig3.0"]:
        swig_executable = spawn.find_executable(executable)
        if swig_executable is not None:
            # Check that SWIG version is ok
            output = subprocess.check_output([swig_executable, "-version"]).decode('utf-8')
            swig_version = re.findall(r"SWIG Version ([0-9.]+)", output)[0]
            if LooseVersion(swig_version) >= LooseVersion(swig_minimum_version):
                break
            swig_executable = None
    if swig_executable is None:
        raise OSError("Unable to find SWIG version %s or higher." % swig_minimum_version)
    print("Found SWIG: %s (version %s)" % (swig_executable, swig_version))

    return swig_executable


class Build_Ext_find_swig3(_build_ext):
    def find_swig(self):
        return get_swig_executable()

# We do some trickery to assure SWIG is always run before installing the
# generated files.
# http://stackoverflow.com/questions/12491328/python-distutils-not-include-the-swig-generated-module
from setuptools.command.install import install
from distutils.command.build import build

class CustomBuild(build):
    def run(self):
        self.run_command('build_ext')
        build.run(self)

class CustomInstall(install):
    def run(self):
        self.run_command('build_ext')
        install.run(self)

PY3 = sys.version_info[0] > 2
py3opt = ['-py3'] if PY3 else []

libraw_wrapper = Extension('raw._libraw',
            sources=['raw/libraw.i'],
            depends=['raw/numpy_out.i'],
            swig_opts=['-c++', '-builtin', '-relativeimport', '-lraw',
                        '-I/usr/include'] + py3opt,
            libraries = ['raw'],
            include_dirs=NUMPY_INCLUDE_DIRS
                           )

setup (name = 'raw',
       version = '0.1.3',
       description = """SWIG based LibRaw bindings""",
       long_description='Low level but pythonic bindings for libraw.',
       classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: BSD License',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Topic :: Multimedia :: Graphics :: Capture :: Digital Camera',
       ],
       keywords='camera raw images photos photography libraw',
       # url = ???????,
       install_requires=[
          'numpy',
       ],
       author      = 'Juan I Carrano',
       author_email='jc@eiwa.ag',
       license = 'BSD',
       ext_modules = [libraw_wrapper],
       packages = ["raw"],
       cmdclass = {"build_ext": Build_Ext_find_swig3,
                "build": CustomBuild, "install": CustomInstall}
       )
