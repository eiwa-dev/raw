/* -*- C -*-  (not really, but good for syntax highlighting) */

/* Wrapper for raw library
 *
 * by Juan Carrano <jc@eiwa.ag>
 *
 * Copyright 2016 EIWA S.A.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above
 *   copyright notice, this list of conditions and the following disclaimer
 *   in the documentation and/or other materials provided with the
 *   distribution.
 * * Neither the name of the  nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

%module(package="raw") libraw

%feature("autodoc", "1");

%{
#include <libraw/libraw.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
%}

%include "typemaps.i"
%include "pybuffer.i"
%include "numpy_out.i"

/* This function loads the Numpy API.
 * Is HAS to be called when the module is loaded */
static int _import_array(void);

/* %apply float{[4]}; */

// Memory management
%typemap(out) SWIGTYPE * //libraw_data_t * //
{
   PyObject * r;

   r = SWIG_NewPointerObj(SWIG_as_voidptr($1), $1_descriptor, $owner | %newpointer_flags);
   //printf("about to set _c_parent! $1_name %s %p \n", __FUNCTION__, self);
   PyObject_SetAttrString(r, "_c_parent",
      (self == r)? Py_None: self); // avoid self reference

   $result = r;
}

// Array handling

%typemap(out) unsigned short *libraw_rawdata_t::raw_image
{
  //%array2dwrap(NPY_USHORT, arg1->sizes.raw_height , arg1->sizes.raw_width)
  %arrayNDwrap(NPY_USHORT, 2, _d2(arg1->sizes.raw_height , arg1->sizes.raw_width))
}

%typemap(out) unsigned short (*libraw_data_t::image)[4]
{
  %arrayNDwrap(NPY_USHORT, 3,
      _d3(arg1->sizes.iheight , arg1->sizes.iwidth, 4))
}

// Function typemaps

%define %pybuffer_binaryvoid(TYPEMAP, SIZE)
%typemap(in) (TYPEMAP, SIZE)
  (int res, Py_ssize_t size = 0, const void *buf = 0) {
  res = PyObject_AsReadBuffer($input, &buf, &size);
  if (res<0) {
    PyErr_Clear();
    %argument_fail(res, "(TYPEMAP, SIZE)", $symname, $argnum);
  }
  $1 = ($1_ltype) buf;
  $2 = ($2_ltype) size;
}
%enddef

%pybuffer_binaryvoid(char *BUFFER, size_t SIZE)
%apply (char *BUFFER, size_t SIZE) { (void *buffer, size_t size) };

// Niceties

%typemap(out) char** LibRaw::cameraList
{
   PyObject * r;
   int i, n_cameras = LibRaw::cameraCount();
   r = PyList_New(n_cameras);

   for (i = 0; i < n_cameras; i++) {
      PyList_SetItem(r, i, PyString_FromString($1[i]));
   }

   $result = r;
}

%include <libraw/libraw_types.h>
%include <libraw/libraw.h>

%pythoncode %{
_import_array()
%}
