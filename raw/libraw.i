/* -*- C -*-  (not really, but good for syntax highlighting) */

/* Wrapper for raw library */

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
