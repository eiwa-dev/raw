/* -*- C -*-  (not really, but good for syntax highlighting) */

// Typemaps to wrap C pointers and arrays to numpy NDArrays

%define _d1(x)
  {x}
%enddef
%define _d2(x,y)
  {x, y}
%enddef
%define _d3(x,y,z)
  {x,y,z}
%enddef

%define %arrayNDwrap(DATA_TYPECODE, NDIMS, dim_sizes)

  npy_intp dims[NDIMS] = dim_sizes;
  PyObject* obj = PyArray_SimpleNewFromData(NDIMS, dims, DATA_TYPECODE, (void*)($1));
  PyArrayObject* array = (PyArrayObject*) obj;

  if (!array) SWIG_fail;

  Py_INCREF(self); /* PyArray_SetBaseObject **steals** a reference */
  PyArray_SetBaseObject(array, self);

  $result = SWIG_Python_AppendOutput($result,obj);

%enddef

%define %numpy_outmaps(DATA_TYPE, DATA_TYPECODE)

%typemap(out) DATA_TYPE[ANY]
{
  %arrayNDwrap(DATA_TYPECODE, 1, _d1($1_dim0))
}

%typemap(out) DATA_TYPE[ANY][ANY]
{
  %arrayNDwrap(DATA_TYPECODE, 2, _d2($1_dim0, $1_dim1))
}

%enddef

%numpy_outmaps(signed char       , NPY_BYTE     )
%numpy_outmaps(unsigned char     , NPY_UBYTE    )
%numpy_outmaps(short             , NPY_SHORT    )
%numpy_outmaps(unsigned short    , NPY_USHORT   )
%numpy_outmaps(int               , NPY_INT      )
%numpy_outmaps(unsigned int      , NPY_UINT     )
%numpy_outmaps(long              , NPY_LONG     )
%numpy_outmaps(unsigned long     , NPY_ULONG    )
%numpy_outmaps(long long         , NPY_LONGLONG )
%numpy_outmaps(unsigned long long, NPY_ULONGLONG)
%numpy_outmaps(float             , NPY_FLOAT    )
%numpy_outmaps(double            , NPY_DOUBLE   )

