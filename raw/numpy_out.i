/* -*- C -*-  (not really, but good for syntax highlighting) */

/* Typemaps to wrap C pointers and arrays to numpy NDArrays
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

