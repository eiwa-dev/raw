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

"""Wrapper around libraw to implement RAII"""

__author__ = [  "Juan Carrano <jc@eiwa.ag>"]
__version__ = "0.1.1"

from . import libraw

class Raw(libraw.LibRaw):
    """Wapper to make the LibRaw class safer.

    This class implements RAII.

    These methods of LibRaw cannot be called:
        'recycle', 'free_image', 'open_file', 'open_buffer'
    """

    _FORBIDDEN_METHODS = {'recycle', 'free_image',
                                        'open_file', 'open_buffer'}

    def __init__(self, filename = None, buffer = None, unpack = True):
        """Initialize LibRaw and open an image.
        Specify "filename" if the image is on a file on disk.
        Specify "buffer" if the image is on a buffer on memory.

        If unpack is True, the image will be decoded (LibRaw.unpack()
        will be called).
        """
        if ((filename is None and buffer is None) or
            (filename is not None and buffer is not None)):
            raise ValueError("You must specify EITHER filename or buffer")

        super(Raw, self).__init__()

        if filename is not None:
            super(Raw, self).open_file(filename)
        else:
            super(Raw, self).open_buffer(buffer)

        if unpack:
            self.unpack()

    def __getattribute__(self, name):
        """Proxy that forwards all attribute accesses to the superclass,
        except those methods that are forbidden."""
        _allowed = object.__getattribute__(self, '_FORBIDDEN_METHODS')
        if name in _allowed:
            raise AttributeError("Method %s is disabled"%name)
        else:
            return object.__getattribute__(self, name)
