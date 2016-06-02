"""Wrapper around libraw to implement RAII"""

__author__ = "Juan Ignacio Carrano <jc@eiwa.ag>"
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
