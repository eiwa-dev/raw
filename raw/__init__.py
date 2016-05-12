"""Wrapper around libraw to implement RAII"""

__author__ = "Juan Ignacio Carrano <jc@eiwa.ag>"

from . import libraw

class Raw(libraw.LibRaw):
    """Wapper to make the LibRaw class safer"""

    _FORBIDDEN_METHODS = {'recycle', 'free_image',
                                        'open_file', 'open_buffer'}

    def __init__(self, filename, unpack = True):
        super(Raw, self).__init__()
        super(Raw, self).open_file(filename)

        if unpack:
            self.unpack()

    def __getattribute__(self, name):
        _allowed = object.__getattribute__(self, '_FORBIDDEN_METHODS')
        if name in _allowed:
            raise AttributeError("Method %s is disabled"%name)
        else:
            return object.__getattribute__(self, name)
