swig3.0 -python -c++ -builtin -I/usr/include libraw.i
g++ -dA -ggdb -fPIC -c libraw_wrap.cxx -I/usr/include/python2.7
g++ -dA -ggdb -shared -lraw libraw_wrap.o -o _libraw.so
