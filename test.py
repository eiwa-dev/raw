import time
import raw
import matplotlib.pyplot as plt

print(hex(id(raw)))

tic = time.time()
print(tic)

i = raw.Raw('/home/jcarrano/imgdata/DSC02866.ARW')

toc = time.time()
print('TIME:', toc-tic)

print('#1')
idata = i.imgdata
r = idata.rawdata.raw_image
h = idata.rawdata.sizes.raw_height
w = idata.rawdata.sizes.raw_width

print('#2')
print(r,h,w)

plt.imshow(r, cmap = 'gray', interpolation = 'none')
plt.show()

tic = time.time()
print(tic)
i.raw2image()
toc = time.time()
print('TIME:', toc-tic)

p = idata.image
print(p.shape)
plt.imshow(p[...,:3], interpolation = 'none')
plt.show()
