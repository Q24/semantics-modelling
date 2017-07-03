#!/usr/bin/python
# -*- coding: utf-8 -*-
import dill
import numpy as np
import array
import os
def load(filename):
    cache = []
    with open(filename, "rb") as f:
        done = False
        while not done:
            while len(cache) < 1001:
                try:
                    cache.append(dill.load(f))
                except EOFError:
                    done = True
                    break

            for thing in cache:
                yield thing

            cache = []
class FileArray():

    # shape[0], shape[1], dtype
    NUM_HEADER_BYTES =  4+4+2


    def __init__(self, path, shape = None, dtype = 'f4'):
        self.path = path
        if os.path.exists(path):

            with open(path, 'rb') as f:
                f.seek(0)
                height = np.fromstring(f.read(4), dtype='i4')[0]
                f.seek(4)
                width = np.fromstring(f.read(4), dtype='i4')[0]
                if shape:
                    assert shape[0] == height and shape[1] == width

                self.shape = (height, width)
                f.seek(8)
                self.dtype = np.dtype(f.read(2))
                if dtype:
                    assert self.dtype.__reduce__()[1][0] == np.dtype(dtype).__reduce__()[1][0]
        else:
            assert shape
            self.shape = shape

            dtype_descriptor = np.dtype(dtype).__reduce__()[1][0]
            assert len(dtype_descriptor) == 2

            self.dtype = np.dtype(dtype_descriptor)

            with open(path, 'wb') as f:
                f.write('')
                f.seek((shape[0]*shape[1]*4) - 1)  # 1073741824 bytes in 1 GiB
                f.write("\0")
                f.seek(0)
                f.write(np.array(shape, dtype='i4').tostring())
                f.write(dtype_descriptor)


    def open(self):
        if hasattr(self,'f'):
            if not self.f.closed:
                print 'file already open'

        self.f = open(self.path, 'rb+')

    def close(self):
        if hasattr(self,'f'):
            self.f.close()

    def write(self, index, arr):
        self.f.seek(self.NUM_HEADER_BYTES + (index * self.dtype.itemsize * self.shape[1]))
        self.f.write(arr.tostring())



    def read(self, index):
        self.f.seek(self.NUM_HEADER_BYTES + (index * self.dtype.itemsize * self.shape[1]))
        return np.fromstring(self.f.read(self.dtype.itemsize * self.shape[1]), dtype=self.dtype)

    def read_chunk(self, index, rows):
        self.f.seek(self.NUM_HEADER_BYTES + (index * self.dtype.itemsize * self.shape[1]))
        retrieved = self.f.read(self.dtype.itemsize * self.shape[1]*rows)
        per_row_bytes = self.dtype.itemsize*self.shape[1]
        return [np.fromstring(substring, dtype=self.dtype) for substring in [retrieved[x:x+per_row_bytes] for x in xrange(0, len(retrieved), per_row_bytes)]]

    def write_chunk(self, index, arrays):
        self.f.seek(self.NUM_HEADER_BYTES + (index * self.dtype.itemsize * self.shape[1]))
        self.f.write(np.matrix(arrays, dtype=self.dtype).tostring())


if __name__ == '__main__':
    #save_embs_to_file()
    #'''
    arr = FileArray('testfile.bin', (100,10), dtype='f4')
    arr.open()

    arr.write_chunk(3, [range(10), range(10), range(10)])

    print arr.read_chunk(3, 3)

    arr.close()

    '''
    with open('testfile.bin', 'wb') as f:
        f.write('')
        f.seek(1073741824 - 1)  # 1073741824 bytes in 1 GiB
        f.write("\0")
    '''