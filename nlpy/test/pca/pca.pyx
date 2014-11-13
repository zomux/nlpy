# distutils: language = c++

from cpython cimport array
from array import array
from cython.parallel import parallel, prange
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libcpp.vector cimport vector
cdef extern from "stdio.h":
    int printf(char *format, ...) nogil

cdef class KL(object):

    cdef int* a
    cpdef vector[int] vect


    def __init__(self):
        self.a = <int *> PyMem_Malloc(5 * sizeof(int))
        self.a[2] = 533


    cpdef reduce(self):
        cdef int i
        for i in prange(self.vect.size(), nogil=True, num_threads=8):
            self.vect[i] = self.vect[i] ** 2

    cpdef add(self, int i):
        self.vect.push_back(i)

    cpdef int read(self, int i):
        return self.vect[i]




def hey(li):

    cdef int *a = <int *> PyMem_Malloc(len(li) * sizeof(int))
    cdef int l = len(li)
    cdef Py_ssize_t i
    for i in prange(l, nogil=True):
        a[i] = 1
    print a[2]