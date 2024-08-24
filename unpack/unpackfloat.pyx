from libc.stdint cimport uint32_t, int32_t
from libc.stdlib cimport malloc, free
from cpython cimport PyObject, Py_INCREF
cimport cython

import numpy as np
cimport numpy as cnp

cnp.import_array()


cdef extern from "unpack_float_acphy.c":
    int32_t* unpack_float_acphy(int nbits, int autoscale, int shft, 
                       int fmt, int nman, int nexp, int32_t nfft, 
                       uint32_t* H)


cdef class ArrayWrapper:
    cdef void* data_ptr
    cdef int size

    cdef set_data(self, int size, void* data_ptr):
        """ Set the data of the array
        This cannot be done in the constructor as it must recieve C-level
        arguments.
        Parameters:
        -----------
        size: int
            Length of the array.
        data_ptr: void*
            Pointer to the data            
        """
        self.data_ptr = data_ptr
        self.size = size

    def __array__(self):
        """ Here we use the __array__ method, that is called when numpy
            tries to get an array from the object."""
        cdef cnp.npy_intp shape[1]
        shape[0] = <cnp.npy_intp> self.size
        # Create a 1D array, of length 'size'
        ndarray = cnp.PyArray_SimpleNewFromData(1, shape,
                                               cnp.NPY_INT32, self.data_ptr)
        return ndarray

    def __dealloc__(self):
        """ Frees the array. This is called by Python when all the
        references to the object are gone. """
        free(<void*>self.data_ptr)
        

def unpack_float(int nbits, int autoscale, int shft, int fmt, int nman, 
                 int nexp, int nfft, const uint32_t[:] H):
    cdef:
        uint32_t * ch
        int i

    ch = <uint32_t *> malloc(len(H)*cython.sizeof(uint32_t))
    if ch is NULL:
        raise MemoryError()
    for i in xrange(len(H)):
        ch[i] = H[i]
        
    cdef int32_t *arr
    cdef cnp.ndarray ndarr
    # Call the C function
    arr = unpack_float_acphy(nbits, autoscale, shft, fmt, nman, nexp, nfft, <uint32_t*>&ch[0])
    
    array_wrapper = ArrayWrapper()
    array_wrapper.set_data(2*len(H), <void*> arr) 
    ndarr = np.array(array_wrapper, copy=False)
    # Assign our object to the 'base' of the ndarray object
    ndarr.base = <PyObject*> array_wrapper
    # Increment the reference count, as the above assignement was done in
    # C, and Python does not know that there is this additional reference
    Py_INCREF(array_wrapper)

    return ndarr