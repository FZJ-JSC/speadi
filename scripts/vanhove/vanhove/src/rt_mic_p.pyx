# cython: profile=False
# cython: linetrace=False
# cython: binding=True
# cython: boundscheck=False
# cython: wraparound=False
# distutils: define_macros=CYTHON_TRACE_NOGIL=0
# distutils: define_macros=CYTHON_TRACE=0

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import parallel, prange
from libc.math cimport round, sqrt


def rt_mic_p(np.ndarray[float, ndim=3] chunk, long[:] g1,
             long[:] g2, np.ndarray[float, ndim=3] box_vectors):
    cdef float[:,:,:] bv = box_vectors
    cdef float[:,:] r1 = chunk[0, g1]
    cdef float[:,:,:] xyz = chunk[:, g2]

    cdef float[:,:,:] rt = np.zeros((chunk.shape[0], g1.shape[0], g2.shape[0]), dtype=np.float32)
    cdef float[:,:,:,:] rtd = np.zeros((chunk.shape[0], g1.shape[0], g2.shape[0], 3), dtype=np.float32)

    cdef int frames = chunk.shape[0]
    cdef int t, i, j, coord

    for t in prange(frames, nogil=True):
        for i in range(g1.shape[0]):
            for j in range(g2.shape[0]):
                rt[t][i][j] = 0
                for coord in range(3):
                    rtd[t][i][j][coord] = r1[i][coord] - xyz[t][j][coord]
                    rtd[t][i][j][coord] -= bv[t][coord][coord] * round(rtd[t][i][j][coord] / bv[t][coord][coord])
                    rtd[t][i][j][coord] = rtd[t][i][j][coord] ** 2
                    rt[t][i][j] += rtd[t][i][j][coord]
                rt[t][i][j] = sqrt(rt[t][i][j])
                rt[t][i][j]
                # remove self interaction part of G(r,t)
                if i == j:
                    rt[t][i][j] = 99.0

    cdef np.ndarray[float, ndim = 3] rt_array = np.asarray(rt)
    return rt_array
