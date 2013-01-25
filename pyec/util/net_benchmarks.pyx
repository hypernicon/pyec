# cython: profile=True
# distutils: language = c++
"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import cython
import os.path
import numpy as np
cimport numpy as np
from pyec.distribution.nn.cnet cimport RnnEvaluator

from .net_benchmark_util import load_concentric

cdef np.ndarray concentric
cdef np.ndarray concentricCorr
concentric, concentricCorr = load_concentric()

cdef float best = 0.0

@cython.boundscheck(False)
cpdef float concentric_spirals(object net) except? -1:
    global best
    cdef float total = 0.0
    cdef float factor = (1./len(concentric))
    cdef list output, pt
    cdef int netCorr
    cdef int i, size
    cdef np.ndarray grid = np.zeros(10000)
    size = len(concentric)
    for i in xrange(size):
        net.clear()
        pt = [concentric[i]]
        output = net.call(pt, 25)
        netCorr = output[0][0] >= 0.0
        grid[i] = netCorr
        total += (concentricCorr[i] == netCorr) * factor
    if total > best:
       best = total
       import pylab
       pylab.imshow(grid.reshape((100,100)), origin='lower')
       pylab.draw()
    return total

@cython.boundscheck(False)
cpdef float concentric_spirals_approx(object net) except? -1:
    cdef float total = 0.0
    cdef int samples = 1000
    cdef float factor = (1./samples)
    cdef list output, pt
    cdef int netCorr
    cdef int i, idx, size
    size = len(concentric)
    for idx in xrange(samples):
        i = np.random.randint(0,size) 
        net.clear()
        pt = [concentric[i]]
        output = net.call(pt, 25)
        netCorr = output[0][0] >= 0.0
        total += (concentricCorr[i] == netCorr) * factor
    if total > 0.65:
        return concentric_spirals(net)
    return total

@cython.boundscheck(False)
cpdef float concentric_spirals_approx_100(object net) except? -1:
    cdef float total = 0.0
    cdef int samples = 100
    cdef float factor = (1./samples)
    cdef int i, idx, size
    cdef list output, pt
    cdef int netCorr
    size = len(concentric)
    for idx in xrange(samples):
        i = np.random.randint(0,size) 
        net.clear()
        pt = [concentric[i]]
        output = net.call(pt, 25)
        if output is None:
            return -1
        netCorr = output[0][0] >= 0.0
        total += (concentricCorr[i] == netCorr) * factor
    if total > 0.5:
        return concentric_spirals_approx(net)
    return total
