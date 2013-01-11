"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os.path
import numpy as np
cimport numpy as np

from .net_benchmark_util import load_concentric

cdef np.ndarray concentric
cdef np.ndarray concentricCorr
concentric, concentricCorr = load_concentric()

cpdef float concentric_spirals(object net):
    cdef float total = 0.0
    cdef float factor = (1./len(concentric))
    cdef int i, size
    size = len(concentric)
    for i in xrange(size):
        net.clear()
        netCorr = net([concentric[i]], times=25)[0][0] >= 0.0
        total += (concentricCorr[i] == netCorr) * factor
    return total

cpdef float concentric_spirals_approx(object net):
    cdef float total = 0.0
    cdef int samples = 1000
    cdef float factor = (1./samples)
    cdef int i, idx, size
    size = len(concentric)
    for idx in xrange(samples):
        i = np.random.randint(0,size) 
        net.clear()
        netCorr = net([concentric[i]], times=25)[0][0] >= 0.0
        total += (concentricCorr[i] == netCorr) * factor
    if total > 0.65:
        return concentric_spirals(net)
    return total

cpdef float concentric_spirals_approx_100(object net):
    cdef float total = 0.0
    cdef int samples = 100
    cdef float factor = (1./samples)
    cdef int i, idx, size
    size = len(concentric)
    for idx in xrange(samples):
        i = np.random.randint(0,size) 
        net.clear()
        netCorr = net([concentric[i]], times=25)[0][0] >= 0.0
        total += (concentricCorr[i] == netCorr) * factor
    if total > 0.5:
        return concentric_spirals_approx(net)
    return total
