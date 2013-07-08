# cython: profile=False
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
cpdef float concentric_spirals(RnnEvaluator net) except? -1:
    global best
    cdef float total = 0.0
    cdef float factor = (1./625.) #len(concentric))
    cdef list output, pt
    cdef float netCorr
    cdef int i, size
    cdef np.ndarray grid
    size = len(concentric)
    for i in xrange(25):#size):
        for j in xrange(25):
            k = i * 400 + j * 16
            net.clear()
            pt = [concentric[k]]
            output = net.call(pt, 25)
            if output is None:
                return -1
            netCorr = output[0][0] >= 0.0
            total += (concentricCorr[k] == netCorr) * factor
    if total > best:
        grid = np.zeros(size)
        for i in xrange(size):
            net.clear()
            pt = [concentric[i]]
            output = net.call(pt, 25)
            grid[i] = output[0][0]
    
        best = total
        #import pylab
        #pylab.imshow(grid.reshape((100,100)), origin='lower')
        #pylab.draw()
    return total

@cython.boundscheck(False)
cpdef float concentric_spirals_approx(RnnEvaluator net) except? -1:
    cdef float total = 0.0
    cdef int samples = 1000
    cdef float factor = (1./samples)
    cdef list output, pt
    cdef float netCorr
    cdef int i, idx, size
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
    if total > 0.65:
        return concentric_spirals(net)
    return total

@cython.boundscheck(False)
cpdef float concentric_spirals_approx_100(RnnEvaluator net) except? -1:
    cdef float total = 0.0
    cdef int samples = 100
    cdef float factor = (1./samples)
    cdef int i, idx, size
    cdef list output, pt
    cdef float netCorr
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


cdef extern from "CartPole.h":
    cdef cppclass CartPole:
        CartPole(int,int)
        void performAction(double, int)
        int outsideBounds()
        void init(int)
        double state[6]
        

@cython.boundscheck(False)
cpdef float cartDoublePoleMarkov(RnnEvaluator net):
    """Adapted from code in Ken Stanley's C++ NEAT code, originally from Faustino Gomez"""
    cdef int steps=0
    cdef float nmarkov_fitness

    cdef float jiggletotal # total jiggle in last 100
    cdef int count  #step counter
    
    cdef list input, output

    cdef CartPole *cartPole = new CartPole(0,1)
  
    cdef int nmarkovmax=100000

    cartPole.init(0);

    net.clear()
    
    while steps < nmarkovmax:
        steps += 1
        input = [[
            cartPole.state[0] / 4.8,
            cartPole.state[1] /2,
            cartPole.state[2]  / 0.52,
            cartPole.state[3] /2,
            cartPole.state[4] / 0.52,
            cartPole.state[5] /2,
        ]]
        output = net.call(input, 1)
        cartPole.performAction(.5*(output[0][0]) + .5,steps);
        if cartPole.outsideBounds():#// if failure
            break		# // stop it now

    return steps


@cython.boundscheck(False)
cpdef float cartDoublePole(RnnEvaluator net):
    """Adapted from code in Ken Stanley's C++ NEAT code, originally from Faustino Gomez"""
    cdef int steps=0
    cdef float nmarkov_fitness

    cdef float jiggletotal # total jiggle in last 100
    cdef int count  #step counter

    cdef list input, output

    cdef CartPole *cartPole = new CartPole(0,0)
  
    cdef int nmarkovmax=100000

    cartPole.init(0);

    net.clear()

    while steps < nmarkovmax:
        steps += 1
        input = [[
            cartPole.state[0] / 4.8,
            cartPole.state[2] / 0.52,
            cartPole.state[4] / 0.52,
        ]]
        output = net.call(input, 1)
        cartPole.performAction(.5*(output[0][0]) + .5,steps);
        if cartPole.outsideBounds():#// if failure
            break		# // stop it now

    del cartPole
    return steps
