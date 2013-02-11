"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
import pylab
from pyec.distribution.convolution import SelfConvolution
import time

class PylabDistribution2d(object):
    def __init__(self, center, scale, resolution=100, sleep=0, notebook=False):
        if not isinstance(center, np.ndarray):
            center = np.ones(2) * center
        self.center = center
        
        if not isinstance(scale, np.ndarray):
            scale = np.ones(2) * scale
        self.scale = scale
        
        self.res = resolution
        
        pylab.ion()
        self.shown = False
        self.notebook = notebook
        if not self.notebook:
            pylab.show()
            
        self.sleep = sleep
    
    def bucket(self, x):
        z = .5 + .5 * ((x - self.center) / self.scale)
        y = np.floor(z*self.res)
        return int(y[0]), int(y[1])
   
    def report(self, opt, pop):
        if isinstance(opt, SelfConvolution):
            opt = opt.opt
        grid = np.zeros((self.res, self.res))
        total = 0
        while total < 10000:
            p = opt()
            for x in p:
               if total < 10000:
                   try:
                      grid[self.bucket(opt.config.space.convert(x))] = 1./10000.
                   except IndexError:
                      pass
                   total += 1
        extentLow = list(self.center-self.scale)
        extentHigh = list(self.center+self.scale)
        pylab.clf()
        pylab.imshow(-np.log(grid), origin='lower',
                     extent=[extentLow[0], extentHigh[0], extentLow[1], extentHigh[1]])
        if self.notebook:
            pylab.show()
        else:
            pylab.draw()
        time.sleep(self.sleep)
        