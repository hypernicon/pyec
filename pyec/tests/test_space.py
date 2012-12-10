"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from pyec.space import *
import numpy as np
import unittest

class TestSpace(unittest.TestCase):
    def test_euclidean(self):
        dim5 = Euclidean(dim=5, center=10.0, scale=2.0)
        self.assertEqual(dim5.dim, 5)
        self.assertEqual(dim5.center, 10.0)
        self.assertEqual(dim5.scale, 2.0)
        
        # check in_bounds
        x = dim5.random()
        self.assertTrue(dim5.in_bounds(x))
        self.assertTrue((x == dim5.convert(x)).all())
        self.assertEqual(dim5.area(), 1.0)
        self.assertEqual(dim5.type, np.ndarray)
        y = np.random.randn(10)
        self.assertFalse(dim5.in_bounds(y))
        z = np.random.randn(2)
        self.assertFalse(dim5.in_bounds(z))
   
        # check extent
        lower, upper = dim5.extent()
        self.assertEqual(np.shape(lower), (dim5.dim,))
        self.assertEqual(np.shape(upper), (dim5.dim,))
        self.assertTrue((lower == -np.inf).all())
        self.assertTrue((upper == np.inf).all())
   
        # check the distribution
        ws = np.array([dim5.random() for i in xrange(2500)])
        mean = np.average(ws, axis=0)
        sd = np.sqrt(np.average((ws - mean)**2, axis=0))
        self.assertTrue(np.abs(mean - dim5.center).sum()/5. < 1e-1)
        self.assertTrue(np.abs(sd - dim5.scale).sum()/5. <  1e-1)
        
        # check gaussInt
        self.assertTrue(np.abs(dim5.gaussInt(0)-.5) <  1e-2)
        self.assertTrue(np.abs(dim5.gaussInt(1.0)-.8413) < 1e-2)
        self.assertTrue(np.abs(dim5.gaussInt(-1.0)-.1587) < 1e-2)
        
        # check proportion
        rectC = 9 * np.ones(5)
        rectS = np.ones(5)
        rectC2 = rectC.copy()
        rectC2[1] = 9.5
        rectS2 = rectS.copy()
        rectS2[1] = .5
        rect1 = Hyperrectangle(5, rectC,rectS)
        rect2 = Hyperrectangle(5, rectC2,rectS2)
        self.assertTrue(np.abs(dim5.proportion(rect1, dim5, 1) - 0.3413)
                        < 1e-2)
        self.assertTrue(np.abs(dim5.proportion(rect2, rect1, 1) - .1915 / .3413)
                        < 1e-2)                    
        
        # check hash
        for w in ws:
            self.assertNotEqual(dim5.hash(x), dim5.hash(w))
