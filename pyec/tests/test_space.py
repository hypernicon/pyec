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
        self.assertTrue((dim5.center - 10.0 < 1e-2).all())
        self.assertTrue((dim5.scale - 2.0 < 1e-2).all())
        
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
        rect1 = Hyperrectangle(rectC - rectS, rectC + rectS)
        rect2 = Hyperrectangle(rectC2 - rectS2, rectC2 + rectS2)
        self.assertTrue(np.abs(dim5.proportion(rect1, dim5, 1) - 0.3413)
                        < 1e-2)
        self.assertTrue(np.abs(dim5.proportion(rect2, rect1, 1) - .1915 / .3413)
                        < 1e-2)                    
        
        # check hash
        for w in ws:
            self.assertNotEqual(dim5.hash(x), dim5.hash(w))

    def test_hyperrectangle(self):
        center = np.arange(5)
        scale = np.arange(5) + 1.0
        rect = Hyperrectangle(center - scale, center + scale)
        self.assertEqual(rect.dim, 5)
        self.assertTrue((rect.center == np.arange(5)).all())
        self.assertTrue((rect.scale == 1.0 + np.arange(5)).all())
        self.assertTrue(rect.in_bounds(np.arange(5)))
        self.assertTrue(rect.in_bounds(1.0 + 2*np.arange(5)))
        self.assertFalse(rect.in_bounds(np.array([-2.,100.,-3,200.,-1000.])))
        self.assertFalse(rect.in_bounds(np.array([0.,1.,5.1,3.,4.0])))
        
        # test extent
        lower, upper = rect.extent()
        self.assertTrue((lower == -1.0).all())
        self.assertTrue((upper == 1.0 + 2*np.arange(5)).all())
        
        # test area and proportion
        rect2 = Hyperrectangle(-1.0*np.ones(5), 1.0*np.ones(5))
        center = np.zeros(5)
        center[2] = -0.5
        scale = np.ones(5)
        scale[2] = 0.5
        rect3 = Hyperrectangle(center-scale, center+scale)
        dim5 = Euclidean(5, 0.0, 1.0)
        self.assertTrue(np.abs(rect.area() - 3840.0) < 1e-2)
        self.assertTrue(np.abs(rect2.area() - 32.0) < 1e-2)
        rect3.owner = rect2
        rect3.parent = rect2
        self.assertTrue(np.abs(rect3.area(index=2) - 16.0) < 1e-2)
        rect3._area = None
        self.assertTrue(np.abs(rect3.area() - 16.0) < 1e-2)
        rect3._area = None
        rect3.owner = dim5
        rect2._area = .6826 ** 5
        rect2.owner = dim5
        rect2.parent = dim5
        self.assertTrue(np.abs(rect3.area(index=2) - .3413 * (.6826 ** 4)) < 1e-2)
        self.assertTrue(np.abs(dim5.proportion(rect3,rect2,2) - .5) < 1e-2)
        self.assertTrue(np.abs(rect.proportion(rect3,rect2, 2) - .5) < 1e-2)
        
        # test random
        x = rect.random()
        self.assertTrue(rect.in_bounds(x))
        ws = np.array([rect.random() for i in xrange(2500)])
        self.assertTrue(np.abs(np.average(ws, axis=0)-rect.center).max() < 1e-1)
        self.assertTrue(np.abs(ws.max(axis=0) - upper).max() < 1e-1)
        self.assertTrue(np.abs(ws.min(axis=0) - lower).min() < 1e-1)
        onecenter = upper.copy()
        onecenter[0] = rect.center[0]
        above = (ws <= onecenter).all(axis=1).sum()
        below = (ws > onecenter).any(axis=1).sum()
        self.assertTrue(np.abs(above - below) < 100)
        
    def test_binary(self):
        bin = Binary(10)
        self.assertEqual(bin.dim, 10)
        self.assertEqual(bin.area(), 1.0)
        
        # test extent
        lower, upper = bin.extent()
        self.assertEqual(lower.base, 0L)
        self.assertEqual(lower.known, 0L)
        self.assertEqual(upper.base, -1L)
        self.assertEqual(upper.known, 0L)
        
        # test random
        x = bin.random()
        self.assertEqual(((1L << 10) - 1L), (x.known & ((1L << 10) - 1L)))
        self.assertTrue(bin.in_bounds(x))
        idx = np.random.randint(0,10)
        above = 0
        below = 0
        for i in xrange(2500):
            z = bin.random()
            if z[idx]:
                above += 1
            else:
                below += 1
        self.assertTrue(np.abs(above-below) < 250)
        
    def test_binary_real(self):
        bin = BinaryReal(2, 4, 10.0, 2.0)
        self.assertEqual(bin.dim, 8)
        self.assertEqual(bin.realDim, 2)
        self.assertEqual(bin.bitDepth, 4)
        self.assertTrue((bin.center == 10.0).all())
        self.assertTrue((bin.scale == 2.0).all())
        self.assertTrue((bin.adj == 8.0).all())
        self.assertTrue((bin.scale2 == 4.0).all())
        
        # test convert
        from pyec.util.TernaryString import TernaryString
        point = TernaryString(0L,-1L,8)
        self.assertTrue(np.abs(bin.convert(point) - np.array([8.0,8.0])).max()
                        < 1e-2)
        point = TernaryString(-1L,-1L, 8)
        self.assertTrue(np.abs(bin.convert(point) - np.array([11.75,11.75])).max()
                        < 1e-2)
        point = TernaryString((1L << 4) | 1L, -1L, 8)
        self.assertTrue(np.abs(bin.convert(point) - np.array([10.0,10.0])).max()
                        < 1e-2)
        
    def test_binary_rectangle(self):
        from pyec.util.TernaryString import TernaryString
        rect = BinaryRectangle(TernaryString(8L,10L,4))
        
        # test in_bounds
        point = TernaryString(15L, 15L, 4)
        self.assertFalse(rect.in_bounds(point))
        point = TernaryString(13L, 15L, 4)
        self.assertTrue(rect.in_bounds(point))
        
        # test extent
        lower, upper = rect.extent()
        self.assertEqual((lower.base & 15L), 8L)
        self.assertEqual((lower.known & 15L), 15L)
        self.assertEqual((upper.base & 15L), 13L)
        self.assertEqual((upper.known & 15L), 15L)
        
        # test area
        self.assertTrue(np.abs(rect.area() - .25) < 1e-2)
        rect._area = None
        rect.parent = BinaryRectangle(TernaryString(8L,8L,4))
        print rect.area()
        self.assertTrue(np.abs(rect.area() - .25) < 1e-2)
        
        # test random
        options = [8L,9L,12L,13L]
        counts = [0,0,0,0]
        for i in xrange(2500):
            x = rect.random()
            self.assertTrue(rect.in_bounds(x))
            self.assertTrue((x.base & 15L) in options)
            counts[options.index(x.base & 15L)] += 1
        for cnt in counts:
            self.assertTrue(np.abs(cnt - 625) < 50)
        