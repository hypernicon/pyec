"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
from pyec.util.TernaryString import TernaryString, binary
import unittest

class TestTernaryString(unittest.TestCase):
   def test_init(self):
      t = TernaryString(56L, 1023L, 10)
      self.assertEqual(t.base, 56L)
      self.assertEqual(t.known, 1023L, 10)
      self.assertEqual(t.length, 10)
      self.assertEqual(len(t), 10)
      
   def test_str(self):
      t = TernaryString(23L, 1023L, 10)
      self.assertEqual(str(t), '1110100000')
      self.assertEqual(repr(t), '1110100000')
      
   def test_eq(self):
      t1 = TernaryString(1L, 1023L, 10)
      t2 = TernaryString(1L, 1023L, 10)
      t3 = TernaryString(1L, 1021L, 10)
      t4 = TernaryString(1L, 1023L, 11)
      t5 = TernaryString(3L, 1021L, 10)
      t6 = TernaryString(3L, 1023L, 10)
      self.assertEqual(t1,t2)
      self.assertNotEqual(t1,t3)
      self.assertNotEqual(t1,t4)
      self.assertNotEqual(t1,t5)
      self.assertNotEqual(t1,t6)
      self.assertEqual(t3,t5)
      self.assertNotEqual(t5, t6)
      self.assertEqual(t1, 1L)
      self.assertEqual(t3, 3L)
      self.assertEqual(t3, 1L)
      self.assertNotEqual(t3, 5L)
      self.assertNotEqual(t1, 1023L)
      
   def test_cmp(self):
      t1 = TernaryString(1L, 1023L, 10)
      t2 = TernaryString(1L, 999L, 10)
      self.assertTrue(t1 <= t1)
      self.assertTrue(t2 <= t1)
      self.assertFalse(t1 < t1)
      self.assertTrue(t2 < t1)
      self.assertTrue(t1 > t2)
      self.assertFalse(t1 > t1)
      self.assertFalse(t2 > t1)
      t3 = TernaryString(3L, 7L, 10)
      self.assertFalse(t3 <= t1)
      self.assertFalse(t3 >= t1)
      self.assertFalse(t3 < t1)
      self.assertFalse(t3 > t1)
      
   def test_shift(self):
      t1 = TernaryString(273L, 962L, 5)
      t2 = t1 >> 2
      t3 = t1 << 2
      self.assertEqual(t2.base, 273L >> 2)
      self.assertEqual(t2.known, 962L >> 2)
      self.assertEqual(t2.length, 5)
      self.assertEqual(t3.base, 273L << 2)
      self.assertEqual(t3.known, 962L << 2)
      self.assertEqual(t3.length, 5)
      t1 <<= 2
      self.assertEqual(t1,t3)
      self.assertNotEqual(t1, t2)
      t1 >>= 4
      self.assertEqual(t1, t2)
      self.assertNotEqual(t1, t3)
      
   def test_xor(self):
      t1 = TernaryString(0L, 2L, 2)
      t2 = TernaryString(1L, 2L, 2)
      t3 = TernaryString(2L, 2L, 2)
      t4 = TernaryString(3L, 2L, 2)
      self.assertEqual(t1 ^ t2, t1)
      self.assertEqual(t1 ^ t3, t4)
      self.assertEqual(t1 ^ t3, t2 ^ t4)
      self.assertEqual(t1 ^ t3, t3 ^ t1)
      
      t5 = TernaryString(1L, 1L, 2)
      self.assertEqual(t5 ^ t1, TernaryString(3L, 0L, 2))
      
      t6 = TernaryString(1L, 3L, 2)
      self.assertEqual(t1^t6, t1)
      
      t7 = TernaryString(0L, 3L, 2)
      self.assertEqual(t6^t7, t6)
      
      self.assertEqual(t1 ^ 3, t4)
      self.assertEqual(t7 ^ 1, t6)
      
      t1 ^= t3
      self.assertEqual(t1, t4)
      
   def test_and(self):
      t1 = TernaryString(0L, 2L, 2)
      t2 = TernaryString(1L, 2L, 2)
      t3 = TernaryString(2L, 2L, 2)
      t4 = TernaryString(3L, 2L, 2)
      t5 = TernaryString(1L, 1L, 2)
      t6 = TernaryString(1L, 3L, 2)
      t7 = TernaryString(0L, 3L, 2)
      
      self.assertEqual(t1 & t2, t1)
      self.assertEqual(t1 & t6, t2)
      self.assertEqual(t6 & t7, t7)
      self.assertEqual(t3 & t4, t3)
      self.assertEqual(t5 & t7, t7 & t5)
      
      self.assertNotEqual(t6, t2)
      t6 &= t2
      self.assertEqual(t6, t2)
      
      self.assertEqual(t1 & 3, t1)
      self.assertEqual(t2 & 7, t1)
      self.assertEqual(t3 & 2, t3)
      
      self.assertNotEqual(t4.base, 2L)
      t4 &= 2L
      self.assertEqual(t4.base, 2L)
     
   def test_or(self):
      t1 = TernaryString(0L, 2L, 2)
      t2 = TernaryString(1L, 2L, 2)
      t3 = TernaryString(2L, 2L, 2)
      t4 = TernaryString(3L, 2L, 2)
      t5 = TernaryString(1L, 1L, 2)
      t6 = TernaryString(1L, 3L, 2)
      t7 = TernaryString(0L, 3L, 2)
      
      self.assertEqual(t1 | t2, t2)
      self.assertEqual(t1 | t6, t6)
      self.assertEqual(t6 | t7, t6)
      self.assertEqual(t3 | t4, t3)
      self.assertEqual(t5 | t7, t6)
      self.assertEqual(t5 | t7, t7 | t5)
      
      t6 |= t2
      self.assertEqual(t6.base, 1L)
      self.assertEqual(t6.known, 3L)
      
      self.assertEqual(t1 | 3, t4)
      self.assertEqual(t2 | 7, t3)
      self.assertEqual(t3 | 2, t3)
      
      t4 |= 2
      self.assertEqual(t4.base, 3L)
      self.assertEqual(t4.known, 2L)
      
   def test_dict(self):
      t1 = TernaryString(14L, 7L, 4)
      self.assertFalse(t1[0])
      self.assertTrue(t1[1])
      self.assertTrue(t1[2])
      self.assertFalse(t1[3])
      
      t1[3] = True
      self.assertTrue(t1[3])
      
      t1[3] = False
      self.assertFalse(t1[3])
      self.assertEqual(t1.known & 8L, 8L)
      
      del t1[3]
      self.assertEqual(t1.known & 8L, 0L)
      
      t1[2:4] = TernaryString(2L, 2L, 2)
      self.assertEqual(t1.base, 10L)
      self.assertEqual(t1.known, 11L)
      self.assertEqual(t1.length, 4)
      
      t2 = t1[1:3]
      self.assertEqual(t2.base % 4L, 1L)
      self.assertEqual(t2.known, 1L)
      self.assertEqual(t2.length, 2)
      
      t1[:2] = 3L
      self.assertEqual(t1.base, 11L)
      self.assertEqual(t1.known, 11L)
      self.assertEqual(t1.length, 4)
      
   def test_iter(self):
      t1 = TernaryString(14L, 7L, 4)
      bools = [bit for bit in t1]
      self.assertEqual(bools, [False, True, True, False])
      
   def test_distance(self):
      first = np.array([1,0,1,0,1,0,1], dtype=int)
      second = np.array([1,1,1,0,0,0,0], dtype=int)
      t1 = TernaryString.fromArray(first)
      t2 = TernaryString.fromArray(second)
      self.assertEqual(t1.distance(t2), t2.distance(t1))
      self.assertEqual(abs(first - second).sum(), t1.distance(t2))
      
   def test_arrays(self):
      first = np.array([1,0,1,0,1,0,1], dtype=int)
      second = np.array([1,1,1,0,0,0,0], dtype=int)
      t1 = TernaryString.fromArray(first)
      t2 = TernaryString.fromArray(second)
      self.assertEqual(t1.known, 127L)
      self.assertEqual(t1.base, 85L)
      self.assertEqual(t1.length, 7)
      self.assertEqual(t2.known, 127L)
      self.assertEqual(t2.base, 7L)
      self.assertEqual(t2.length, 7)
      self.assertTrue(np.abs(first - t1.toArray()).max() < 0.01)
      self.assertTrue(np.abs(second - t2.toArray()).max() < 0.01)
      
   def test_random(self):
      total = np.zeros(100)
      for i in xrange(10000):
         total += TernaryString.random(100).toArray() / 10000.0
      print "Total max: ", 
      self.assertTrue(np.abs(total - .5).max() < .025)
      
   