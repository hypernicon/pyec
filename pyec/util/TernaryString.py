"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from numpy import *
import copy

def binary(x, digits=10):
   ret = ""
   num = x
   mask = 1L
   for i in xrange(digits):
      ret += str(num & mask)
      num >>= 1
   return ret

class TernaryString(object):
   """A ternary string with three values: True, False, and Unknown"""
   def __init__(self, base, known):
      """
         base is an (integer) object whose bytes are treated as a bit string
         known is a mask to determine which values are indefinite
      """
      self.base = base
      self.known = known
   
   def __str__(self):
      return str(self.known & self.base)
      
   def __repr__(self):
      return repr(self.known & self.base)
   
   def __eq__(self, x):
      if isinstance(x, TernaryString):
         return (self.known == x.known) and ((self.base & self.known) == (x.base & x.known))
      else:
         return (self.known & self.base) == (self.known & x)
   
   def __ne__(self, x):
      return (self.known != x.known) or ((self.base & self.known) != (x.base & x.known))
   
   def __lt__(self, x):
      """Test whether the known portions are known and equal in the other"""
      return self.__le__(x) and self.__ne__(x)
   
   def __le__(self, x):
      """Test whether the known portions are known and equal in the other"""
      return ((self.known & x.known) == self.known) \
         and ((self.base & self.known) == (x.base & self.known))

   
   def __gt__(self, x):
      return self.__ge__(x) and self.__ne__(x)
   
   def __ge__(self, x):
      """Test whether the known portions are known and equal in the other"""
      return ((self.known & x.known) == x.known) \
         and ((self.base & x.known) == (x.base & x.known)) 
   
   def __add__(self, x):
      return TernaryString(self.base | x.base, self.known | x.known)
   
   def __mult__(self, x):
      if isinstance(x, TernaryString):
         return TernaryString(self.base & x.base, self.known & x.known)
      elif isinstance(x, ndarray):
         y = zeros(len(x))
         for i in xrange(len(y)):
            if self[i]:
               y[i] = x[i]
         return y
      return None
            
   def __getitem__(self, i):
      """True if index i is known and equal to 1, else False"""
      if isinstance(i, slice):
         base = copy.copy(self.base)
         known = copy.copy(self.known)
         if i.stop:
            mask = ((1L) << (i.stop + 1)) - 1L
            known &= mask
         if i.start:
            base >>= i.start
            known >>= i.start
         return TernaryString(base, known)
      return ((self.base & (1L << i)) & self.known) != 0L
      
   def __setitem__(self, i, val):
      mask = 1L << i
      val = bool(val)
      if val:
         self.known |= mask
         self.base |= mask
      else:
         self.known |= mask
         self.base &= ~mask
      
   def distance(self, x, upTo):
      """hamming distance"""
      mask = 1L
      z = (self.base & ~x.base) | (~self.base & x.base)
      total = 0
      for i in xrange(upTo):
         if (mask & z) > 0:
            total += 1
         mask <<= 1
      return total
         
   def toArray(self, numBits):
      x = []
      for i in xrange(numBits):
         x.append(self[i] and 1.0 or 0.0)
      return array(x)
      
   @classmethod   
   def fromArray(cls, arr):
      ret = TernaryString(0L, 0L)
      for i in xrange(len(arr)):
         ret[i] = arr[i] > 0.0
      return ret

   """
      TODO: __len__, __getitem__, __setitem__, __str__
   """
      
