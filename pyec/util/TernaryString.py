"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from numpy import *
import binascii
import copy
import struct


def binary(x, digits=10):
   """Convert a number to its binary representation.
   
   :param x: The number to convert
   :type x: ``int`` or ``long``
   :param digits: The number of binary digits to convert, from right to left
   :type digits: ``int``
   :returns: A string of zeros and ones with the least significant bits at the
             left.
             
   """
   ret = ""
   num = x
   mask = 1L
   for i in xrange(digits):
      ret += str(num & mask)
      num >>= 1
   return ret
   

class TernaryString(object):
   """A ternary string with three values: True, False, and Unknown
   
   :params base: An object whose bits are treated as a bit string
   :type base: typically ``int`` or ``long``
   :params known: An object whose bits determine whether the value at that
                  bit is known in ``base``
   :type known: typically ``int`` or ``long``
   :params length: The maximum number of bits allowed
   :type length: ``int``
   
   """
   def __init__(self, base, known, length=10000):
      """
         base is an (integer) object whose bytes are treated as a bit string
         known is a mask to determine which values are indefinite
         length is the maximum number of bits
      """
      self.base = base
      self.known = known
      self.length = length
   
   def __str__(self):
      return binary(self.known & self.base, self.length)
      
   def __repr__(self):
      return binary(self.known & self.base, self.length)
   
   def __eq__(self, x):
      if isinstance(x, TernaryString):
         return ((self.known == x.known) and
                 ((self.base & self.known) == (x.base & x.known)) and
                 self.length == x.length)
      else:
         return ((self.known & self.base) == (self.known & x))
   
   def __ne__(self, x):
      return not self.__eq__(x)
   
   def __lt__(self, x):
      """Test whether the known portions are known and equal in the other"""
      return self.__le__(x) and self.__ne__(x)
   
   def __le__(self, x):
      """Test whether the known portions are known and equal in the other"""
      return (((self.known & x.known) == self.known) 
              and ((self.base & self.known) == (x.base & self.known)))

   def __irshift(self, x):
      self.base >>= x
      self.known >>= x
      return self

   def __rshift__(self, x):
      return TernaryString(self.base >> x, self.known >> x, self.length)
   
   __rrshift__ = __rshift__
   
   def __ilshift__(self, x):
      self.base <<= x
      self.known <<= x
      return self
   
   def __lshift__(self, x):
      return TernaryString(self.base << x, self.known << x, self.length)
   
   __rlshift__ = __lshift__
   
   def __ixor__(self, x):
      if isinstance(x, TernaryString):
         #where are they KNOWN to be different
         self.base ^= x.base
         self.known &= x.known
      else:
         self.base ^= x
      return self
   
   def __xor__(self, x):
      if isinstance(x, TernaryString):
         return TernaryString(self.base ^ x.base,
                              self.known & x.known,
                              self.length)
      else:
         return TernaryString(self.base ^ x, self.known, self.length)
   
   __rxor__ = __xor__
   
   def __iand__(self, x):
      if isinstance(x, TernaryString):
         self.base &= x.base
         self.known &= x.known
      else:
         self.base &= x
      return self
   
   def __and__(self, x):
      if isinstance(x, TernaryString):
         return TernaryString(self.base & x.base,
                              self.known & x.known,
                              self.length)
      else:
         return TernaryString(self.base & x, self.known, self.length)
   
   __rand__ = __and__

   def __ior__(self, x):
      if isinstance(x, TernaryString):
         self.base = self.base & self.known | x.base & x.known
         self.known |= x.known
      else:
         self.base |= x
      return self
   
   def __or__(self, x):
      if isinstance(x, TernaryString):
         return TernaryString(self.base & self.known | x.base & x.known,
                              self.known | x.known,
                              self.length)
      else:
         return TernaryString(self.base | x, self.known, self.length)
   
   __ror__ = __or__
   
   def __invert__(self):
      return TernaryString(~self.base, self.known, self.length)
   
   def __gt__(self, x):
      return self.__ge__(x) and self.__ne__(x)
   
   def __ge__(self, x):
      """Test whether the known portions are known and equal in the other"""
      return (((self.known & x.known) == x.known) 
              and ((self.base & x.known) == (x.base & x.known))) 
   
   def __add__(self, x):
      return self.__or__(x)
   
   def __mul__(self, x):
      if isinstance(x, ndarray):
         y = zeros(len(x))
         for i in xrange(len(y)):
            if self[i]:
               y[i] = x[i]
         return y
      return self.__and__(x)
            
   def __getitem__(self, i):
      """True if index i is known and equal to 1, else False"""
      if isinstance(i, slice):
         base = copy.copy(self.base)
         known = copy.copy(self.known)
         length = self.length
         if i.stop:
            mask = ((1L) << (i.stop)) - 1L
            known &= mask
            length = i.stop
         if i.start:
            base >>= i.start
            known >>= i.start
            length -= i.start
         return TernaryString(base, known, length)
      return ((self.base & (1L << i)) & self.known) != 0L
      
   def __setitem__(self, i, val):
      if isinstance(i, slice):
         mask = -1L
         if i.stop:
            mask = ((1L) << (i.stop)) - 1L
         if i.start:
            mask &= ~(((1L) << (i.start)) - 1L)
            
         if isinstance(val, TernaryString):
            self.known = ((self.known & ~mask) |
                          ((val.known << (i.start or 0)) & mask))
            self.base = (self.base & ~mask) | ((val.base << (i.start or 0)) & mask)
         else:
            self.known |= mask
            self.base = (self.base & ~mask) | ((val << (i.start or 0)) & mask)
         return
   
      mask = 1L << i
      val = bool(val)
      if val:
         self.known |= mask
         self.base |= mask
      else:
         self.known |= mask
         self.base &= ~mask
         
   def __delitem__(self, i):
      """Mark the item as unknown"""
      if isinstance(i, slice):
         mask = -1L
         if i.stop:
            mask = ((1L) << (i.stop + 1)) - 1L
         if i.start:
            mask &= ~(((1L) << (i.start + 1)) - 1L)
            
         self.known &= ~mask
         return
         
      mask = 1L << i
      self.known &= ~mask

   def __len__(self):
      return self.length
   
   def __iter__(self):
      mask = 1L
      for i in xrange(self.length):
         if (mask & self.base & self.known) > 0L:
            yield True
         else:
            yield False
         mask <<= 1
      raise StopIteration

      
   def distance(self, x):
      """hamming distance"""
      mask = 1L
      z = (self.base & self.known) ^ (x.base & x.known)
      total = 0
      for i in xrange(self.length):
         if (mask & z) > 0:
            total += 1
         mask <<= 1
      return total
         
   def toArray(self, numBits=None):
      if numBits is None:
         numBits = self.length
      x = []
      for i in xrange(numBits):
         x.append(self[i] and 1.0 or 0.0)
      return array(x)
      
   @classmethod   
   def fromArray(cls, arr):
      ret = TernaryString(0L, 0L, len(arr))
      for i in xrange(len(arr)):
         ret[i] = arr[i] > 0.0
      return ret

   @classmethod
   def random(cls, length):
      numBytes = int(ceil(length / 8.0))
      numFull  = length / 8
      initial = ''
      if numBytes != numFull:
         extra = length % 8
         initMask = 0
         for i in xrange(extra):
            initMask <<= 1
            initMask |= 1
         initial = struct.pack('B',initMask)
            
      base = long(binascii.hexlify(random.bytes(numBytes)), 16)
      known = long(binascii.hexlify(initial + '\xff'*numFull), 16)
      return TernaryString(base, known, length)   
