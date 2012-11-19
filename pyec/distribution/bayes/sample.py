"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from numpy import *
import binascii, struct
from pyec.util.TernaryString import TernaryString


class BayesRandomizer(object):
   def __call__(self, network):
      return random.randn(len(network.variables))

class BinaryRandomizer(BayesRandomizer):
   def __call__(self, network):
      numBytes = int(ceil(len(network.variables) / 8.0))
      numFull  = len(network.variables) / 8
      initial = ''
      if numBytes != numFull:
         extra = len(network.variables) % 8
         initMask = 0
         for i in xrange(extra):
            initMask <<= 1
            initMask |= 1
         initial = struct.pack('B',initMask)
            
      base = long(binascii.hexlify(random.bytes(numBytes)), 16)
      known = long(binascii.hexlify(initial + '\xff'*numFull), 16)
      return TernaryString(base, known)

class MultinomialRandomizer(BayesRandomizer):
   def __call__(self, network):
      cats = network.variables[0].categories
      ret = zeros(len(cats))
      for variable in network.variables:
         ret[variable.index] = random.randint(0, len(cats[variable.index])) + 1
      return ret


class BayesSampler(object):
   pass
   
   
class DAGSampler(BayesSampler):
   def __call__(self, network, initial = None):
      # order the variables in a DAG
      # call the variables one by one
      if initial is None:
         ret = network.randomize()
      else:
         ret = initial
      for variable in network.variables:
         if initial is None or not variable.isset(initial):
            ret = variable(ret)
      return ret


class GibbsSampler(BayesSampler):
   def __init__(self, equilibrium):
      self.equilibrium = equilibrium

   def __call__(self, network, initial = None):
   
      # get a random state
      ret = network.randomize()
      numNotSet = len(network.variables)
      if initial is not None:
         numNotSet = 0
         for variable in network.variables:
            if variable.isset(initial):
               ret[variable.index] = initial[variable.index]
         numNotSet += 1
      if numNotSet == 0:
         return initial
      if numNotSet == 1:
         for variable in network.variables:
            if initial is None or not variable.isset(initial):
               ret = variable.marginal(ret, network)
         return ret
      for i in xrange(self.equilibrium):
         index = random.randint(0, len(network.variables))
         while network.variables[index].isset(ret):
            index = random.randint(0, len(network.variables))
         ret = network.variables[index].marginal(ret, network)
      return ret
      

