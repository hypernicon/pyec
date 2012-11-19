"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from numpy import *
from scipy.special import gamma, gammaln
from pyec.util.cache import LRUCache
import numpy as np      

class BayesianDirichletScorer(object):
   def __init__(self):
      self.cache = {}
      self.varCache = LRUCache()

   def matchesPrior(self, data, configuration):
      return 1.0
      
   def matches(self, data, config):
      """count number of instances of configuration in data"""
      cnt = 0L
      for x in data:
         if config <= x:
            cnt += 1
      return cnt
      
      
   def __call__(self, network, data):
      network.computeEdgeStatistics()
      total = 0.0
      total -= network.edgeRatio * len(data) * 10
      for variable in network.variables:
         varKey = str(variable.index) + str(variable.parents)
         if self.varCache.has_key(varKey):
            total += self.varCache[varKey]
            continue
         start = total
         for configuration in variable.configurations():
            prior = self.matchesPrior(data, configuration)
            total += gammaln(prior)
            total -= gammaln(prior + self.matches(data, configuration))
            for val in variable.values():
               priorVal = self.matchesPrior(data, configuration + val)
               total -= gammaln(priorVal)
               total += gammaln(priorVal + self.matches(data, configuration + val))
               
         self.varCache[varKey] = total - start
      return total / len(data)

class BayesianInformationCriterion(object):
   def __init__(self, lmbda = .5):
      self.lmbda = lmbda

   def __call__(self, network, data):
      network.computeEdgeStatistics()
      total = 0
      
      total -= np.log(network.numFreeParameters()) * self.lmbda * np.log(len(data))
      if len(network.changed) > 0:
         total += network.likelihoodChanged(data)
      else:
         total += network.likelihood(data)
      
      return total / len(data)
      
      
class ConditionalLogLikelihood(object):
   def __init__(self, index, lmbda = .5):
      self.lmbda = lmbda
      self.index = index

   def __call__(self, network, data):
      network.computeEdgeStatistics()
      total = 0
      
      total -= np.log(network.numFreeParameters()) * self.lmbda * np.log(len(data))
      total += network.conditionalLikelihood(self.index, data)
      
      
      return total / len(data)
