"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import traceback, sys

from numpy import *

from basic import StructureSearch, CyclicException, DuplicateEdgeException, IrreversibleEdgeException
from pyec.distribution.bayes.net import BayesNet


class StaticDecayProb(object):
   def __init__(self, p=0.3333333):
      self.p = p
      
   def __call__(self, network):
      return self.p ** network.decay

class ProbDecayInvEdgeRatio(object):
   def __init__(self, alpha=0.75):
      self.alpha = alpha

   def __call__(self, network):
      decay = network.decay
      edgeRatio = 1. - network.edgeRatio
      adjEdgeRatio = edgeRatio + 1./(len(network.variables)**2)
      num = 1. - (adjEdgeRatio ** self.alpha)
      denom = 1. - (edgeRatio ** self.alpha)
      if num <= 0.0 or denom <= 0.0:
         return 0.0
      return (.25 + .75*(1. - num / denom)) ** decay 

class ProbDecayEdgeRatio(object):
   def __init__(self, alpha=0.5):
      self.alpha = alpha
      
   def __call__(self, network):
      decay = network.decay
      adjEdgeRatio = 1./(len(network.variables)**2) + network.edgeRatio
      num = 1. - (adjEdgeRatio ** self.alpha)
      denom = 1. - (network.edgeRatio ** self.alpha)
      if denom == 0.0:
         return 0.0
      return (num / denom) ** decay 

class StructureProposal(StructureSearch):
   def __init__(
      self, 
      config,
      prem= ProbDecayInvEdgeRatio(), 
      padd= ProbDecayEdgeRatio(), 
      prev= ProbDecayEdgeRatio()
   ):
      """
         params:
         
         prem - function to prob of removing node given network
         padd - function to prob of adding node given network
         prev - function to prob of reversing an edge given network
      """
      self.prem = prem
      self.padd = padd
      self.prev = prev
      self.config = config
      self.network = None
      self.data = None

   def __getstate__(self):
      return {'prem':self.prem,'padd':self.padd,'prev':self.prev}
      
   def __setstate__(self, state):
      self.data = None
      self.network = None
      self.config = None
      self.prem = state['prem']
      self.prev = state['prev']
      self.padd = state['padd']
         
   def canReverse(self, newChild, newParent):
      check = super(StructureProposal, self).canReverse(newChild, newParent)
      if len(newChild.parents) >= self.config.branchFactor:
         return False
      return check
   
   def maybeChange(self, p, changer, existing=True):
      changed = False
      self.network.computeEdgeStatistics()
      try:
         if random.random_sample() < p(self.network):
            # pick a random edge
            if existing:
               if len(self.network.edges) == 0:
                  return False
               index = random.randint(0, len(self.network.edges))
               parent, child = self.network.edges[index]
               changer(parent.index, child, self.data)
            else:
               indexFrom = indexTo = 0
               exists = False
               attempts = 0
               maxAttempts = self.network.numVariables ** 2
               while (indexFrom == indexTo or exists) and (attempts < maxAttempts):
                  indexFrom = random.randint(0, len(self.network.variables))
                  indexTo = random.randint(0, len(self.network.variables))
                  exists = False
                  
                  for edge in self.network.edges:
                     if edge[0] == self.network.variables[indexFrom] and edge[1] == self.network.variables[indexTo]:
                        exists = True
                        break   
                
                  #check branch factor
                  if len(self.network.variables[indexTo].parents) > self.config.branchFactor:
                     exists = True
                     
                  if hasattr(self.config, 'cmpIdx'):
                     if self.network.variables[indexFrom].index != self.config.cmpIdx and self.network.variables[indexTo].index != self.config.cmpIdx:
                        exists = True
                
                  attempts += 1
               if attempts >= maxAttempts or exists: 
                  return False
               changer(self.network.variables[indexTo], self.network.variables[indexFrom], self.data)
            changed = True
            self.network.computeEdgeStatistics()
      except CyclicException, msg:
         pass
      except DuplicateEdgeException, msg:
         pass
      except IrreversibleEdgeException, msg:
         pass
      except Exception, msg:
         print Exception, msg
         traceback.print_exc(file=sys.stdout)
      return changed

   def batch(self, size, networks = None, data = None, **kwargs):
      if networks is None:
         networks = [None for i in xrange(size)]
      
      return [self.__call__(networks[i], data, **kwargs) for i in xrange(size)]

   def __call__(self, network=None, data=None, **kwargs):
      if network:
         self.network = network
      elif kwargs.has_key('prior'):
         self.network = kwargs['prior']
      else:
         self.network = BayesNet(self.config)
         self.network.updateVariables(self.config.data)
      if data:
         self.data = data
      else:
         self.data = self.config.data
      self.network.computeEdgeStatistics()
      
      total = 0
      changes = 1
      while total == 0: # changes > 0:
         changes = 0
         
         # remove an edge
         if self.maybeChange(self.prem, self.removeEdge):
            changes += 1 
         
         # reverse an edge
         if self.maybeChange(self.prev, self.reverseEdge):
            changes += 1
         
         # add an edge
         if self.maybeChange(self.padd, self.addEdge, False):
            changes += 1
         
         total += changes
      
      self.network.sort()
      
      return self.network
            
   def __getstate__(self):
      self.data = None
      self.network = None
      return self.__dict__
      
   def __setstate__(self,state):
      state['network'] = None
      state['data'] = None
      self.__dict__.update(state)
      
                     
   def adjust(self, acceptanceRatio):
      """Called by simulated annealing to update generation statistics"""

   def densityRatio(self, x, y, i):
      """Called by simulated annealing to adjust for assymetric densities
         The density governing this proposal should be symmetric.
         Still need to check this claim
      """
      return 1.0
         
         
