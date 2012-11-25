"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import copy
import numpy as np

from pyec.distribution.basic import PopulationDistribution
from pyec.config import Config
from pyec.history import History
from pyec.space import Euclidean

NM_REFLECT_INIT = 1
NM_REFLECT = 5
NM_EXPAND = 2
NM_CONTRACT = 3
NM_SHRINK = 4

class NelderMeadHistory(History):
   """A history that saves the state for the Nelder-Mead algorithm
   
   """
   attrs = ["dim", "alpha", "beta", "gamma", "delta", "tolerance",
            "state", "reflectScore", "shrinkIdx", "scale", "current",
            "centroid"]
   
   def __init__(self, dim, scale, alpha, beta, gamma, delta, tolerance):
      super(NelderMeadHistory, self).__init__()
      self.dim = dim
      self.scale = scale
      self.alpha = alpha
      self.beta = beta
      self.gamma = gamma
      self.delta = delta
      self.tolerance = tolerance
      self.current = None
      self.vertices = []
      self.state = NM_REFLECT_INIT
      self.centroid = None
      self.reflectScore = None
      self.shrinkIdx = None
      
   def __getstate__(self):
      state = super(NelderMeadHistory, self).__getstate__()
      
      for attr in self.attrs:
         val = getattr(self, attr)
         if isinstance(val, np.ndarray):
            val = val.copy()
         state[attr] = val
         
      return state

   def __setstate_(self, state):
      state = copy.copy(state)
      
      for attr in self.attrs:
         val = state.pop(attr)
         if isintance(val, np.ndarray):
            val = val.copy()
         setattr(self, attr, val)
      
      super(NelderMeadHistory, self).__setstate__(state)
      
   def refresh(self):
      self.vertices = sorted(self.vertices, key=lambda x:x[1], reverse=True)
      if len(self.vertices) > 1:
         self.centroid = np.sum(np.array([v[0] for v in self.vertices[:-1]]),
                                axis=0) / (len(self.vertices) - 1)   
         dist = 0.0
         for v, s in self.vertices:
            dist += (np.sqrt(((self.centroid - v) ** 2).sum())
                     / len(self.vertices))
            
         if dist < self.tolerance:
            self.vertices = []
            self.centroid = None
            self.reflectScore = None
            self.shrinkIdx = None
            self.current = None
            self._empty = True
            self.state = NM_REFLECT_INIT   
   
   def reflect(self):
      if self.centroid is not None:
         self.current = (self.centroid +
                         self.alpha * (self.centroid - self.vertices[-1][0]))
      
   def expand(self):
      if self.centroid is not None:
         self.reflect()
         self.current = (self.centroid +
                         self.gamma * (self.current - self.centroid))
    
   def contract(self):
      if self.centroid is not None:
         if self.reflectScore > self.vertices[-1][1]:
            self.reflect()
            self.current = (self.centroid +
                            self.beta * (self.current - self.centroid))
         else:
            self.current = (self.centroid +
                            self.beta * (self.vertices[-1][0] - self.centroid))
         
   def shrink(self):
      if self.centroid is not None:
         for i in xrange(len(self.vertices) - 1):
            self.vertices[i+1] = [self.vertices[0][0] +
                                  self.delta * (self.vertices[i+1][0] -
                                                self.vertices[0][0]), None]
         self.shrinkIdx = 1
         self.current = self.vertices[self.shrinkIdx][0]

   def internalUpdate(self, pop):
      startState = self.state
      self.current = None
      if len(self.vertices) < self.dim + 1:
         self.vertices.append(pop[0])
         if len(self.vertices) < self.dim + 1:
            self.current = self.vertices[0][0].copy()
            idx = len(self.vertices) - 1
            self.current[idx] += self.scale
            """
            if abs(self.current[idx] - self.config.center) > self.config.scale:
               self.current[idx] = self.config.center + self.config.scale
            """
         else:
            self.refresh()
      if self.current is None:
         # reflect
         if self.state == NM_REFLECT_INIT:
            self.reflect()
            self.state = NM_REFLECT
         # check reflect
         elif self.state == NM_REFLECT:
            self.reflectScore = pop[0][1]
            if (self.reflectScore > self.vertices[-2][1] and
                self.reflectScore <= self.vertices[0][1]):
               self.state = NM_REFLECT
               self.vertices[-1] = pop[0]
               self.refresh()
               self.reflect()
            elif self.reflectScore > self.vertices[0][1]:
               self.expand()
               self.state = NM_EXPAND
            elif self.reflectScore <= self.vertices[-2][1]:
               self.contract()
               self.state = NM_CONTRACT
            else:
               self.shrink()
               self.state = NM_SHRINK
         # expand
         elif self.state == NM_EXPAND:
            if pop[0][1] > self.reflectScore:
               self.state = NM_REFLECT
               self.vertices[-1] = pop[0]
               self.refresh()
               self.reflect()      
            else:
               self.state = NM_REFLECT
               self.reflect()
               self.vertices[-1] = (self.current, self.reflectScore)
               self.refresh()
               self.reflect()
         # contract
         elif self.state == NM_CONTRACT:
            if self.reflectScore > self.vertices[-1][1]:
               if pop[0][1] >= self.reflectScore:
                  self.state = NM_REFLECT
                  self.vertices[-1] = pop[0]
                  self.refresh()
                  self.reflect()
               else:
                  self.shrink()
                  self.state = NM_SHRINK
            else:
               if pop[0][1] > self.vertices[-1][1]:
                  self.state = NM_REFLECT
                  self.vertices[-1] = pop[0]
                  self.refresh()
                  self.reflect()
               else:
                  self.shrink()
                  self.state = NM_SHRINK
            
         # shrink
         elif self.state == NM_SHRINK:
            self.vertices[self.shrinkIdx] = pop[0]
            self.shrinkIdx += 1
            if self.shrinkIdx >= len(self.vertices):
               self.shrinkIdx = None
               self.state = NM_REFLECT
               self.refresh()
               self.reflect()
            else:
               self.current = self.vertices[self.shrinkIdx][0]
            
         else:
            print "MISSING STATE: ", self.state
      
      
      if self.current is None:
         print startState
         print self.state
         print self.vertices
         raise Exception, "current is none"
 

   
   
class NelderMead(PopulationDistribution):
   """The Nelder-Mead method of optimization for real spaces.
   
   """
   config = Config(alpha = 1.0,
                   beta = .5,
                   gamma = 2.,
                   delta = .5,
                   tol = 1e-10) # tolerance for vertex spread before restart
   
   def __init__(self, **kwargs):
      super(NelderMead, self).__init__(**kwargs)
      if not isinstance(self.config.space, Euclidean):
         raise ValueError("Cannot use Nelder-Mead in non-Euclidean spaces.")
      self.config.populationSize = 1
      def history():
         return NelderMeadHistory(self.config.space.dim,
                                  self.config.space.scale,
                                  self.config.alpha,
                                  self.config.beta,
                                  self.config.gamma,
                                  self.config.delta,
                                  self.config.tol)
      self.config.history = history
    
   def compatible(self, history):
      return isinstance(history, NelderMeadHistory)
    
   def batch(self, popSize):
      # regardless of requested size, we only return 1 item, the current
      # proposal for nelder mead
      center = self.config.space.center
      scale = self.config.space.center
      current = self.history.current
      self.history.current = np.minimum(center+scale,
                                        np.maximum(center-scale,current))
      return [self.history.current]
      
