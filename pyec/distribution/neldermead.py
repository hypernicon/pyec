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
   
   def __init__(self, config):
      super(NelderMeadHistory, self).__init__(config)
      self.attrs |= set(["state", "reflectScore", "shrinkIdx", "scale",
                         "current","centroid"])
      self.dim = self.config.space.dim
      self.scale = self.config.space.scale
      self.alpha = self.config.alpha
      self.beta = self.config.beta
      self.gamma = self.config.gamma
      self.delta = self.config.delta
      self.tolerance = self.config.tol
      self.current = None
      self.vertices = []
      self.state = NM_REFLECT_INIT
      self.centroid = None
      self.reflectScore = None
      self.shrinkIdx = None
      
   def refresh(self):
      self.vertices = sorted(self.vertices, key=lambda x:x[1])
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
            if self.config.initial is None:
               self.current = self.config.space.random()
            elif hasattr(self.config.initial, 'batch'):
               self.current = self.config.initial.batch(1)[0]
            else:
               self.current = self.config.initial()
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
            self.current[idx] += self.scale[idx]
            """
            if abs(self.current[idx] - self.config.center) > self.config.scale:
               self.current[idx] = self.config.center + self.config.scale
            """
         else:
            self.refresh()
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
            if (self.better(self.reflectScore, self.vertices[-2][1]) and
                not self.better(self.reflectScore, self.vertices[0][1])):
               self.state = NM_REFLECT
               self.vertices[-1] = pop[0]
               self.refresh()
               self.reflect()
            elif self.better(self.reflectScore, self.vertices[0][1]):
               self.expand()
               self.state = NM_EXPAND
            elif not self.better(self.reflectScore, self.vertices[-2][1]):
               self.contract()
               self.state = NM_CONTRACT
            else:
               self.shrink()
               self.state = NM_SHRINK
         # expand
         elif self.state == NM_EXPAND:
            if self.better(pop[0][1], self.reflectScore):
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
            if self.better(self.reflectScore, self.vertices[-1][1]):
               if not self.better(self.reflectScore, pop[0][1]):
                  self.state = NM_REFLECT
                  self.vertices[-1] = pop[0]
                  self.refresh()
                  self.reflect()
               else:
                  self.shrink()
                  self.state = NM_SHRINK
            else:
               if self.better(pop[0][1], self.vertices[-1][1]):
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
                   tol = 1e-10, # tolerance for vertex spread before restart
                   history = NelderMeadHistory)
   
   def __init__(self, **kwargs):
      super(NelderMead, self).__init__(**kwargs)
      if not isinstance(self.config.space, Euclidean):
         raise ValueError("Cannot use Nelder-Mead in non-Euclidean spaces.")
      self.config.populationSize = 1
    
   def compatible(self, history):
      return isinstance(history, NelderMeadHistory)
    
   def batch(self, popSize):
      # regardless of requested size, we only return 1 item, the current
      # proposal for nelder mead
      current = self.history.current
      if not self.config.space.in_bounds(current):
         center, scale = self.config.space.extent()
         self.history.current = np.minimum(center+scale,
                                           np.maximum(center-scale,current))
      return [self.history.current]
      
