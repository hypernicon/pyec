"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from numpy import *
from pyec.distribution.basic import PopulationDistribution, FixedCube, Gaussian
from pyec.config import Config, ConfigBuilder

NM_REFLECT_INIT = 1
NM_REFLECT = 5
NM_EXPAND = 2
NM_CONTRACT = 3
NM_SHRINK = 4

class NMConfigurator(ConfigBuilder):
   
   def __init__(self, *args):
      super(NMConfigurator, self).__init__(NelderMead)
      self.cfg.restartTolerance = 1e-10
      self.cfg.usePrior = False
      self.cfg.initialDistribution = Gaussian(self.cfg)
   
class NelderMead(PopulationDistribution):
   def __init__(self, cfg):
      super(NelderMead, self).__init__(cfg)
      self.current = self.config.initialDistribution()
      self.vertices = []
      self.state = NM_REFLECT_INIT
      self.centroid = None
      self.reflectScore = None
      self.shrinkIdx = None
      self.alpha = hasattr(cfg,'alpha') and cfg.alpha or 1.
      self.beta = hasattr(cfg,'beta') and cfg.beta or .5
      self.gamma = hasattr(cfg,'gamma') and cfg.gamma or 2.
      self.delta = hasattr(cfg,'delta') and cfg.delta or .5
      
      
   @classmethod
   def configurator(cls):
      return NMConfigurator(cls)

   def batch(self, popSize):
      # regardless of requested size, we only return 1 item, the current
      # proposal for nelder mead
      return [self.current]
      
   def refresh(self):
      self.vertices = sorted(self.vertices, key=lambda x:x[1], reverse=True)
      if len(self.vertices) > 1:
         self.centroid = sum(array([v[0] for v in self.vertices[:-1]]), axis=0) / (len(self.vertices) - 1)   
         dist = 0.0
         for v, s in self.vertices:
            dist += sqrt(((self.centroid - v) ** 2).sum()) / len(self.vertices)
         if dist < self.config.restartTolerance:
            if self.config.printOut:
               print "restarting"
            self.vertices = []
            self.centroid = None
            self.reflectScore = None
            self.shrinkIdx = None
            self.current = FixedCube(self.config).batch(1)[0]
            self.state = NM_REFLECT_INIT   
   
   def reflect(self):
      if self.centroid is not None:
         self.current = self.centroid + self.alpha * (self.centroid - self.vertices[-1][0])
      
   def expand(self):
      if self.centroid is not None:
         self.reflect()
         self.current = self.centroid + self.gamma * (self.current - self.centroid)
    
   def contract(self):
      if self.centroid is not None:
         if self.reflectScore > self.vertices[-1][1]:
            self.reflect()
            self.current = self.centroid + self.beta * (self.current - self.centroid)
         else:
            self.current = self.centroid + self.beta * (self.vertices[-1][0] - self.centroid)
         
   def shrink(self):
      if self.centroid is not None:
         for i in xrange(len(self.vertices) - 1):
            self.vertices[i+1] = [self.vertices[0][0] + self.delta * (self.vertices[i+1][0] - self.vertices[0][0]), None]
         self.shrinkIdx = 1
         self.current = self.vertices[self.shrinkIdx][0]
      
   def update(self, n, pop):
      startState = self.state
      self.current = None
      if len(self.vertices) < self.config.dim + 1:
         self.vertices.append(pop[0])
         if len(self.vertices) < self.config.dim + 1:
            self.current = self.vertices[0][0].copy()
            idx = len(self.vertices) - 1
            self.current[idx] += self.config.scale
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
            if self.reflectScore > self.vertices[-2][1] and self.reflectScore <= self.vertices[0][1]:
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
      if self.config.bounded:
         center = self.config.center
         scale = self.config.scale
         if hasattr(self.config.in_bounds, 'extent'):
            center, scale = self.config.in_bounds.extent()
         self.current = maximum(minimum(self.current, ones(self.config.dim) * (center + scale)), ones(self.config.dim) * (center - scale))
      
         