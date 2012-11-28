"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import inspect
import numpy as np

from basic import PopulationDistribution
from pyec.config import Config
from pyec.history import LocalMinimumHistory
from pyec.space import Euclidean

class PSOHistory(LocalMinimumHistory):
   """A :class:`History` for Particle Swarm Optimization.
   Rembers the local best and the velocities.
   
   """
   def __init__(self, config):
      super(PSOHistory, self).__init__(config)
      self._positions = None
      self._velocities = None
      self.lowerv = None
      self.upperv = None
      self.attrs |= set(["_velocities", "_positions", "upperv", "lowerv"])
      
   def velocities(self):
      return self._velocities
   
   def positions(self):
      return self._positions

   def updateVelocity(self):
      popSize = self.config.populationSize
      if self._velocities is None:
         if self.config.initial is None:
            self._velocities = np.array([self.config.space.random()
                                         for i in xrange(popSize)])
         elif (inspect.isclass(self.config.initial) and
               isinstance(self.config.initial, PopulationDistribution)):
            self._velocities = np.array([self.config.initial.batch(popSize)])
         else:
            self._velocities = np.array([self.config.initial()
                                         for i in xrange(popSize)])
         return
      
      rp = np.outer(np.random.random_sample(popSize),
                    np.ones(self.config.dim))
      rg = np.outer(np.random.random_sample(popSize),
                    np.ones(self.config.dim))
      
      #print shape(rp), shape(self.bestLocal), shape(self.bestGlobal), shape(self.positions), shape(self.velocities)
      bestLocal = np.array([x for x,s in self.localBestPop])
      bestGlobal = self.minSolution
      velocities = (self.config.omega * self._velocities 
                    + self.config.phip * rp * (bestLocal - self._positions) 
                    + self.config.phig * rg * (bestGlobal - self._positions))   
      del self._velocities
      self._velocities = np.maximum(self.lowerv,
                                    np.minimum(self.upperv, velocities))
      del rp
      del rg
      
   def internalUpdate(self, population):
      super(PSOHistory, self).internalUpdate(population)
      initialize = False
      if self._positions is not None:
         del self._positions
         initialize = True
      
      self._positions = np.array([x for x,s in population])
      
      if self.config.space.constraint is not None:
         center, scale = self.config.space.constraint.extent()
         self._positions = maximum(self._positions, center-scale)
         self._positions = minimum(self._positions, center+scale)
         
      if initialize:
         self.upperv = self._positions.max(axis=0)
         self.lowerv = self._positions.min(axis=0)
      
      self.updateVelocity() 


class ParticleSwarmOptimization(PopulationDistribution):
   """Particle Swarm Optimization.
       
      Config parameters
      
      * omega -- The decay factor for velocities
      * phig -- The global best component in velocity update
      * phip -- The local best component in velocity update
   
   """
   config = Config(history=PSOHistory,
                   omega=-.5,
                   phig=2.0,
                   phip=2.0)
   
   def __init__(self, **kwargs):
      super(ParticleSwarmOptimization, self).__init__(**kwargs)
      if self.config.space.type != np.ndarray:
         raise ValueError("Space must have type numpy.ndarray")
   
   def compatible(self, history):
      return isinstance(history, PSOHistory)
   
   def batch(self, popSize):
      positions = self.history.positions() + self.history.velocities()
      
      return positions
