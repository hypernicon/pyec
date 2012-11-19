"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from numpy import *
from basic import PopulationDistribution, FixedCube, Gaussian
from pyec.config import Config, ConfigBuilder

class PSOConfigurator(ConfigBuilder):
   registryKeys = ('omega', 'phig', 'phip', 'center', 'scale')
   registry = {
      'sphere': (-.5, 2.0, 2.0, 0, 5.12),
      'ellipsoid': (-.5, 2.0, 2.0, 0, 5.12),
      'rotatedEllipsoid': (-.5, 2.0, 2.0, 0, 5.12),
      'rosenbrock': (-.5, 2.0, 2.0, 0, 5.12),
      'rastrigin': (-.5, 2.0, 2.0, 0, 5.12),
      'miscaledRastrigin': (-.5, 2.0, 2.0, 0, 5.12),
      'schwefel': (-.5, 2.0, 2.0, 0, 512),
      'salomon': (-.5, 2.0, 2.0, 0, 30),
      'whitley': (-.5, 2.0, 2.0, 0, 5.12),
      'ackley': (-.5, 2.0, 2.0, 0, 30),
      'langerman': (-.5, 2.0, 2.0, 0, 15),
      'shekelsFoxholes': (-.5, 2.0, 2.0, 0, 15),
      'shekel2': (-.5, 2.0, 2.0, 5, 10),
      'rana': (-.5, 2.0, 2.0, 0, 520),
      'griewank': (-.5, 2.0, 2.0, 0, 600)
   }

   def __init__(self, *args):
      super(PSOConfigurator, self).__init__(ParticleSwarmOptimization)
      self.cfg.sort = False
      self.cfg.omega = -.5
      self.cfg.phig = 2.0
      self.cfg.phip = 2.0
      
   def postConfigure(self, cfg):   
      if cfg.bounded:
         cfg.initialDistribution = FixedCube(cfg)
      else:
         cfg.initialDistribution = Gaussian(cfg)



class ParticleSwarmOptimization(PopulationDistribution):
   unsorted = True
   
   def __init__(self, cfg):
      super(ParticleSwarmOptimization, self).__init__(cfg)
      self.initial = cfg.initialDistribution
      if hasattr(self.initial, 'batch'):
         self.positions = array(self.initial.batch(cfg.populationSize))
         self.velocities = array(self.initial.batch(cfg.populationSize))
      else:
         self.positions = array([self.initial() for i in xrange(cfg.populationSize)])
         self.velocities = array([self.initial() for i in xrange(cfg.populationSize)])
      #print self.positions, self.velocities
      self.upperv = self.positions.max(axis=0)
      self.lowerv = self.positions.min(axis=0)
      self.bestLocal = self.positions.copy()
      self.bestLocalScore = zeros(cfg.populationSize)
      self.omega = cfg.omega
      self.phig = cfg.phig
      self.phip = cfg.phip
      self.bestGlobal = self.bestLocal
      self.bestGlobalScore = None
   
   @classmethod
   def configurator(cls):
      return PSOConfigurator(cls)      
      
   def updateVelocity(self):
      rp = outer(random.random_sample(self.config.populationSize), ones(self.config.dim))
      rg = outer(random.random_sample(self.config.populationSize), ones(self.config.dim))
      
      #print shape(rp), shape(self.bestLocal), shape(self.bestGlobal), shape(self.positions), shape(self.velocities)
      velocities = self.omega * self.velocities \
       + self.phip * rp * (self.bestLocal - self.positions) \
       + self.phig * rg * (self.bestGlobal - self.positions)   
      del self.velocities
      self.velocities = maximum(self.lowerv, minimum(self.upperv, velocities))
      del rp
      del rg
      
      
   def batch(self, popSize):
      # print sqrt((velocities ** 2).sum())
      positions = self.positions + self.velocities
      if self.config.bounded:
         center = self.config.center
         scale = self.config.scale
         if self.config.bounded and hasattr(self.config.in_bounds, 'extent'):
            center, scale = self.config.in_bounds.extent()
         positions = maximum(positions, center-scale)
         positions = minimum(positions, center+scale)
      
      return positions

   def update(self, generation, population):
      del self.positions
      self.positions = array([x for x,s in population])

      idx = 0
      maxScore = 0
      maxOrg = None
      for x,s in population:
         if maxOrg is None or s >= maxScore:
            maxOrg = x
            maxScore = s
         if self.bestGlobalScore is None or s >= self.bestLocalScore[idx]:
            self.bestLocalScore[idx] = s
            self.bestLocal[idx] = x
         idx += 1
      if self.bestGlobalScore is None or maxScore >= self.bestGlobalScore:
         self.bestGlobalScore = maxScore
         self.bestGlobal = maxOrg
         
      self.updateVelocity()
      #print self.positions   
