"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from numpy import *
from pyec.distribution.basic import PopulationDistribution, FixedCube, Gaussian
from pyec.config import Config, ConfigBuilder

   

class DifferentialEvolution(PopulationDistribution):
   """
      Implements Differential Evolution (DE) as described by:
      
      Storn, Ranier and Price, Kenneth. Differential Evolution -- A simple and efficient adaptive scheme for global optimization over continuous spaces. 1995. 
      
      See <http://en.wikipedia.org/wiki/Differential_evolution> for algorithm details.    
      
      See <http://www.hvass-labs.org/people/magnus/publications/pedersen10good-de.pdf> for a good discussion of parameter settings for DE.

      Config parameters:
      
      * CR -- crossover probability 
      * F -- the learning rate
      * initialDistribution -- the initial distribution to generate the first population
      * populationSize -- the population size
      * center -- the center of the space
      * scale -- the scale of the space
      * dim -- the number of real dimensions in the space
      
      :param cfg: The configuration object for Differential Evolution.
      :type cfg: :class:`Config`

   """


   unsorted = True

   def __init__(self, cfg):
      super(DifferentialEvolution, self).__init__(cfg)
      self.CR = self.config.crossoverProb
      self.F = self.config.learningRate
      self.initial = self.config.initialDistribution
      initial = hasattr(self.initial, 'batch') and self.initial.batch(self.config.populationSize) or [self.initial() for i in xrange(self.config.populationSize)]
      self.xs = zip(initial, [None for i in xrange(self.config.populationSize)])
      

   @classmethod
   def configurator(cls):
      return DEConfigurator(cls)
      
   def batch(self, popSize):
      idx = 0
      ys = []
      for x,s in self.xs:
         y = ones(self.config.dim)
         if True: 
            i1 = idx
            while i1 == idx:
               i1 = random.randint(0,self.config.populationSize)
            i2 = i1
            while i1 == i2 or i2 == idx:
               i2 = random.randint(0,self.config.populationSize)
            i3 = i2
            while i1 == i3 or i2 == i3 or i3 == idx:
               i3 = random.randint(0,self.config.populationSize)
         
            a, s1 = self.xs[i1]
            b, s2 = self.xs[i2]
            c, s3 = self.xs[i3]
         
            d = random.randint(0, len(x))
            y = copy(x)
            idx2 = 0
            for yi in y:
               r = random.random_sample()
               if idx2 == d or r < self.CR:
                  y[idx2] = a[idx2] + self.F * (b[idx2] - c[idx2]) 
               idx2 += 1 
         ys.append(y)
         idx += 1
      return ys
      

   def update(self, generation, population):
      idx = 0
      # print [s for x,s in self.xs][:5]
      for y,s2 in population:
         x,s = self.xs[idx]
         if s is None or s2 >= s:
            self.xs[idx] = y,s2 
         idx += 1


class DEConfigurator(ConfigBuilder):
   """
      A :class:`ConfigBuilder` object to create a :class:`DifferentialEvolution` instance.
      
      Default parameters are CR = .2, F = .5; these can be changed
      by editing the `cfg` property of this object.
   """

   def __init__(self, *args):
      super(DEConfigurator, self).__init__(DifferentialEvolution)
      self.cfg.sort = False
      self.cfg.crossoverProb = .2
      self.cfg.learningRate = .5
   
   def postConfigure(self, cfg):
      if cfg.varInit is None:
         cfg.initialDistribution = FixedCube(cfg)
      else:
         cfg.usePrior = False
         cfg.initialDistribution = Gaussian(cfg)         
