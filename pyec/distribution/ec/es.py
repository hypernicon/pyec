"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from pyec.distribution.convolution import Convolution
from pyec.distribution import Gaussian as SimpleGaussian
from pyec.distribution import BernoulliTernary as SimpleBernoulli
from pyec.distribution import FixedCube
from pyec.distribution.bayes.mutators import *
from pyec.distribution.bayes.sample import DAGSampler
from pyec.distribution.ec.mutators import *
from pyec.distribution.ec.selectors import *
from pyec.config import Config, ConfigBuilder

import logging
log = logging.getLogger(__file__)  
      
class SimpleExtension(PopulationDistribution):
   def __init__(self, toExtend, extension):
      self.toExtend = toExtend
      self.extension = extension
      self.config = self.toExtend.config
    
   def __call__(self):
      return append(self.toExtend.__call__(), self.extension, axis=0)

   def batch(self, popSize):
      return [self.__call__() for i in xrange(popSize)]

   def update(self, generation, population):
      self.toExtend.update(generation, population)

class ESConfigurator(ConfigBuilder):
   """
      A :class:`ConfigBuilder` for a standard (mu/rho +, lambda)--ES.
      
      By default:
      
      * mu = 10
      * lambda = 50
      * rho = 1 (no crossover)
      * "Plus" style selection is used.
      * If rho > 1, dominant crossover is used.
      * Adaptive mutation is used.
   """

   def __init__(self, *args):
      super(ESConfigurator, self).__init__(EvolutionStrategy)
      self.cfg.crossover = "dominant"
      self.cfg.selection = "plus"
      self.cfg.mu = 10
      self.cfg.lmbda = 50
      self.cfg.rho = 1
      self.cfg.space = "real"
      self.cfg.mutation = "es"
      self.cfg.cmaCumulation = .025
      self.cfg.cmaCorrelation = .025
      self.cfg.cmaDamping = .00005
      
class EvolutionStrategy(Convolution):
   """
      Implements a configurable Evolution Strategy. 
      
      See <http://en.wikipedia.org/wiki/Evolution_strategy> for details and references.
      
      Config parameters:
      
      * mu -- The number of parents to create the next generation
      * rho -- The number of parents for crossover
      * selection -- The type of selection, either "plus" or "comma"
      * crossover -- The type of crossover, either "dominant" or "intermediate"
      * mutation -- Either "cma" or "es"; "es" is default.
      * dim -- The dimension of the binary or real space being optimized
      * space -- Either "real" or "binary"; the type of vector space being optimized
      * bounded -- Whether the search domain is constrained.
      * populationSize -- The size of the population for each generation.
      
      
      The parameter "lambda" is determined by the populationSize and mu, along with the choice of selection (plus or comma).
      
      In real space, initial distribution is either a :class:`FixedCube` if the search is constrained, or a :class:`Gaussian` if not. In binary space, the initial distribution is a random :class:`Bernoulli`. 
      
      Standard mutation ("es") adapts the mutation parameters for each
      member of the population. If mutation is "cma", then the algorithm of 
      Hansen and Ostermeier (1996) is used. Note that the 1996 algorithm 
      for CMA differs from the modern version (2001) and maintains
      separate mutation parameters for each solution.
      
      Adaptive parameters are not implemented for binary spaces.
      
      Extra parameters for CMA:
      
      * cmaCumulation (.025)
      * cmaCorrelation (.025)
      * cmaDamping (.00005)
   
      See :mod:`pyec.distribution.ec.selectors` for selection methods and :mod:`pyec.distribution.ec.mutators` for mutation distributions.
   
   """
   
   unsorted = False
   def __init__(self, config):
      """ 
         Config options:
             mu - number of parents
             rho - number of parents for crossover
             selection - (plus, comma)
             crossover - (dominant, intermediate)

      """
      self.config = config
      self.selectors = []
      self.selectors.append(EvolutionStrategySelection(config))
      self.selector = Convolution(self.selectors)
      self.mutators = []
      if config.rho > 1:
         if config.crossover == 'dominant':
            crosser = DominantCrosser(config)
         elif config.crossover == 'intermediate':
            crosser = IntermediateCrosser(config)
         else:
            raise Exception, "Unknown crossover method"
         order = config.rho
         self.mutators.append(Crossover(self.selector, crosser, order))
      if config.space == 'real':
         if hasattr(config, 'mutation') and config.mutation == 'cma':
            self.mutators.append(CorrelatedEndogeneousGaussian(config))
            if config.bounded:
               initial = SimpleExtension(FixedCube(config), self.buildRotation(config))
            else:
               initial = SimpleExtension(SimpleGaussian(config), self.buildRotation(config))
         else:
            self.mutators.append(EndogeneousGaussian(config))
            if config.bounded:
               initial = SimpleExtension(FixedCube(config), ones(config.dim))
            else:
               initial = SimpleExtension(SimpleGaussian(config), ones(config.dim))
      elif config.space == 'binary':
         bitFlip = 0.05
         if hasattr(config, 'bitFlipProbs'):
            bitFlip = config.bitFlipProbs
         self.mutators.append(Bernoulli(bitFlip))
         initial = SimpleBernoulli(config)
      else:
         raise Exception, "Unknown space"
      self.mutator = Convolution(self.mutators)
      
      super(EvolutionStrategy, self).__init__([self.selector, self.mutator], initial)
      
   def convert(self, x):
      return x[:self.config.dim]
         
   def buildRotation(self, config):
      ret = []
      for i in xrange(config.dim):
         for j in xrange(config.dim):
            if i == j:
               ret.append(config.varInit)
            elif j > i:
               ret.append(0.0)
      ret = append(array(ret), zeros(config.dim), axis=0)
      ret = append(array(ret), ones(config.dim), axis=0)
      ret = append(array(ret), zeros(config.dim), axis=0)
      return ret
      
   @classmethod
   def configurator(cls):
      return ESConfigurator(cls)

