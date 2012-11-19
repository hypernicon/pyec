"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from pyec.config import Config, ConfigBuilder
from pyec.distribution.convolution import Convolution
from pyec.distribution import Gaussian as SimpleGaussian
from pyec.distribution import BernoulliTernary as SimpleBernoulli
from pyec.distribution import FixedCube
from pyec.distribution.ec.mutators import *
from pyec.distribution.ec.selectors import *

import logging
log = logging.getLogger(__file__)

class RGAConfigurator(ConfigBuilder):
   """
      A :class:`ConfigBuilder` for a real-coded Genetic Algorithm.
      
      See the source code for defaults (uniform crossover, ranking selection, gaussian mutation).
   """
   def __init__(self, *args):
      alg = GeneticAlgorithm
      if len(args) > 0:
         alg = args[0]
      super(RGAConfigurator, self).__init__(alg)
      self.cfg.elitist = False
      self.cfg.selection = "ranking"
      self.cfg.rankingMethod = "linear"
      self.cfg.selectionPressure = 1.8
      self.cfg.crossover = "uniform"
      self.cfg.crossoverOrder = 2
      self.cfg.space = "real"
      self.cfg.mutation = "gauss"


class GAConfigurator(RGAConfigurator):
   """
      A :class:`ConfigBuilder` for a standard genetic algorithm (binary encoding) to search in real spaces.
      
      See source code for defaults (16-bit representation, mutation rate .05).
   """
   def __init__(self, *args):
      super(GAConfigurator, self).__init__(*args)
      self.cfg.space = "binary"
      self.cfg.activeField = "binary"
      self.cfg.binaryDepth = 16

   def postConfigure(self, cfg):
      cfg.rawdim = cfg.dim
      cfg.rawscale = cfg.scale
      cfg.rawcenter = cfg.center
      cfg.dim = cfg.dim * cfg.binaryDepth
      cfg.center = .5
      cfg.scale = .5 
      cfg.bitFlipProbs = .05


class GeneticAlgorithm(Convolution):
   """
      A genetic algorithm for optimization of real functions on Euclidean spaces.
      
      Support various selection, crossover, and mutation methods.
      Supports binary and real encodings.
      
      Config parameters
      * elitist = One of (True, False), whether to keep the best solution so far in the population.
      * selection = One of (proportional, tournament, ranking), different selection methods.
      * rankingMethod = One of (linear, nonlinear), when using ranking selection.
      * crossover = One of (none, uniform, onePoint, twoPoint, intermediate, onePointDual), the type of crossover to use.
      * crossoverOrder = Integer, the number of parents for recombination (default 2).
      * space = One of (real, binary); the type of encoding.
      * mutation = One of (gauss, cauchyOne), the type of mutation in real encodings (binary uses a standard Bernoulli mutation).
      * varInit = float or float array, standard deviation (for real space).
      * bitFlipProbs = float or float array, mutation probability (for binary space).
      
      Binary encoding uses other parameters to encode/decode the parameters
      * rawdim -- The number of real dimensions.
      * rawscale -- The scale of the space in real dimensions.
      * rawcenter -- The center of the space in real dimensions.
      * binaryDepth -- The number of bits to use for each real parameter.  
      
      Generic config parameters
      * dim -- The dimension of the search domain (for binary, the total number of bits)
      * center -- The center of the search domain (.5 for binary)
      * scale -- The scale of the search domain (.5 for binary)
      * bounded -- Whether the optimization is constrained. 
      
      See :mod:`pyec.distribution.ec.selectors` for selection methods and :mod:`pyec.distribution.ec.mutators` for mutation distributions.
      
      :params config: The configuration parameters.
      :type config: :class:`Config`
   """


   def __init__(self, config):
      """ 
         Config options:
             elitist = (True, False)
             selection = (proportional, tournament, ranking)
             rankingMethod = (linear, nonlinear)
             crossover = (none, uniform, onePoint, twoPoint, intermediate, onePointDual)
             crossoverOrder = integer
             space = (real, binary)
             mutation = (gauss, cauchyOne)
             varInit = float or float array, standard deviation (for real space)
             bitFlipProbs = float or float array, mutation probability (for binary space)

      """
      self.config = config
      self.selectors = []
      if hasattr(config, 'elitist') and config.elitist:
         self.selectors.append(Elitist())
      if hasattr(config, 'selection'):
         if config.selection == 'proportional':
            self.selectors.append(Proportional(config))
         elif config.selection == 'tournament':
            self.selectors.append(Tournament(config))
         elif config.selection == 'esp':
            self.selectors.append(ESPSelectionPrimary(config))
         elif config.selection == 'ranking':
            if hasattr(config, 'rankingMethod') and config.rankingMethod == 'nonlinear':
               config.ranker = NonlinearRanker(config.selectionPressure, config.populationSize)
            else:
               config.ranker = LinearRanker(config.selectionPressure)
            self.selectors.append(Ranking(config))
         else:
            raise Exception, "Unknown selection method"
      else:
         #config.ranker = LinearRanker(1.8)
         #self.selectors.append(Ranking(config))
         self.selectors.append(Proportional(config))
      if len(self.selectors) == 1:
         self.selector = self.selectors[0]
      else:
         self.selector = Convolution(self.selectors, passScores=True)
      self.mutators = []
      if not hasattr(config, 'crossover'):
         config.crossover = "uniform"
      if config.crossover != "none":
         if config.crossover == 'uniform':
            crosser = UniformCrosser(config)
         elif config.crossover == 'onePoint':
            crosser = OnePointCrosser(config)
         elif config.crossover == 'onePointDual':
            crosser = OnePointDualCrosser(config)
         elif config.crossover == 'twoPoint':
            crosser = TwoPointCrosser(config)
         elif config.crossover == 'intermediate':
            crosser = IntermediateCrosser(config)
         else:
            raise Exception, "Unknown crossover method"
         order = 2
         if hasattr(config, 'crossoverOrder'):
            order = int(config.crossoverOrder)
         if config.selection == "esp":
            secondary = ESPSelectionSecondary(config)
            self.selectors[0].secondary = secondary
         else:
            secondary = self.selector
         self.mutators.append(Crossover(secondary, crosser, order))
      if config.space == 'real':
         variance = .05
         if hasattr(config, 'varInit') and config.varInit is not None:
            variance = config.varInit
         config.stddev = variance
         if config.mutation == "cauchyOne":
            self.mutators.append(OnePointCauchy(config))
         else:
            self.mutators.append(Gaussian(config))
         if config.bounded:
            initial = FixedCube(config)
         else:
            initial = SimpleGaussian(config)
      elif config.space == 'binary':
         self.mutators.append(Bernoulli(config))
         initial = SimpleBernoulli(config)
      else:
         raise Exception, "Unknown space"
      self.mutator = Convolution(self.mutators)
      
      passScores = len(self.selectors) > 1
      super(GeneticAlgorithm, self).__init__(self.selectors + self.mutators, initial, passScores=passScores)
   
   def convert(self, x):
      if self.config.space == "binary":
         ns = array([i+1 for i in xrange(self.config.binaryDepth)] * self.config.rawdim)
         ms = .5 ** ns
         y = reshape(x * ms, (self.config.binaryDepth, self.config.rawdim))
         y = y.sum(axis=0).reshape(self.config.rawdim)
         return y * self.config.rawscale + self.config.rawcenter
      return x
         
   @classmethod
   def configurator(cls):
      return GAConfigurator(cls)      
