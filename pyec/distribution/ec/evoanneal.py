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
from pyec.distribution.bayes.structure.proposal import StructureProposal
from pyec.distribution.bayes.sample import DAGSampler
from pyec.distribution.ec.mutators import *
from pyec.distribution.ec.selectors import *
from pyec.config import Config, ConfigBuilder
from pyec.util.combinatorics import factorial

from scipy.special import erf

import logging
log = logging.getLogger(__file__)  

class REAConfigurator(ConfigBuilder):
   """
      A :class:`ConfigBuilder` for Real-Space Evolutionary Annealing.
      
      See source for defaults (Tournament annealing selection, area-sensitive gaussian mutation, learning rate = 1, no crossover).
   """
   def __init__(self):
      super(REAConfigurator, self).__init__(EvolutionaryAnnealing)
      self.cfg.shiftToDb = True
      self.cfg.taylorDepth = 10
      self.cfg.selection = "tournament"
      self.cfg.varInit = 1.0
      self.cfg.varDecay = 2.5
      self.cfg.varExp = 0.25
      self.cfg.learningRate = 1.0
      self.cfg.pressure = 0.025
      self.cfg.crossover = "none"
      self.cfg.mutation = "gauss"
      self.cfg.activeField = "point"
      self.cfg.partition = True
      self.cfg.passArea = True
      
   def setTournament(self, pressure=0.025):
      self.cfg.selection = "tournament"
      self.cfg.taylorDepth = 0
      self.cfg.pressure = pressure
            
   def setProportional(self, taylorDepth=100):
      self.cfg.taylorDepth = taylorDepth
      self.cfg.selection = "proportional"
      
   def setGaussian(self):
      self.cfg.mutation = "gauss"

   def setCauchy(self):
      self.cfg.mutation = "cauchy"
      
   def setCrossover(self, enable=True):
      if enable:
         self.cfg.crossover = "uniform"
      else:
         self.cfg.crossover = "none"

   def postConfigure(self, cfg):
      if cfg.bounded:
         cfg.initialDistribution = FixedCube(cfg)
      else:
         cfg.initialDistribution = SimpleGaussian(cfg)

class BEAConfigurator(REAConfigurator):
   """
      A :class:`ConfigBuilder` for a binary-encoded evolutionary annealing instance.
      
      See souce for defaults (Tournament annealing selection, uniform crossover, 16-bit conversions, area-sensitive bernoulli mutation).
   """
   def __init__(self):
      super(BEAConfigurator, self).__init__()
      self.setTournament()
      self.setCrossover()
      self.cfg.mutation = "bernoulli"
      self.cfg.binaryDepth=16
      self.cfg.activeField="binary"
      
   def postConfigure(self, cfg):
      cfg.rawdim = cfg.dim
      cfg.rawscale = cfg.scale
      cfg.rawcenter = cfg.center
      cfg.dim = cfg.dim * cfg.binaryDepth
      cfg.center = .5
      cfg.scale = .5 
      cfg.initialDistribution = SimpleBernoulli(cfg)

class BayesEAConfigurator(REAConfigurator):
   """
      A :class:`ConfigBuilder` for an evolutionary annealing structure search for a Bayesian network. 
      
      See source for defaults (Tournament annealing, DAGs, area-sensitive structure proposals).
   """
   def __init__(self):
      super(BayesEAConfigurator, self).__init__()
      self.setTournament()
      self.cfg.varInit = 1.0
      self.cfg.varDecay = 1.0
      self.cfg.varExp = 0.25
      self.cfg.bounded = False
      self.cfg.initialDistribution = None
      self.cfg.data = None
      self.cfg.randomizer = None
      self.cfg.sampler = DAGSampler()
      self.cfg.numVariables = None
      self.cfg.variableGenerator = None
      self.cfg.crossoverProb = 0.50
      self.cfg.mutation = "structure"
      self.cfg.crossover = "none"
      self.cfg.activeField = "bayes"
      self.cfg.passArea = True
      self.cfg.learningRate = 0.005
      self.cfg.branchFactor = 1000000

   def setCrossover(self, merge=True, enable=True):
      if enable and merge:
         self.cfg.crossover = "merge"
      elif enable and not merge:
         self.cfg.crossover = "crossbayes"
      else:
         self.cfg.crossover = "none"
         
         
         
      
class EvolutionaryAnnealing(Convolution):
   """
      The evolutionary annealing algorithm as described in
      
      <http://nn.cs.utexas.edu/downloads/papers/lockett.thesis.pdf>
      
      Lockett, Alan. General-Purpose Optimization Through Information Maximization. Ph.D. Thesis (2011).
      
      Config parameters:
      * learningRate -- A multiplier for the annealed cooling schedule
      * selection -- One of (proportional, tournament), the type of annealed selection to use
      * crossover -- One of (none, uniform, onePoint, twoPoint, intermediate, merge, crossbayes), the type of crossover to use.
      * mutation -- One of (gauss, uniform, uniformBinary, cauchy, bernoulli, structure), the type of mutation to use. Must match the space being searched.
      * passArea -- Whether to use area-sensitive mutation distributions.
      * varInit -- Scaling factor for standard deviation in Gaussian spaces. Defaults to 1.0.
      * initialDistribution -- The initial distribution to sample.
      
      
      See :mod:`pyec.distribution.ec.selectors` for selection methods and :mod:`pyec.distribution.ec.mutators` for mutation distributions.
   """
   def __init__(self, config):
      self.config = config
      self.binary = False
      if hasattr(config, 'varInit'):
         config.stddev = config.varInit
      self.selectors = []
      if config.selection == "proportional":
         self.selectors.append(ProportionalAnnealing(config))
      elif config.selection == "tournament":
         self.selectors.append(TournamentAnnealing(config))
      else:
         raise Exception, "Unknown Selection"
      
      if len(self.selectors) == 1:
         self.selector = self.selectors[0]
      else:
         self.selector = Convolution(self.selectors, passScores = True)
      self.mutators = []
      
      if hasattr(config, 'crossover') and config.crossover != "none":
         selector = self.selector
         if config.crossover == 'uniform':
            crosser = UniformCrosser(config)
         elif config.crossover == 'onePoint':
            crosser = OnePointCrosser(config)
         elif config.crossover == 'twoPoint':
            crosser = TwoPointCrosser(config)
         elif config.crossover == 'intermediate':
            crosser = IntermediateCrosser(config)
         elif config.crossover == 'merge':
            crosser = Merger()
         elif config.crossover == 'crossbayes':
            crosser = UniformBayesCrosser(config)
         else:
            raise Exception, "Unknown crossover method"
         order = 2
         if hasattr(config, 'crossoverOrder'):
            order = int(config.crossoverOrder)
         else:
            config.crossoverOrder = order
         self.mutators.append(Crossover(selector, crosser, order))
      
      
      
      if config.mutation == "gauss":
         if config.passArea:
            self.mutators.append(AreaSensitiveGaussian(config))
         else:
            self.mutators.append(DecayedGaussian(config))
      elif config.mutation == "uniform":
         config.passArea = False
         self.mutators.append(UniformArea(config))
      elif config.mutation == "uniformBinary":
         self.binary = not self.config.binaryPartition
         config.passArea = True
         self.mutators.append(UniformAreaBernoulli(config))
      elif config.mutation == "cauchy":
         self.mutators.append(DecayedCauchy(config))
      elif config.mutation == "bernoulli":
         self.binary = not self.config.binaryPartition
         if config.passArea:
            self.mutators.append(AreaSensitiveBernoulli(config))
         else:
            self.mutators.append(DecayedBernoulli(config))
      elif config.mutation == "structure":
         if config.passArea:
            self.mutators.append(AreaSensitiveStructureMutator(config))
         else:
            self.mutators.append(StructureMutator(config))
         config.initialDistribution = StructureProposal(config)
         config.structureGenerator = config.initialDistribution
            
      self.mutator = Convolution(self.mutators)
      passScores = len(self.selectors) > 1
      super(EvolutionaryAnnealing, self).__init__(self.selectors + self.mutators, config.initialDistribution, passScores)
      

   
   def convert(self, x):
      if self.binary:
         ns = array([i+1 for i in xrange(self.config.binaryDepth)] * self.config.rawdim)
         ms = .5 ** ns
         y = reshape(x.__mult__(ms), (self.config.binaryDepth, self.config.rawdim))
         y = y.sum(axis=0).reshape(self.config.rawdim)
         return y * self.config.rawscale + self.config.rawcenter
      elif hasattr(self.config, 'convert') and self.config.convert:
         return x[:self.config.dim]
      return x
   
   @property
   def var(self):
      try:
         return self.mutators[-1].sd   
      except:
         try:
            return self.mutators[-1].bitFlipProbs
         except:
            try:
               return self.mutators[-1].decay
            except:
               return 0.0
                  
   @classmethod
   def configurator(cls):
      return REAConfigurator(cls)
      
   

         

