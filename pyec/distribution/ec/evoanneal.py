"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import numpy as np

from pyec.config import Config
from pyec.distribution.convolution import Convolution
from pyec.distribution.bayes.mutators import *
from pyec.distribution.bayes.structure.proposal import StructureProposal
from pyec.distribution.ec.selectors import Selection
from pyec.distribution.ec.mutators import (Mutation,
                                           Bernoulli,
                                           Crossover,
                                           Gaussian,
                                           AreaSensitiveGaussian,
                                           AreaSensitiveBernoulli)
from pyec.history import History
from pyec.space import Euclidean, Binary
from pyec.util.combinatorics import factorial
from pyec.util.partitions import (Segment,
                                  Partition,
                                  ScoreTree,
                                  AreaTree,
                                  Point,
                                  SeparationAlgorithm,
                                  VectorSeparationAlgorithm,
                                  LongestSideVectorSeparationAlgorithm,
                                  BinarySeparationAlgorithm,
                                  BayesSeparationAlgorithm)
from pyec.util.RunStats import RunStats

from scipy.special import erf

import logging
log = logging.getLogger(__file__)  

class PartitionHistory(History):
   """A :class:`History` that records all points and uses those points
   to partition the search domain usina a :class:`Partition` object. Also
   stores a :class:`ScoreTree` and a :class:`AreaTree`.
   
   This class is memory intensive, so be careful about convolving or
   convexly combining it, since each of those combinations may cause
   multiple copies of this history to be stored in memory.

   """
   def __init__(self, config):
      super(PartitionHistory, self).__init__(config)
      self.segment = Segment(config=self.config)
      self.config.stats = RunStats()
      self.stats = self.config.stats
      self.stats.recording = self.config.record
      self.separator = self.config.separator(config)
      self.attrs |= set(["segment", "stats"])
   
   def temperature(self):
      n = self.updates / self.config.divisor
      if hasattr(self.config.schedule, '__call__'):
         return self.config.schedule(n)
      elif self.config.schedule == "linear":
         return 1. / (n * self.config.learningRate)
      elif self.config.schedule == "log":
         return 1. / (np.log(n) * self.config.learningRate)
      elif self.config.schedule == "discount":
         return 1. / (self.config.temp0 * (self.config.discount ** n))
      elif self.config.schedule == "log_area":
         return 1./-(np.log(self.segment.partitionTree.largestArea())
                     * self.config.learningRate)
      else:
         return 1.0
      
   def internalUpdate(self, population):
      """Insert all new points into the partition.
      
      """
      if population[0][1] is None:
         # skip unscored updates inside convolution
         return
      
      if not isinstance(population[0][0], self.config.space.type):
         # again, skip intermediate
         return
      
      pts = [Point(self.segment, x, None, s)
             for x,s in population]
      self.config.stats.start("save")
      Point.bulkSave(self.segment, pts, self.stats)
      self.config.stats.stop("save")


class TaylorPartitionHistory(PartitionHistory):
   def __init__(self, config):
      super(TaylorPartitionHistory, self).__init__(config)

   def internalUpdate(self, population):
      if population[0][1] is None:
         # skip unscored updates inside convolution
         return
      
      super(TaylorPartitionHistory, self).internalUpdate(population)
      bottom = .5 * floor(2*(1./self.temperature())/self.config.learningRate)
      if  bottom > 1./self.segment.taylorCenter:
         self.segment.scoreTree.resetTaylor(self.segment, 1./bottom, self.config)


class Annealing(Selection):
   """
      Base class for annealed selection. See e.g.
      
      Lockett, A. and Miikkulainen, R. Real-space Evolutionary Annealing, 2011.
      
      For a more up-to-date version, see Chapters 11-13 of 
      
      <http://nn.cs.utexas.edu/downloads/papers/lockett.thesis.pdf>.
      
      Config parameters (many like simulated annealing)
      
      * schedule -- The cooling schedule to use. May be any callable function, or
                 a string, one of ("log", "linear", "discount"). If it is a
                 callable, then it will be passed the ``updates`` value in
                 :class:`History` and should return a floating point value
                 that will divide the exponent in the Boltzmann distribution.
                 That is, ``schedule(n)`` should converge to zero as n goes to
                 infinity.
      * learningRate -- A divisor for the cooling schedule, used for built-in 
                     schedules "log" and "inear". As a divisor, it divides the
                     temperature but multiplies the exponent.
      * temp0 -- An initial temperature for the temperature decay in "discount"
      * divisor -- A divisor that divides the ``updates`` property of
                :class:`History`, scaling the rate of decline in the temperature
      * discount -- The decay factor for the built-in discount schedule; ``temp0``
                 is multiplied by ``discount`` once each time the ``update``
                 method is called
      * separator -- A :class:`SeparationAlgorithm` for partitioning regions
      
   """
   config = Config(schedule="log_area",
                   learningRate = 1.0,
                   divisor = 1.0,
                   temp0 = 1.0,
                   discount = .99,
                   separator = SeparationAlgorithm,
                   history = PartitionHistory)
   
   def __init__(self, **kwargs):
      super(Annealing, self).__init__(**kwargs)
      if self.config.jogo2012:
         self.config.schedule = "log"
         self.config.separator = VectorSeparationAlgorithm
   
   def compatible(self, history):
      return isinstance(history, PartitionHistory)
   
   def sample(self):
      """
         Child classes should override this method in order to select a point
         from the active segment in :class:`pyec.util.partitions.Point`.
         
         The actual return should be a partition node.
         
      """
      pass
      
   def batch(self, m):
      return [self.sample() for i in xrange(m)]


class ProportionalAnnealing(Annealing):
   """
      Proportional Annealing, as described in 
      
      Lockett, A. And Miikkulainen, R. Real-space Evolutionary Annealing (2011).
      
      See :class:`Annealing` for more details.
      
      Config parameters:
      
      * taylorCenter -- Center for Taylor approximation to annealing densities.
      * taylorDepth -- The number of coefficients to use for Taylor approximation of the annealing density.
      
   """
   config = Config(taylorCenter = 1.0,
                   taylorDepth = 10,
                   history = TaylorPartitionHistory)
   
   def compatible(self, history):
      return isinstance(history, TaylorPartitionHistory)
         
   def sample(self):
      return Point.sampleProportional(self.history.segment,
                                      self.history.temperature(),
                                      self.config)
  
  
class TournamentAnnealing(Annealing):
   """
      Tournament Annealing, as described in Chapter 11 of
      
      <http://nn.cs.utexas.edu/downloads/papers/lockett.thesis.pdf>
      
      See :class:`Annealing` for more details.
      
   """
   config = Config(pressure=0.025)
   
   def sample(self):
      return Point.sampleTournament(self.history.segment,
                                    self.history.temperature(),
                                    self.config)


class AreaStripper(Mutation):
   """:class:`Annealing` and its descendents pass along a tuple; this class
   extracts the point for use with other methods.
   
   """
   def mutate(self, x):
      return x[0].point

AnnealingCrossover = (
   ((TournamentAnnealing << AreaStripper) <<
    ((TournamentAnnealing >> 1) << AreaStripper) <<
    Crossover)
)

RealEvolutionaryAnnealing = (
   TournamentAnnealing << AreaSensitiveGaussian
)[Config(separator=LongestSideVectorSeparationAlgorithm)]

RealEvolutionaryAnnealingJogo2012 = RealEvolutionaryAnnealing[Config(
   jogo2012=True,
   schedule="log",
   separator=VectorSeparationAlgorithm
)]

CrossedRealEvolutionaryAnnealing = (
   AnnealingCrossover << Gaussian
)[Config(separator=LongestSideVectorSeparationAlgorithm)]

BinaryEvolutionaryAnnealing = (
   TournamentAnnealing << AreaStripper << Bernoulli
)[Config(separator=BinarySeparationAlgorithm)]

CrossedBinaryEvolutionaryAnnealing = (
   AnnealingCrossover << Bernoulli
)[Config(separator=BinarySeparationAlgorithm)]

BayesEvolutionaryAnnealing = (
   AnnealingCrossover << StructureProposal
)[Config(separator=BayesSeparationAlgorithm,
         crosser=Merger, #UniformBayesCrosser,
         schedule="log",
         learningRate=0.1,
         crossoverProb=.25)]

