"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from numpy import *
import copy, traceback, sys
from pyec.distribution.basic import PopulationDistribution
from pyec.util.partitions import Point, Partition, ScoreTree, Segment
from pyec.util.TernaryString import TernaryString

import logging
logger = logging.getLogger(__file__)

class Selection(PopulationDistribution):
   """A selection method (Abstract class)"""
   pass

class BestSelection(Selection):
   """
      Select the best member of the population.
      
      :param config: The configuration parameters.
      :type config: :class:`Config`
   """
   def __init__(self, config):
      super(Selection, self).__init__(config)
      self.best = None
      self.score = None
      
   def __call__(self):
      return self.best

   def update(self, generation, population):
      for x, s in population:
         if s >= self.score:
            self.best = x
            self.score = s

class EvolutionStrategySelection(Selection):
   """
      Implements standard selection for evolution strategies.
      
      The property ``config.selection`` determines the type of selection, either "plus" or "comma".
      
      :param config: The configuration parameters.
      :type config: :class:`Config`
   """
   def __init__(self, config):
      super(EvolutionStrategySelection, self).__init__(config)
      self.total = 0
      self.mu = config.mu
      self.plus = config.selection == 'plus' # plus or comma selection, for ES

   def __call__(self):
      idx = random.randint(0,self.mu-1)
      return self.population[idx]

   def batch(self, popSize):
      if self.plus:
         return self.population \
          + [self.__call__() for i in xrange(popSize - self.mu)]
      else:
         return [self.__call__() for i in xrange(popSize)]

   def update(self, generation, population):
      """the population is ordered by score, take the first mu as parents"""
      self.population = [x for x,s in population][:self.mu]
      

class Proportional(Selection):
   """
      Fitness proportional selection (roulette wheel selection).
      
      See <http://en.wikipedia.org/wiki/Fitness_proportionate_selection>.
      
      Fitness values must be nonnegative.
      
      :param config: The configuration parameters.
      :type config: :class:`Config`
   """
   def __init__(self, config):
      super(Proportional, self).__init__(config)
      self.total = 0
      self.matchedPopulation = []

   def __call__(self):
      rnd = random.random_sample() * self.total
      for x,amt in self.matchedPopulation:
         if amt >= rnd:
            return x
      return self.matchedPopulation[-1][0]
      

   def batch(self, popSize):
      return [self.__call__() for i in xrange(popSize)]

   def update(self, generation, population):
      self.population = copy.deepcopy(population)
      self.total = sum([s for x,s in population])
      self.matchedPopulation = []
      amt = 0
      for x,s in population:
         amt += s
         self.matchedPopulation.append((x,amt))
      return self.population

   
class Tournament(Selection):
   """
      Tournament selection on the entire population.
      
      See <http://en.wikipedia.org/wiki/Tournament_selection>.
      
      Config parameters:
      * selectionPressure -- The probability of choosing the best member of the population.
      
      :param config: The configuration parameters.
      :type config: :class:`Config`
   """
   def __init__(self, config):
      super(Tournament, self).__init__(config)
      self.pressure = config.selectionPressure
      self.total = 0
      self.matchedPopulation = []


   def __call__(self):
      rnd = random.random_sample() * self.total
      for x,amt in self.matchedPopulation:
         if amt >= rnd:
            return x
      return self.matchedPopulation[-1][0]
    

   def batch(self, popSize):
      return [self.__call__() for i in xrange(popSize)]

   def update(self, generation, population):
      self.population = copy.deepcopy(population)
      self.matchedPopulation = []
      amt = self.pressure
      self.total = 0
      for x,s in population:
         self.matchedPopulation.append((x,amt))
         self.total += amt
         amt *= (1 - self.pressure)
      return self.population

class Ranker(object):
   """
      A policy to allocate selection probabilities under ranking selection.
   """
   def __call__(self, rank, popSize):
      """
         Determine the probability of choosing the individual with a given rank inside a population of a given size.
         
         :param rank: The rank of the solution under consideration.
         :type rank: int
         :param popSize: The size of the population.
         :type popSize: int
      """
      pass

class LinearRanker(Ranker):
   """
      Computes the probability of selecting a solution from a population using linear ranking selection.
      
      See e.g. <http://www.pohlheim.com/Papers/mpga_gal95/gal2_3.html>
   """
   def __init__(self, pressure):
      """pressure between 1.0 and 2.0"""
      self.pressure = pressure

   def __call__(self, rank, popSize):
      return 2 - self.pressure + (2 * (self.pressure-1))* ((rank-1.0)/(popSize - 1.0)) 
      
class NonlinearRanker(Ranker):
   """
      Computes the probability of selecting a solution from a population using nonlinear ranking selection.
   
      See e.g. <http://www.pohlheim.com/Papers/mpga_gal95/gal2_3.html>
   """

   def __init__(self, pressure, popSize):
      self.pressure = pressure
      self.coeffs = [self.pressure for i in xrange(popSize)]
      self.coeffs[0] -= popSize
      self.root = roots(self.coeffs)[0].real

   def __call__(self, rank, popSize):
      """ root is root of (pressure * sum_k=0^(popSize-1) x^k) - popSize * x ^(popSize - 1)"""
      return self.root ** (rank - 1.0) 


class Ranking(Selection):
   """
      Ranking selection, e.g. <http://www.pohlheim.com/Papers/mpga_gal95/gal2_3.html>
   
      Takes a ranking function which weights individuals according to rank
      Rank is 1 for lowest, K for highest in population of size K
      
      Config parameters
      * ranker -- A :class:`Ranker` instance for the ranking policy.
      
      :param config: The configuration parameters.
      :type config: :class:`Config`
   """
   def __init__(self, config):
      super(Ranking, self).__init__(config)
      self.ranker = config.ranker
      self.total = 0
      self.matchedPopulation = []

   def __call__(self):
      rnd = random.random_sample() * self.total
      for x,amt in self.matchedPopulation:
         if amt >= rnd:
            return x
      return self.matchedPopulation[-1][0]
   
   def density(self, idx):
      return self.ranker(idx, len(self.matchedPopulation)) / self.total

   def batch(self, popSize):
      return [self.__call__() for i in xrange(popSize)]

   def update(self, generation, population):
      self.population = population # copy.deepcopy(population)
      self.matchedPopulation = []
      amt = 0
      idx = len(population)
      for x,s in population:
         amt += self.ranker(idx, len(population))
         self.matchedPopulation.append((x,amt))
         idx -= 1
      self.total = amt
      return self.population

class Elitist(Selection):
   """
      Implements elitism by replacing the worst member of the population
      with the best solution seen so far. If the best solution is a member
      of the current population, nothing is changed.
      
      Elitism can be applied to an algorithm by convolving elitism with the 
      algorithm.
      
      :param config: The configuration parameters.
      :type config: :class:`Config`
   """
   def __init__(self, config):
      super(Elitist, self).__init__(config)
      self.maxScore = -1e100
      self.maxOrg = None
      self.population = None

   def batch(self, popSize):
      return self.population

   def update(self, generation, population):
      if population[0][1] > self.maxScore or self.maxOrg is None:
         self.maxScore = population[0][1]
         self.maxOrg = population[0][0]
         self.population = copy.deepcopy(population)
      else:
         self.population = [(self.maxOrg, self.maxScore)]
         self.population.extend(population)
         self.population = self.population[:-1]


class Annealing(Selection):
   """
      Base class for annealed selection. See e.g.
      
      Lockett, A. and Miikkulainen, R. Real-space Evolutionary Annealing, 2011.
      
      For a more up-to-date version, see Chapters 11-13 of 
      
      <http://nn.cs.utexas.edu/downloads/papers/lockett.thesis.pdf>.
      
      Config parameters
      * taylorCenter -- Center for Taylor approximation to annealing densities.
      * taylorDepth -- The number of coefficients to use for Taylor approximation of the annealing density.
      * activeField -- One of (point, binary, bayes, other); refers to the properties of :class:`pyec.util.partitions.Point` and differs according to the space being searched. Future versions will deprecate this aspect.  
      * initialDistribution -- A distribution used to select central points in the first generation.
      * anneal -- Whether to apply annealing or use a constant temperature of 1.0; defaults to ``True``.
      * segment -- The name of the :class:`pyec.util.partitions.Segment` object that refers to the points and partitions.
      
      :param config: The configuration parameters.
      :type config: :class:`Config`
   """
   def __init__(self, config):
      super(Annealing, self).__init__(config)
      self.n = 0
      self.ids = []
      self.segment = None
      self.activeField = config.activeField
      config.taylorCenter = 1.0
      if not hasattr(config, 'taylorDepth'):
         config.taylorDepth = 0
   
   def sample(self):
      """
         Child classes should override this method in order to select a point
         from the active segment in :class:`pyec.util.partitions.Point`.
      """
      pass
                                                         
                                                                                                                                                                     
   def __call__(self, **kwargs):
      # handle initial case
      id = None
      area = 1.0
      if self.n == 0:
         center = self.config.initialDistribution()
      else:
         # select a mixture point
         try:
            point, area = self.sample()
            center = getattr(point, self.activeField)
            id = point.id
            self.ids.append(id)
         except Exception, msg:
            traceback.print_exc(file=sys.stdout)
            center = self.config.initialDistribution()
            
      if self.config.passArea:
         return center, area
      else:
         return center
      
   def batch(self, m):
      self.ids = []
      ret = [self() for i in xrange(m)]
      self.config.primaryPopulation = self.ids
      return ret
      
   def update(self, n, population):
      rerun = self.n == n
      self.n = n
      if hasattr(self.config, 'anneal') and not self.config.anneal:
         self.temp = 1.0
      elif hasattr(self.config, 'anneal_log') and self.config.anneal_log:
         self.temp = log(n)
      else:
         self.temp = -log(Partition.segment.partitionTree.largestArea())
         #print "TEMP: ", self.temp, self.temp**2, log(n)
      if self.segment is None:
         self.segment = Segment.objects.get(name=self.config.segment)
      if not rerun:
         if hasattr(self.config.fitness, 'train'):
            self.config.fitness.train(self, n)
      
class ProportionalAnnealing(Annealing):
   """
      Proportional Annealing, as described in 
      
      Lockett, A. And Miikkulainen, R. Real-space Evolutionary Annealing (2011).
      
      See :class:`Annealing` for more details.
   """
   def __init__(self, config):
      if not hasattr(config, 'taylorDepth'):
         config.taylorDepth = 10
      super(ProportionalAnnealing, self).__init__(config)
      
   def sample(self):
      return Point.objects.sampleProportional(self.segment, self.temp, self.config)
      #print "sampled: ", ret.binary, ret.score
      #return ret
      
   def update(self, n, population):
      rerun = n == self.n
      super(ProportionalAnnealing, self).update(n, population)
      if not rerun and .5 * floor(2*self.temp) > self.config.taylorCenter:
         ScoreTree.objects.resetTaylor(self.segment, .5 * floor(2 *self.temp), self.config)
      
class TournamentAnnealing(Annealing):
   """
      Tournament Annealing, as described in Chapter 11 of
      
      <http://nn.cs.utexas.edu/downloads/papers/lockett.thesis.pdf>
      
      See :class:`Annealing` for more details.
   """
   def sample(self):
      return Point.objects.sampleTournament(self.segment, self.temp, self.config)
      
      
