"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import copy
import numpy as np
import random
import sys
import traceback


from pyec.config import Config
from pyec.distribution.basic import PopulationDistribution
from pyec.history import SortedMarkovHistory
from pyec.util.TernaryString import TernaryString

import logging
logger = logging.getLogger(__file__)

class Selection(PopulationDistribution):
   """A selection method (Abstract class)"""
   config = Config(history=SortedMarkovHistory)
   
   def needsScores(self):
      return True
   
   def compatible(self, history):
      return hasattr(history, 'lastPopulation')

class BestSelection(Selection):
   """
      Select the best member of the prior population.
      
   """
   
   def __init__(self, **kwargs):
      super(Selection, self).__init__(**kwargs)
      
   def batch(self, popSize):
      return [self.history.lastPopulation()[0][0] for i in xrange(popSize)]
   

class EvolutionStrategySelection(Selection):
   """
      Implements standard selection for evolution strategies.
      
      The property ``config.selection`` determines the type of selection,
      either "plus" or "comma". The config should provide \mu, and the
      population size together with mu and the selection type determines
      lambda.
      
      Config parameters
      
      * mu -- The number of parents to be taken from the prior population
      * selection -- Either "plus" selection or "comma"; "comma" is default
      
   """
   config = Config(mu=None,
                   selection="comma")
   
   def __init__(self, **kwargs):
      super(EvolutionStrategySelection, self).__init__(**kwargs)
      self.total = 0
      self.mu = self.config.mu
      self.plus = self.config.selection == 'plus' # plus or comma selection, for ES

   def sample(self):
      idx = np.random.randint(0,self.mu-1)
      return self.lastPopulation()[idx][0]

   def batch(self, popSize):
      if self.plus:
         return ([x for x,s in population[:self.mu]]
                  + [self.sample() for i in xrange(popSize - self.mu)])
      else:
         return [self.sample() for i in xrange(popSize)]


class GeneralizedProportional(Selection):
   """
      Fitness proportional selection with a modulating function.
      
      Config parameters:
      
      * modulator -- A funtion applied to the fitness; proportional selection
                     is performed on the image of this function. For fitness
                     $f$, this is $modulator\cdot f$. The modulator is given
                     the index of the element in the population second.
                     The modulator is passed
                     the configuration function as a third argument; if this
                     causes an exception, then on the score is passed. That is,
                     the fitness ``s = f(x)`` is computed, and then a call is
                     made to ``modulator(s,i,cfg)``, and if this causes an
                     exception, ``modulator(s,i)`` is used.

   """
   config = Config(modulator=None)
   
   def __init__(self, **kwargs):
      super(GeneralizedProportional, self).__init__(**kwargs)
      self.total = 0
      self.matchedPopulation = []

   def sample(self):
      rnd = np.random.random_sample() * self.total
      for x,amt in self.matchedPopulation:
         if amt >= rnd:
            return x
      return self.matchedPopulation[0][0]

   def batch(self, popSize):
      return [self.sample() for i in xrange(popSize)]

   def update(self, history, fitness):
      super(Proportional, self).update(history, fitness)
      self.buildProbabilities(self.history.lastPopulation())
      
   def buildProbabilities(self, lastPopulation):
      """Take the last population (with scores) and recompute the
      sampling vector for proportional selection.
      
      :param lastPopulation: The last population, of size
                             ``config.populationSize``
      :type lastPopulation: A list of (point, score) tuples
      
      """
      try:
         population = [(p[0], self.config.modulator(p[1],i,cfg))
                       for i,p in enumerate(lastPopulation)]
      except:
         population = [(x, self.config.modulator(s))
                       for x,s in lastPopulation]
      self.total = sum([s for x,s in population])
      self.matchedPopulation = []
      
      amt = 0
      for x,s in population:
         amt += s
         self.matchedPopulation.append((x,amt))


class Proportional(GeneralizedProportional):
   """
      Fitness proportional selection (roulette wheel selection).
      
      See <http://en.wikipedia.org/wiki/Fitness_proportionate_selection>.
      
      Fitness values must be nonnegative.
      
      This :class:`Selection` method is just :class:`GeneralizedProportional`
      with a modulating function
      
   """
   config = Config(modulator = lambda s,i,cfg:
                      cfg.minimize and -abs(s) or abs(s))

class ExponentiatedProportional(GeneralizedPropotional):
   """Fitness propotional selection, but with an exponentional modulationg
   function so that any fitness values may be used.
   
   $p(x) = \exp(\frac{-f(x)}{T})$
   
   Config parameters:
   
   * T -- A "temperature" to divide the explicand.
   
   """
   config = Config(T=1.0, # a temperature value
                   modulator=lambda s,i,cfg:
                      cfg.minimize and np.exp(-s/cfg.T) or np.exp(s/cfg.T))

   
class Tournament(GeneralizeProportional):
   """
      Tournament selection on the entire population.
      
      See <http://en.wikipedia.org/wiki/Tournament_selection>.
      
      This class extends :class:`GeneralizedProportional`, but only to
      get the method ``buildProbabilities`` (and the particular use of
      ``batch``). Its modulating function is
      built-in and is just the rank of the member in the population.
      
      Config parameters:
      * pressure -- The probability of choosing the best
                    member of the randomly subsampled population.
      * order -- The size of the subsample to select from, or ``None``
                 for tournament selection over the entire population
      
   """
   config = Config(pressure=0.1,
                   order=None,
                   history=SortedMarkovHistory)
   
   def __init__(self, **kwargs):
      super(Tournament, self).__init__(**kwargs)
      self.pressure = config.pressure
      self.order = self.config.order or self.config.populationSize
      self.ordered = None

   def compatible(self, history):
      return super(Tournament, self).compatible(history) and history.sorted

   def sample(self):
      newpop = random.shuffle(self.ordered)[:self.order]
      newpop = sorted(newpop, key=lambda x: x[1])
      idx = 0
      while True:
         rnd = np.random.random_sample()
         if rnd <= self.pressure:
            return newpop[idx % self.order][0]
         idx += 1

   def buildProbabilities(self, lastPopulation):
      self.ordered = [(p[0],i) for i,p in
                      enumerate(self.history.lastPopulation())]  


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
   def __init__(self, config):
      """pressure between 1.0 and 2.0"""
      self.pressure = config.pressure
      self.popSize = config.populationSize

   def __call__(self, rank):
      return (2 - self.pressure +
              (2 * (self.pressure-1))* ((rank-1.0)/(self.popSize - 1.0))) 
      
class NonlinearRanker(Ranker):
   """
      Computes the probability of selecting a solution from a population using nonlinear ranking selection.
   
      See e.g. <http://www.pohlheim.com/Papers/mpga_gal95/gal2_3.html>
   """

   def __init__(self, config):
      self.pressure = config.pressure
      self.coeffs = [self.pressure for i in xrange(self.config.populationSize)]
      self.coeffs[0] -= popSize
      self.root = roots(self.coeffs)[0].real

   def __call__(self, rank):
      """ root is root of (pressure * sum_k=0^(popSize-1) x^k) - popSize * x ^(popSize - 1)"""
      return self.root ** (rank - 1.0) 


class Ranking(GeneralizedProportional):
   """
      Ranking selection, e.g.
      <http://www.pohlheim.com/Papers/mpga_gal95/gal2_3.html>
   
      Takes a ranking function which weights individuals according to rank
      Rank is 1 for lowest, K for highest in population of size K
      
      Note that the ranking function is a class, instantiated with the
      algorithm's config. When this method is initialized, the ranking
      function is instanted and placed in ``config.rankerInst``, unless
      the ``config.rankerInst`` is provided first
      
      Config parameters
      
      * pressure -- The selection pressure passed to the ranker.
      * ranker -- A :class:`Ranker` subclass for the ranking policy.
      * rankerInst -- A :class:`Ranker` instance for the ranking policy;
                      if not provided, then ``ranker(config)`` is used.
    
   """
   config = config(pressure=1.0,
                   ranker=LinearRanker,
                   modulator=lambda s,i,cfg:
                     cfg.rankerInst(cfg.populationSize-i))
   
   def __init__(self, **kwargs):
      super(Ranking, self).__init__(**kwargs)
      if self.config.rankerInst is None:
         self.config.rankerInst = self.config.ranker(self.config)
  
   def density(self, idx):
      return self.ranker(idx, len(self.matchedPopulation)) / self.total


class Elitist(Selection):
   """
      Implements elitism by replacing the worst member of the population
      with the best solution seen so far. If the best solution is a member
      of the current population, nothing is changed.
      
      Elitism can be applied to an algorithm by piping the algorithm
      with elitism, i.e.::
      
      ElitistVersion = .2 * Elitist | .8 * SomeAlgorithm
      
      for a new version of ``SomeAlgorithm`` that selects the first
      20 % of the population as the best solutions so far, and fills
      in the rest of the population with ``SomeAlgorithm``.
      
   """
   def batch(self, popSize):
      return self.population


