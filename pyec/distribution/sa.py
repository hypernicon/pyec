"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np

from .basic import PopulationDistribution
from pyec.distribution.bayes.structure.proposal import StructureProposal
from pyec.distribution.ec.mutators import Gaussian, Bernoulli

from pyec.config import Config
from pyec.history import DoubleMarkovHistory

_ = Config

class SimulatedAnnealingAcceptance(PopulationDistribution):
   """A selector that implements the acceptance probability for simulated
   annealing. Computes the acceptance ratio and samples it. If the population
   size is greater than one, then this distribution maintains an array
   of accepted values, and can be used to run multiple concurrent
   markov chains.
   
   Config parameters
   
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
   * restart -- A probability of restarting, tested at each update
   * divisor -- A divisor that divides the ``updates`` property of
                :class:`History`, scaling the rate of decline in the temperature
   * discount -- The decay factor for the built-in discount schedule; ``temp0``
                 is multiplied by ``discount`` once each time the ``update``
                 method is called
                
   """
   config = Config(schedule="log",
                   learningRate = 1.0,
                   temp0 = 1.0,
                   restart = 0.0,
                   divisor = 1.0,
                   discount = .99,
                   populationSize = 1.0,
                   history = DoubleMarkovHistory)

   def compatible(self, history):
      return (hasattr(history, 'lastPopulation')
              and hasattr(history, 'penultimate'))

   def batch(self, popSize):
      temp = self.temperature()
      last = self.history.lastPopulation()
      penultimate = self.history.penultimate()
      scoreProposed = np.array([s for x,s in lastPopulation])
      scoreAccepted = np.array([s for x,s in penultimate])
      exponent = (scoreProposed - scoreAccepted) / temp
      if not self.config.minimize:
         exponent = -exponent
      probs = minimum(1.0, np.exp(exponent))
      selection = np.random.binomial(1, probs, shape(probs))
      result = []
      for i,sel in enumerate(selection):
         if sel > 0.5:
            result.append(last[i][0])
         else:
            result.append(penultimate[i][0])
      return result
      
   def temperature(self):
      n = self.history.updates / self.config.divisor
      if hasattr(self.config.schedule, '__call__'):
         return self.config.schedule(n)
      elif self.config.schedule == "linear":
         return 1. / (n * self.learningRate)
      elif self.config.schedule == "log":
         return 1. / (log(n) * self.learningRate)
      elif self.config.schedule == "discount":
         return 1. / (self.config.temp0 * (self.config.discount ** n))


# Euclidean space
RealSimulatedAnnealing = (
   Gaussian[_(sd=.005)] << SimulatedAnnealingAcceptance
)

# fixed-length bit strings
BinarySimulatedAnnealing = (
   Bernoulli[_(p=.01)] << SimulatedAnnealingAcceptance
)

# Structure search in a Bayes net
BayesNetSimulatedAnnealing = (
   StructureProposal[_(branchFactor=5)] <<
   SimulatedAnnealingAcceptance[_(learningRate=0.1,
                                  temp0=100.,
                                  discount=0.95,
                                  divisor=400.)]
)[_(minimize=False)]
