"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from numpy import *
from basic import PopulationDistribution, Gaussian, FixedCube
from pyec.distribution.bayes.structure.proposal import StructureProposal

from pyec.config import ConfigBuilder

class SAConfigurator(ConfigBuilder):
   
   def __init__(self, *args):
      super(SAConfigurator, self).__init__(SimulatedAnnealing)
      self.cfg.usePrior = True
      self.cfg.divisor = 1.0
      self.cfg.schedule = "log"
      self.cfg.learningRate = 1.0
      self.cfg.varInit = .005
      self.cfg.restartProb = .001
      self.cfg.initialTemp = 1.0
      
   def postConfigure(self, cfg):
      cfg.initial = cfg.initialDistribution = FixedCube(cfg)
      cfg.proposal = Gaussian(cfg)

class BayesSAConfigurator(SAConfigurator):
   def __init__(self):
      super(BayesSAConfigurator, self).__init__()
      self.cfg.varInit=1
      self.cfg.restartProb = .01
      self.cfg.learningRate = 0.1
      self.cfg.branchFactor = 5
      self.cfg.schedule = "discount"
      self.cfg.initialTemp = 100
      self.cfg.tempDiscount = 0.95
      self.cfg.divisor = 400

   def postConfigure(self, cfg):
      cfg.structureGenerator = StructureProposal(cfg)
      cfg.proposal = cfg.structureGenerator
      cfg.initial = cfg.initialDistribution = cfg.structureGenerator

# now define general sampler
class SimulatedAnnealing(PopulationDistribution):
   def __init__(self, config):
      super(SimulatedAnnealing, self).__init__(self)
      self.dim = config.dim
      self.proposal = config.proposal
      self.config = config
      self.points = None
      self.accepted = 0
      self.learningRate = config.learningRate
      self.total = 0
      self.offset = 0
      self.n = 1
      config.sort = False
      self.best = None
      self.bestScore = None
      self.bestVar = self.config.varInit
      
   @classmethod
   def configurator(cls):
      return SAConfigurator(cls)   
      
   def proposalInner(self, **kwargs):
      x = self.proposal(**kwargs)
      while self.config.bounded and not self.config.in_bounds(x):
         x = self.proposal(**kwargs)
      return x
      
   def batch(self, howMany):
      if self.points is None:
         if hasattr(self.config.initialDistribution, 'batch'):
            return self.config.initialDistribution.batch(howMany)
         return [self.config.initialDistribution() for i in xrange(howMany)]
      return [self.proposalInner(prior=self.points[i][0], idx=i) for i in xrange(howMany)]

   def density(self, point):
      return self.proposal.density(point, self.points[0][0])

   @property
   def var(self):
      if hasattr(self.proposal, 'var'):
         if hasattr(self.proposal.var, '__len__'):
            return self.proposal.var.sum() / len(self.proposal.var)
         return self.proposal.var
      elif hasattr(self.proposal, 'bitFlipProbs'):
         return self.proposal.bitFlipProbs
      
   def temperature(self):
      if hasattr(self.config.schedule, '__call__'):
         return self.config.schedule(self.n - self.offset)
      elif self.config.schedule == "linear":
         return 1. / self.learningRate * (((self.n - self.offset)/self.config.divisor))
      elif self.config.schedule == "log":
         return 1. / self.learningRate * (log((self.n - self.offset)/self.config.divisor))
      elif self.config.schedule == "discount":
         rate = floor(self.n / self.config.divisor)
         return self.config.initialTemp * (self.config.tempDiscount ** rate)

   def update(self, n, population):
      self.n = n
      if self.points is None:
         self.accepted = zeros(len(population))
         self.total = ones(len(population))
         self.points = [(x,s) for x,s in population]
         return
      #print "generation ", n, ":", [s for x,s in self.points]
      #print self.proposal.var
      
      if self.best is None: 
         self.best = [None for i in xrange(len(population))]
         self.bestScore = [None for i in xrange(len(population))]
         self.bestVar = self.bestVar * ones(len(population))
      for i in xrange(len(population)):
         if self.best[i] is None or self.bestScore[i] < population[i][1]:
            self.best[i] = population[i][0]
            self.bestScore[i] = population[i][1]
            if hasattr(self.proposal, 'var'):
               if hasattr(self.proposal.var, '__len__'):
                  self.bestVar[i] = self.proposal.var[i]
               else:
                  self.bestVar[i] = self.proposal.var
         self.total[i] += 1
         if self.points[i][1] < population[i][1]:
            #print "accepted better", i, self.points[i][1], population[i][1]
            #print 1.0, self.points[i][1], population[i][1]
            self.points[i] = population[i]
            self.accepted[i] += 1
         else:
            temp = self.temperature()
            ratio = exp(-(self.points[i][1] - population[i][1])/temp)
            ratio *= self.proposal.densityRatio(self.points[i][0], population[i][0], i)
            #print ratio, self.points[i][1], population[i][1]
            test = random.random_sample()
            if test <= ratio:
               self.accepted[i] += 1
               #print "accepted worse", i, self.points[i][1], population[i][1]
               self.points[i] = population[i]
                 
      #print self.accepted / self.total
      if n % 25 == 0:
         self.proposal.adjust(self.accepted / self.total)
         self.accepted = zeros(len(population))
         self.total = zeros(len(population))
      
      #randomly restart
      if random.random_sample() < self.config.restartProb:
         self.points = [(self.best[i], self.bestScore[i]) for i in xrange(len(population))]
         self.offset = n - 1 
         if hasattr(self.proposal, 'var'):
            self.proposal.var = self.config.varInit
      
         
      
      
