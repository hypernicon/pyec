"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


"""
   Based on Kolda, Lewis, and Torczon, Optimization by Direct Search: New Perspectives on Some Classical and Modern Methods (2003)
   
    Primarily implemented to enable MADS (Audet and Dennis, 2006)
"""


from numpy import *
from pyec.distribution.basic import PopulationDistribution, FixedCube, Gaussian
from pyec.config import Config, ConfigBuilder


GSS_INIT = 3
GSS_SEARCH = 1
GSS_POLL = 2

class GSSConfigurator(ConfigBuilder):
   
   def __init__(self, *args):
      super(GSSConfigurator, self).__init__(GeneratingSetSearch)
      self.cfg.tolerance = 1e-50
      self.cfg.expandStep = 1.1
      self.cfg.contractStep = .95
      self.cfg.stepInit = .5
      self.cfg.initialDistribution = FixedCube(self.cfg)


class GeneratingSetSearch(PopulationDistribution):
   def __init__(self, config):
      super(GeneratingSetSearch, self).__init__(config)
      self.step = self.config.stepInit * self.config.scale
      if hasattr(self.config, 'penalty'):
         self.penalty = self.config.penalty
      else:
         self.penalty = lambda x: 0
      self.generators = append(identity(self.config.dim), 
       -ones((1,self.config.dim))/sqrt(self.config.dim), axis=0)
      self.directions = zeros((0,self.config.dim))
      self.state = GSS_INIT
      self.center = None
      self.score = None
      
   @classmethod
   def configurator(cls):
      return GSSConfigurator(cls)
      
   def accept(self, x, score):
      if self.center is None:
         self.center = x
         self.score = score
         return True
      if score - self.penalty(self.step) > self.score:
         self.center = x
         self.score = score 
         return True
      return False

   @property
   def var(self):
      return self.step

   def expandStep(self):
      self.step *= self.config.expandStep
      
   def contractStep(self):
      self.step *= self.config.contractStep

   def poll(self):
      raw = self.center + self.step * self.generators
      """
      for i, row in enumerate(raw):
         while (abs(row - self.config.center) > self.config.scale).any():
            row -= self.center
            row *= .5
            row += self.center
      """
      return raw


   def search(self):
      if len(self.directions) == 0:
         return []
      raw = self.center + self.step * self.directions
      """
      for i, row in enumerate(raw):
         while (abs(row - self.config.center) > self.config.scale).any():
            row -= self.center
            row *= .5
            row += self.center
      """
      return raw

   def batch(self, popSize=None):
      if self.state == GSS_INIT:
         return [self.config.initialDistribution()]
      elif self.state == GSS_SEARCH:
         pop = self.search()
         if len(pop) > 0:
            return pop
      # GSS_POLL
      self.state = GSS_POLL
      return self.poll()
      
   def updateGenerators(self, pop):   
      pass

   def updateDirections(self, pop):
      pass
      
   def update(self, epoch, pop):
      if self.state == GSS_INIT:
         self.state = GSS_SEARCH
         self.accept(*(pop[0]))
      elif self.state == GSS_SEARCH:
         accepted = False
         for x,s in pop:
            accepted = accepted or self.accept(x,s)
         if accepted:
            self.expandStep()
            self.updateGenerators(pop)
            self.updateDirections(pop)
         else:
            self.state = GSS_POLL
      elif self.state == GSS_POLL:
         accepted = False
         for x,s in pop:
            accepted = accepted or self.accept(x,s)
         if accepted:
            self.state = GSS_SEARCH
            self.expandStep()
            self.updateGenerators(pop)
            self.updateDirections(pop)
         else:
            self.state = GSS_SEARCH
            self.contractStep()
            if self.step < self.config.tolerance:
               print "restarting"
               self.__init__(self.config)
               return
            #   #self.step = self.config.stepInit * self.config.scale
            self.updateGenerators(pop)
            self.updateDirections(pop)
      else:
         raise Exception, "missing state " + str(self.state)
         

class MADSConfigurator(ConfigBuilder):
   
   def __init__(self, *args):
      super(MADSConfigurator, self).__init__(MeshAdaptiveDirectSearch)
      self.cfg.tolerance = 1e-20
      self.cfg.expandStep = 4.0
      self.cfg.contractStep = .25
      self.cfg.stepInit = .5

class MeshAdaptiveDirectSearch(GeneratingSetSearch):
   def __init__(self, config):
      super(MeshAdaptiveDirectSearch, self).__init__(config)
      self.step = 1.0
      self.stepInit = self.step
      self.searchStep = self.step
      self.root = {}
      self.rootIdx = {}
   
   @classmethod
   def configurator(cls):
      return MADSConfigurator(cls)
   
   def ell(self):
      ell = int(-log(self.searchStep) * log(4))
      if ell > 30: ell = 30
      return ell
   
   def rootDirection(self):
      ell = self.ell()
      if self.root.has_key(ell):
         return
      i = random.randint(0, self.config.dim)
      bnd = 2 ** ell
      root = random.randint(-bnd+1, bnd, self.config.dim)
      if random.random_sample() < .5:
         root[i] = -bnd
      else:
         root[i] = bnd
      self.root[ell] = root
      self.rootIdx[ell] = i
      
   def expandStep(self):
      self.searchStep *= self.config.expandStep
      if self.searchStep > self.stepInit: self.searchStep = self.stepInit
      
   def contractStep(self):
      self.searchStep *= self.config.contractStep 
      if self.searchStep > self.stepInit: self.searchStep = self.stepInit
      
   def updateGenerators(self, pop):
      ell = self.ell()
      self.rootDirection()
      root = self.root[ell]
      idx = self.rootIdx[ell]
      bnd = 2 ** ell
      dim1 = self.config.dim - 1
      below = tri(dim1, k=-1) * random.randint(-bnd+1, bnd, (dim1,dim1))
      diagonal = (random.binomial(1, .5, dim1) * 2. - 1.) * bnd
      lower = below + bnd * diag(diagonal)
      indexes = arange(dim1)
      random.shuffle(indexes)
      basis = zeros((self.config.dim, self.config.dim))
      for i in xrange(self.config.dim):
         for j in xrange(dim1):
            if i == idx:
               basis[i,j] = 0
            else:
               i2 = i
               if i2 > idx:
                  i2 -= 1
               k = indexes[i2]
               if k >= idx:
                  k += 1 
               basis[k,j] = lower[i2,j]
      basis[idx,:] = 0.0
      basis[:,dim1] = root
      indexes2 = arange(self.config.dim + 1)
      basis2 = zeros_like(basis)
      for j in xrange(self.config.dim):
         basis2[:,indexes2[j]] = basis[:,j]
      extra = basis2.sum(axis=1).reshape((1,self.config.dim))
      self.generators = append(basis2, -extra, axis=0)
      self.step = self.config.dim * sqrt(self.searchStep)
      if self.step > 1.: self.step = 1.