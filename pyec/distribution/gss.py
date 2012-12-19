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
import copy
import numpy as np

from pyec.distribution.basic import Distribution, PopulationDistribution
from pyec.config import Config
from pyec.history import History
from pyec.space import Euclidean

GSS_INIT = 3
GSS_SEARCH = 1
GSS_POLL = 2

class GeneratingSetSearchHistory(History):
   
   def __init__(self, cfg):
      super(GeneratingSetSearchHistory, self).__init__(cfg)
      self.attrs |= set(["dim","center","state", "_score", "step", "stepInit",
                         "generators","directions"])
      dim = self.config.space.dim
      scale = self.config.space.scale
      step = self.config.step
      self.state = GSS_INIT
      self.generators = np.append(np.identity(dim), 
                                  -np.ones((1,dim))
                                  /np.sqrt(dim), axis=0)
      self.directions = np.zeros((0,dim))
      self.stepInit = step * scale
      self.step = self.stepInit
      self.center = None
      self._score = None
      self.dim = dim


   def accept(self, x, score):
      if self.center is None:
         self.center = x
         self._score = score
         return True
      
      if self.better(score + self.penalty(self.step), self._score):
         self.center = x
         self._score = score 
         return True
      
      return False

   def penalty(self, step):
      """A penalty to apply during acceptance testing, used to accept
      non-optimal points (or to reject new points that do not provide
      enough improvement). By default, non-optimal points are not
      accepted.
      
      :param step: The current step size
      :type step: ``float``
      
      """
      return 0.0

   def expandStep(self):
      self.step *= self.config.expand

   def contractStep(self):
      self.step *= self.config.contract

   def updateGenerators(self, pop):   
      pass

   def updateDirections(self, pop):
      pass
      
   def internalUpdate(self, pop):
      if self.state == GSS_INIT:
         if len(self.directions) > 0:
            self.state = GSS_SEARCH
         else:
            self.state = GSS_POLL
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
            if len(self.directions) > 0:
               self.state = GSS_SEARCH
            self.expandStep()
            self.updateGenerators(pop)
            self.updateDirections(pop)
         else:
            if len(self.directions) > 0:
               self.state = GSS_SEARCH
               
            self.contractStep()
            if (self.step < self.config.tol).all():
               self.__init__(self.config)
               return
            #   #self.step = self.config.stepInit * self.config.scale
            self.updateGenerators(pop)
            self.updateDirections(pop)
      else:
         raise Exception, "missing state " + str(self.state)


class GeneratingSetSearch(PopulationDistribution):
   """
   Based on Kolda, Lewis, and Torczon, Optimization by Direct Search: New Perspectives on Some Classical and Modern Methods (2003)
   
    Primarily implemented to enable MADS (Audet and Dennis, 2006)
   
   """
   config = Config(tol=1e-50,              # tolerance before restart
                   expand = 1.1,           # multiplier for expansion
                   contract = .95,         # multiplier for contraction
                   step = .5,              # initial step
                   history = GeneratingSetSearchHistory)
   
   def __init__(self, **kwargs):
      super(GeneratingSetSearch, self).__init__(**kwargs)
      if not isinstance(self.config.space, Euclidean):
         raise ValueError("Cannot use Nelder-Mead in non-Euclidean spaces.")
      
      self.config.populationSize = 1
      
   def compatible(self,history):
      return isinstance(history, GeneratingSetSearchHistory)
      
   @property
   def var(self):
      return self.history.step

   def poll(self):
      raw = self.history.center + self.history.step * self.history.generators
      """
      for i, row in enumerate(raw):
         while (abs(row - self.config.center) > self.config.scale).any():
            row -= self.center
            row *= .5
            row += self.center
      """
      return raw

   def search(self):
      if len(self.history.directions) == 0:
         return []
      raw = self.history.center + self.history.step * self.history.directions
      """
      for i, row in enumerate(raw):
         while (abs(row - self.config.center) > self.config.scale).any():
            row -= self.center
            row *= .5
            row += self.center
      """
      return raw

   def batch(self, popSize=None):
      state = self.history.state
      if state == GSS_SEARCH:
         pop = self.search()
         if len(pop) > 0:
            return pop
      elif state == GSS_POLL: # GSS_POLL
         state = GSS_POLL
         return self.poll()
      elif state == GSS_INIT: 
         if isinstance(self.config.initial, Distribution):
            return self.config.initial.sample()
         elif self.config.initial is not None:
            return self.config.initial
         else:
            return self.config.space.random()
      raise Exception("Unknown state in GSS: {0}".format(state))



"""
class MADSHistory(GeneratingSetSearchHistory):
   """"""
   Based on Audet & Dennis 2006. May not be working yet; please verify/fix
   if you need it.
   """"""
   def __init__(self, config):
      super(MeshAdaptiveDirectSearch, self).__init__(config)
      self.searchStep = self.step
      self.root = {}
      self.rootIdx = {}
    
   def ell(self):
      ell = np.int(-np.log(self.searchStep) * np.log(4))
      if ell > 30: ell = 30
      return ell
   
   def rootDirection(self):
      ell = self.ell()
      if self.root.has_key(ell):
         return
      i = np.random.randint(0, self.config.dim)
      bnd = 2 ** ell
      root = np.random.randint(-bnd+1, bnd, self.config.dim)
      if np.random.random_sample() < .5:
         root[i] = -bnd
      else:
         root[i] = bnd
      self.root[ell] = root
      self.rootIdx[ell] = i
      
   def expandStep(self):
      self.searchStep *= self.config.expand
      if self.searchStep > self.stepInit:
         self.searchStep = self.stepInit
      
   def contractStep(self):
      self.searchStep *= self.config.contract 
      if self.searchStep > self.stepInit:
         self.searchStep = self.stepInit
      
   def updateGenerators(self, pop):
      ell = self.ell()
      self.rootDirection()
      root = self.root[ell]
      idx = self.rootIdx[ell]
      bnd = 2 ** ell
      dim1 = self.config.dim - 1
      below = np.tri(dim1, k=-1) * np.random.randint(-bnd+1, bnd, (dim1,dim1))
      diagonal = (np.random.binomial(1, .5, dim1) * 2. - 1.) * bnd
      lower = below + bnd * np.diag(diagonal)
      indexes = np.arange(dim1)
      np.random.shuffle(indexes)
      basis = np.zeros((self.config.dim, self.config.dim))
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
      indexes2 = np.arange(self.config.dim + 1)
      basis2 = np.zeros_like(basis)
      for j in xrange(self.config.dim):
         basis2[:,indexes2[j]] = basis[:,j]
      extra = basis2.sum(axis=1).reshape((1,self.config.dim))
      self.generators = np.append(basis2, -extra, axis=0)
      self.step = self.config.dim * np.sqrt(self.searchStep)
      if self.step > 1.: self.step = 1.
      
"""
