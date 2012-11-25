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
   
   attrs = ["state", "generators", "directions", "step", "center", "_score",
            "expand", "contract", "tolerance", "dim"]
   
   def __init__(self, dim, scale, step, expand, contract, tolerance):
      super(GeneratingSetSearchHistory, self).__init__()
      self.state = GSS_INIT
      self.generators = np.append(np.identity(dim), 
                                  -np.ones((1,dim))
                                  /np.sqrt(dim), axis=0)
      self.directions = np.zeros((0,dim))
      self.step = step * scale
      self.center = None
      self._score = None
      self.expand = expand
      self.contract = contract
      self.tolerance = tolerance
      self.dim = dim

   def __getstate__(self):
      state = super(GeneratingSetSearchHistory, self).__getstate__()
      
      for attr in self.attrs:
         val = getattr(self, attr)
         if isinstance(val, np.ndarray):
            val = val.copy()
         state[attr] = val
         
      return state

   def __setstate_(self, state):
      state = copy.copy(state)
      
      for attr in self.attrs:
         val = state.pop(attr)
         if isintance(val, np.ndarray):
            val = val.copy()
         setattr(self, attr, val)
      
      super(GeneratingSetSearchHistory, self).__setstate__(state)

   def better(self, x, y):
      """Determine one score is better than another. The comparison
      depends on whether minimizing or maximizing. Default is minimization
      
      """
      return x < y
      

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
      self.step *= self.expand

   def contractStep(self):
      self.step *= self.contract

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
            if self.step < self.tolerance:
               print "restarting"
               self.__init__(self.dim,
                             self.scale,
                             self.step,
                             self.expand,
                             self.contract,
                             self.tolerance)
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
                   step = .5)              # initial step
   
   def __init__(self, **kwargs):
      super(GeneratingSetSearch, self).__init__(**kwargs)
      if not isinstance(self.config.space, Euclidean):
         raise ValueError("Cannot use Nelder-Mead in non-Euclidean spaces.")
      
      self.config.populationSize = 1
      
      def history():
         return GeneratingSetSearchHistory(self.config.space.dim,
                                           self.config.space.scale,
                                           self.config.step,
                                           self.config.expand,
                                           self.config.contract,
                                           self.config.tol)
      self.config.history = history
      
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
      
"""