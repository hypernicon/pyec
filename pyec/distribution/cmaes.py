"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import copy

from numpy.linalg import cholesky, eig, inv, det
import numpy as np
from basic import Distribution, PopulationDistribution
from pyec.config import Config
from pyec.history import History

class CmaesHistory(History):
    """A history that stores the parameters for CMA-ES"""
    
    attrs = ["mu", "dim", "restart", "muw", "mueff", "cc", "cs", "c1", "cmu",
             "damps", "chiN", "eigeneval", "sigma"]
    npattrs = ["weights", "B", "D", 
               "covar", "active", "ps", "pc", "mean"]
    
    def __init__(self, dim=1, populationSize=25, mu=None, scale=1.0, 
                 restart=False):
        super(CmaesHistory, self).__init__()
        self.mu = mu or int(.5 * populationSize)
        self.scale = scale
        self.dim = dim
        self.restart = restart
      
        self.mean = None
        self.sigma = self.scale * .5
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)
      
        weights = (np.log(self.mu + .5) - 
                   np.log(np.array([i+1. for i in xrange(self.mu)])))
        weights /= np.sum(weights)
        muw = 1. / (np.sum(weights) ** 2)
        mueff = (np.sum(weights) ** 2) * muw
      
        cc = ((4+mueff/self.dim) / 
              (self.dim + 4 + 2 * mueff / self.dim));
        cs = (mueff + 2) / (self.dim + mueff + 5.)
      
        c1 = 2. / ((self.dim + 1.3)**2 + mueff)
        cmu = 2 * (mueff -2 + 1./mueff) / ((self.dim + 2.) ** 2 + mueff)
        damps = 1 + 2*max([0, np.sqrt((mueff-1)/(self.dim + 1)) - 1]) + cs
      
        self.weights = weights
        self.muw = muw
        self.mueff = mueff
        self.cc = np.float(cc)
        self.cs = np.float(cs)
        self.c1 = np.float(c1)
        self.cmu = np.float(cmu)
        self.damps = np.float(damps)
        self.chiN = np.sqrt(self.dim) * (1. - 1. / (4 * self.dim) + 
                                         1. / (21. * self.dim * self.dim))
        self.chiN = float(self.chiN)
      
        self.eigeneval = 0
        self.B = np.eye(self.dim, self.dim)
        self.D = np.ones(self.dim)
        self.covar = np.dot(self.B, 
                            np.dot(np.diag(self.D ** 2),self.B.transpose())) 
        self.active = np.dot(self.B, 
                             np.dot(np.diag(1./self.D),self.B.transpose()))
   
    def __getstate__(self):
        state = super(CmaesHistory, self).__getstate__()
        
        for attr in self.attrs:
            state[attr] = getattr(self, attr)
        
        for attr in self.npattrs:
            val = getattr(self, attr)
            if val is None:
                state[attr] = None
            else:
                state[attr] = val.copy()
        
        return state
    
    def __setstate__(self, state):
        state = copy.copy(state)
        for attr in self.npattrs:
            val = state[attr]
            if val is not None:
                val = val.copy()
            setattr(self, attr, val)
            del state[attr]
        super(CmaesHistory, self).__setstate__(state)
   
    def internalUpdate(self, population):
        base = np.array([x for x, s in population[:self.mu]])
        if self.mean is None: 
           oldMean = np.average([x for x,s in population])
        else:
           oldMean = self.mean
        self.mean = (np.outer(self.weights, np.ones(self.dim)) 
                     * base).sum(axis=0)
        cc = self.cc
        cs = self.cs
        muw = self.muw
        mueff = self.mueff
        chiN = self.chiN
      
        self.ps *= (1. - cs)
        self.ps += (np.sqrt(cs * (2. - cs) * mueff) * 
                    np.dot(self.active, self.mean - oldMean) / 
                    self.sigma)
       
        isonorm = np.sqrt((self.ps ** 2).sum())
        hsig = (isonorm / 
                np.sqrt(1 - (1 - cs) ** 
                            (2. * self.updates / len(population))) / 
                self.chiN)
        self.pc *= (1 - cc)
        if hsig < 1.4 + 2/(self.dim + 1):
           self.pc += (np.sqrt(cc * (2 - cc) * mueff) * 
                       (self.mean - oldMean) / 
                       self.sigma)  
           matfactor = np.outer(self.pc, self.pc)
        else:
           matfactor = np.outer(self.pc, self.pc) + cc * (2 - cc) * self.covar
      
        artmp = (base - oldMean) / self.sigma
        oldCovar = self.covar
        self.covar *= (1 - self.c1 - self.cmu)
        self.covar += self.c1 * matfactor
        self.covar += (self.cmu * 
                       np.dot(artmp.transpose(),
                              np.dot(np.diag(self.weights),artmp)))
          
        self.sigma *= np.exp((cs / self.damps) * (isonorm / chiN - 1.))
      
        if (self.updates - self.eigeneval > 
            len(population) / (self.c1 + self.cmu) / self.dim / 10.):
            try:
               self.eigeneval = self.updates
               self.covar = (np.triu(self.covar) + 
                             np.triu(self.covar,1).transpose())
               self.B, self.D = eig(self.covar)
               self.D = np.sqrt(np.diag(np.abs(self.D)))
         
               self.active = cholesky(self.covar) # np.dot(self.B, np.dot(np.diag(1. / self.D), self.B.transpose())) 
            except Exception:
               self.covar = oldCovar
        if self.sigma > self.scale:
            self.sigma = self.scale
        detcv = det(self.covar)
        if self.sigma * detcv > self.scale:
            self.sigma /= (detcv * self.sigma) 
            self.sigma *= self.scale
    
        if self.restart and self.sigma * detcv < 1e-25:
            self.__init__(self.dim, len(population), self.mu, self.scale, 
                          self.restart)


class Cmaes(PopulationDistribution):
   """
      The Correlated Matrix Adaptation Evolution Strategy algorithm as described by:
      
      Hansen, N. and Ostermeier, A. Completely derandomized self-adaptation in evolution strategies. In Evolutionary Computation 9(2), pp 159--195, 2001. 
   
      See <http://en.wikipedia.org/wiki/CMA-ES> for details.
   
      A fast-converging population-based optimizer that finds reasonably good
      solutions with relatively few resources. In general, each population is sampled from a multi-variate Gaussian whose parameters are altered to optimize the probability of finding good solutions. 
      
      
      Akimoto (2010) proved that CMA-ES is an instance of Natural Evolution Strategies (Wierstra, 2008), which implements a gradient-following method at the population level. 
      
      Config parameters 
      * mu -- the number of solutions from each generation used as parents for the next. By default equal to half the ``Config.populationSize``.
      * restart -- whether to restart when the distribution collapses; defaults to ``True`` (see e.g. `Auger and Hansen, 2005)
   
      :params cfg: The configuration object for CMA-ES.
      :type cfg: :class:`Config`
   
   """
   config = Config(mu=None, # the number of parents, default half population
                   restart=True, # whether to restart when variance collapses
                   populationSize=25)

   def __init__(self, **kwargs):
      super(Cmaes, self).__init__(**kwargs)
      if not issubclass(self.config.space.type, np.ndarray):
          err = ("This implementation of CMA-ES needs type "
                 "numpy.ndarray, not {0}")
          err = err.format(self.config.space.type)
          raise ValueError(err)
                           
      x = self.config.space.random()
      if not x.dtype == np.float:
          raise ValueError("CMA-ES expects numpy.ndarray with dtype float")
      self.sigma = self.config.space.scale * .5
      #if self.config.initial is None:
      #    self.mean = [self.config.space.random for i in xrange(popSize)]
      #elif hasattr(self.config.initial, 'batch'):
      #    self.mean = self.config.initial.batch(popSize)
      #else:
      #    self.mean = [self.config.initial() for i in xrange(popSize)]
      
      if self.config.mu >= self.config.populationSize:
          raise ValueError("CMA-ES mu must be less than populationSize")
      
      def history():
          return CmaesHistory(self.config.space.dim,
                              self.config.populationSize,
                              self.config.mu,
                              self.config.space.scale,
                              self.config.restart)
      self.config.history = history
      

   def compatible(self, history):
      return isinstance(history, CmaesHistory)

   def sample(self):
      h = self.history
      next = (h.mean + 
              h.sigma * 
              np.dot(h.active, np.random.randn(h.mean.size)))
      return next
            
   def batch(self, popSize):
      if self.history.mean is  None:
         if self.config.initial is None:
             return [self.config.space.random for i in xrange(popSize)]
         elif hasattr(self.config.initial, 'batch'):
             return self.config.initial.batch(popSize)
         else:
             return [self.config.initial() for i in xrange(popSize)]
      return [self.sample() for i in xrange(popSize)]
      
   def density(self, x):
      diff = self.history.mean - x
      pow = -.5 * np.dot(diff, np.dot(inv(self.history.covar), diff))
      d = (((((2*pi)**(len(x))) * det(self.history.covar)) ** -.5) * 
           np.exp(pow))
      return d
   
   @property
   def var(self):
      return self.history.sigma * det(self.history.covar)

   def internalUpdate(self, population):
      pass


