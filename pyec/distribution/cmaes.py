"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from numpy import *
from numpy.linalg import cholesky, eig, inv, det
from basic import PopulationDistribution, FixedCube, Gaussian
from pyec.config import Config, ConfigBuilder


class CmaesConfigurator(ConfigBuilder):
   """
      A :class:`ConfigBuilder` object to construct the 
      Correlated Matrix Adaption Evolution Strategy (CMA-ES).
      
      By default, sets the ratio of `mu` over `lambda` to .5
   """

   def __init__(self, *args):
      super(CmaesConfigurator, self).__init__(Cmaes)
      self.cfg.muProportion = .5
      self.cfg.restart = True
   
   def postConfigure(self, cfg):
      if cfg.varInit is None:
         cfg.initialDistribution = FixedCube(cfg)
      else:
         cfg.usePrior = False
         cfg.initialDistribution = Gaussian(cfg)

   

class Cmaes(PopulationDistribution):
   """
      The Correlated Matrix Adaptation Evolution Strategy algorithm as described by:
      
      Hansen, N. and Ostermeier, A. Completely derandomized self-adaptation in evolution strategies. In Evolutionary Computation 9(2), pp 159--195, 2001. 
   
      See <http://en.wikipedia.org/wiki/CMA-ES> for details.
   
      A fast-converging population-based optimizer that finds reasonably good
      solutions with relatively few resources. In general, each population is sampled from a multi-variate Gaussian whose parameters are altered to optimize the probability of finding good solutions. 
      
      
      Akimoto (2010) proved that CMA-ES is an instance of Natural Evolution Strategies (Wierstra, 2008), which implements a gradient-following method at the population level. 
      
      Config parameters 
      * muProportion -- the ratio of `mu` to `lambda`, i.e. the proportion of solutions from each generation used as parents for the next. By default equal to 0.5.
      * initialDistribution -- the distribution used to sample the initial mean
      * populationSize -- the size of the population for each generation
      * scale -- the scale of the space, used to initialize the variance of the Gaussian
      * dim -- the dimension of the real vector space being searched.
      * restart -- whether to restart when the distribution collapses; defaults to ``True`` (see e.g. `Auger and Hansen, 2005)
   
      :params cfg: The configuration object for CMA-ES.
      :type cfg: :class:`Config`
   
   """

   def __init__(self, cfg):
      super(Cmaes, self).__init__(cfg)
      self.sigma = self.config.scale * .5
      self.mean = self.config.initialDistribution()
      self.ps = zeros(self.config.dim)
      self.pc = zeros(self.config.dim)
      self.mu = int(self.config.populationSize * self.config.muProportion)
      
      weights = log(self.mu + .5) - log(array([i+1. for i in xrange(self.mu)]))
      weights /= sum(weights)
      muw = 1. / sum(weights ** 2)
      mueff = (sum(weights) ** 2) * muw
      
      cc = (4+mueff/self.config.dim) / (self.config.dim + 4 + 2 * mueff / self.config.dim);
      cs = (mueff + 2) / (self.config.dim + mueff + 5.)
      
      c1 = 2. / ((self.config.dim + 1.3)**2 + mueff)
      cmu = 2 * (mueff -2 + 1./mueff) / ((self.config.dim + 2.) ** 2 + mueff)
      damps = 1 + 2*max([0, sqrt((mueff-1)/(self.config.dim + 1)) - 1]) + cs
      
      self.weights = weights
      self.muw = muw
      self.mueff = mueff
      self.cc = cc
      self.cs = cs
      self.c1 = c1
      self.cmu = cmu
      self.damps = damps
      self.chiN = sqrt(self.config.dim) * (1. - 1. / (4 * self.config.dim) + 1. / (21. * self.config.dim * self.config.dim))
      
      self.eigeneval = 0
      self.B = eye(self.config.dim, self.config.dim)
      self.D = ones(self.config.dim)
      self.covar = dot(self.B, dot(diag(self.D ** 2),self.B.transpose())) 
      self.active = dot(self.B, dot(diag(1./self.D),self.B.transpose()))


   @classmethod
   def configurator(cls):
      return CmaesConfigurator(cls)
   
   def __call__(self):
      next = self.mean + self.sigma * dot(self.active, random.randn(self.mean.size))
      return next
            
   def batch(self, popSize):
      if self.mean is  None:
         if hasattr(self.config.initialDistribution, 'batch'):
            return self.config.initialDistribution.batch(popSize)
         else:
            return [self.config.initialDistribution() for i in xrange(popSize)]
      return [self.__call__() for i in xrange(popSize)]
      
   def density(self, x):
      diff = self.mean - x
      pow = -.5 * dot(diff, dot(inv(self.covar), diff))
      d = ((((2*pi)**(len(x))) * det(self.covar)) ** -.5) * exp(pow)
      return d
   
   @property
   def var(self):
      return self.sigma * det(self.covar)

   def update(self, n, population):
      base = array([x for x, s in population[:self.mu]])
      if self.mean is None: 
         oldMean = average([x for x,s in population])
      else:
         oldMean = self.mean
      self.mean = (outer(self.weights, ones(self.config.dim)) * base).sum(axis=0)
      cc = self.cc
      cs = self.cs
      muw = self.muw
      mueff = self.mueff
      chiN = self.chiN
      
      self.ps *= (1. - cs)
      self.ps += sqrt(cs * (2. - cs) * mueff) \
       * dot(self.active, self.mean - oldMean) / self.sigma
       
      isonorm = sqrt((self.ps ** 2).sum())
      hsig = isonorm / sqrt(1 - (1 - cs) ** (2. * n / len(population))) / self.chiN
      self.pc *= (1 - cc)
      if hsig < 1.4 + 2/(self.config.dim + 1):
         self.pc += sqrt(cc * (2 - cc) * mueff) * (self.mean - oldMean) / self.sigma  
         matfactor = outer(self.pc, self.pc)
      else:
         matfactor = outer(self.pc, self.pc) + cc * (2 - cc) * self.covar
      
      artmp = (base - oldMean) / self.sigma
      oldCovar = self.covar
      self.covar *= (1 - self.c1 - self.cmu)
      self.covar += self.c1 * matfactor
      self.covar += self.cmu * dot(artmp.transpose(), dot(diag(self.weights),artmp))
          
      self.sigma *= exp((cs / self.damps) * (isonorm / chiN - 1.))
      
      if n - self.eigeneval > len(population) / (self.c1 + self.cmu) / self.config.dim / 10.:
         try:
            self.eigenval = n
            self.covar = triu(self.covar) + triu(self.covar,1).transpose()
            self.B, self.D = eig(self.covar)
            self.D = sqrt(diag(abs(self.D)))
         
            self.active = cholesky(self.covar) # dot(self.B, dot(diag(1. / self.D), self.B.transpose())) 
         except:
            self.covar = oldCovar
      if self.sigma > self.config.scale:
         self.sigma = self.config.scale
      detcv = det(self.covar)
      if self.sigma * detcv > self.config.scale:
         self.sigma /= (detcv * self.sigma) 
         self.sigma *= self.config.scale
    
      if self.config.restart and self.sigma * detcv < 1e-25:
      #   print "restarting"
         self.__init__(self.config)

