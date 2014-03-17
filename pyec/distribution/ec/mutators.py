"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import binascii
import copy
import inspect
import numpy as np
import struct

from pyec.util.TernaryString import TernaryString
from pyec.config import Config
from pyec.distribution.basic import PopulationDistribution
#from pyec.util.partitions import Segment, Partition
from pyec.history import MarkovHistory, MultiStepMarkovHistory
from pyec.space import EndogeneousProduct

class Crosser(object):
   """ 
      Base class to perform recombination on a given list of organisms.
      
      :param config: The configuration parameters.
      :type config: :class:`Config`
      
   """
   def __init__(self, config):
      self.config = config
   
   def __call__(self, orgs, prob):
      """
         Perform recombination.
         
         :param orgs: List of parent organisms (may be more than 2)
         :type orgs: list
         :param prob: Probability of performing any crossover. Returns the first item in ``orgs`` if a bernoulli test on this parameter fails.
         :type prob: float between 0 and 1
         :returns: The recombined organism.
         
      """
      return orgs[0]


class UniformCrosser(Crosser):
   """
      Uniform crossover.
      
      See <http://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)#Uniform_Crossover_and_Half_Uniform_Crossover>
   
   """
   def __call__(self, orgs, prob):
      if np.random.random_sample() > prob:
         return orgs[0]
      if self.config.space.type == np.ndarray:
         rnd = np.random.random_sample(len(orgs[0])).round()
         return rnd * orgs[0] + (1 - rnd) * orgs[1]
      elif self.config.space.type == TernaryString:
         rnd = np.random.bytes(len(str(orgs[0])) * 8)
         rnd = long(binascii.hexlify(rnd), 16)
         base = rnd & orgs[0].base | ~rnd & orgs[1].base
         known = rnd & orgs[0].known | ~rnd & orgs[1].known
         return TernaryString(base, known, self.config.space.dim)
      else:
         err = "Unknown type for UniformCrossover: {0}"
         raise NotImplementException(err.format(self.config.space.type))


class OnePointDualCrosser(Crosser):
   """
      One point crossover, mixes two individuals to create two new individuals by splicing them together at a random index.
      
      So:: 
      
         xxxxxx
         oooooo
      
      might become::
      
         xxoooo
         ooxxxx
      
      And both items are returned.
      
      See <http://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)#One-point_crossover>.
      
   """
   dual = True
   def __call__(self, orgs, prob=1.0):
      if np.random.random_sample() > prob:
         return orgs[0], orgs[1]
      if isinstance(orgs[0], np.ndarray):
         idx = np.random.random_integers(0, len(orgs[0]) - 1)
         return (np.append(orgs[0][:idx], orgs[1][idx:], axis=0),
                 np.append(orgs[1][:idx], orgs[0][idx:], axis=0))
      elif isinstance(orgs[0], TernaryString):
         idx = np.random.random_integers(0, orgs[0].length - 1)
         ret1 = TernaryString(orgs[0].base, orgs[0].known, orgs[0].length)
         ret2 = TernaryString(orgs[1].base, orgs[1].known, orgs[1].length)
         ret1[idx:] = orgs[1][idx:]
         ret2[idx:] = orgs[0][idx:]
         return ret1,ret2
      else:
         raise ValueError("Received unknown type in OnePointDualCrosser.")


class OnePointCrosser(Crosser):
   """
      Cross two organisms at one point and return only one crossed version.
      
      Unlike :class:`OnePointDualCrosser`, this discards the second variant.
   
      See <http://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)#One-point_crossover>.
   """
   def __call__(self, orgs, prob=1.0):
      if np.random.random_sample() > prob:
         return orgs[0], orgs[1]
      if isinstance(orgs[0], np.ndarray):
         idx = np.random.random_integers(0, len(orgs[0]) - 1)
         return np.append(orgs[0][:idx], orgs[1][idx:], axis=0)
      elif isinstance(orgs[0], TernaryString):
         idx = np.random.random_integers(0, orgs[0].length - 1)
         ret = TernaryString(orgs[0].base, orgs[0].known, orgs[0].length)
         ret[idx:] = orgs[1][idx:]
         return ret
      else:
         raise ValueError("Received unknown type in OnePointCrosser")
      
class TwoPointCrosser(Crosser):
   """
      Cross two organism by wrapping them into a circle, picking two indices, and combining the two rings. 
      
      See <http://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)#Two-point_crossover>.
   """
   def __call__(self, orgs, prob=1.0):
      if np.random.random_sample() > prob:
         return orgs[0]
      idx1 = np.random.random_integers(0, len(orgs[0]) - 1)
      idx2 = idx1
      while idx1 == idx2:
         idx2 = np.random.random_integers(0, len(orgs[0]) - 1)
      
      idx = 0
      if idx1 > idx2:
         idx = 1
         a = idx1
         idx1 = idx2
         idx2 = a

      if isinstance(orgs[0], np.ndarray):
         return np.append(np.append(orgs[idx][:idx1],
                                    orgs[1-idx][idx1:idx2],axis=0),
                          orgs[idx][idx2:],
                          axis=0)
      elif isinstance(orgs[0], TernaryString):
         ret = TernartString(orgs[idx].base, orgs[idx].known, orgs[idx].length)
         ret[idx1:idx2] = orgs[1-idx]

class IntermediateCrosser(Crosser):
   """
      Cross multiple organisms by averaging their values component-wise.
      
      Normally used by evolution strategies.
   """
   def __call__(self, orgs, prob=1.0):
      if isinstance(orgs[0], list):
         return [np.array(x).sum(axis=0) / len(orgs) for x in zip(*orgs)]
      return np.array(orgs).sum(axis=0) / len(orgs)
      
      
class DominantCrosser(Crosser):
   """
      Cross multiple organisms using generalized uniform crossover.
   """
   def __call__(self, orgs, prob=1.0):
      if isinstance(orgs[0], list):
         return [self.__call__(x, prob) for x in zip(*orgs)]
      x = []
      for i in xrange(len(orgs[0])):
         idx = np.random.randint(0, len(orgs))
         x.append(orgs[idx][i])
      return np.array(x)
      
      
class DifferentialCrosser(Crosser):
   def __init__(self, learningRate, crossoverProb):
      self.CR = crossoverProb
      self.F = learningRate

   def __call__(self, orgs, prob=1.0):
      y, a, b, c = orgs
      d = np.random.randint(0, len(y))
      idx2 = 0
      for yi in y:
         r = np.random.random_sample()
         if idx2 == d or r < self.CR:
            y[idx2] = a[idx2] + self.F * (b[idx2] - c[idx2]) 
            idx2 += 1
      return y


class Crossover(PopulationDistribution):
   """
      Performs recombination using a crossover policy given by
      a :class:`Crosser` instance.
   
      Config parameters
      
      * crossoverProb -- A probability that is sampled; if a bernouli test
                         on this value fails, then no crossover is performed,
                         and the first selected organism is chosen
      * order -- The number of organisms to crossover at each step
      * crosser -- A :class:`Crosser` to perform the crossover
      
   """
   config = Config(crossoverProb=1.0, # probability of performing crossover
                   order=2, # number of solutions to crossover
                   crosser=UniformCrosser, # descendant of Crosser
                   history=MultiStepMarkovHistory) # num steps must match order
   
   def __init__(self, **kwargs):
      super(Crossover, self).__init__(**kwargs)
      self.dual = (hasattr(self.config.crosser, 'dual')
                   and self.config.crosser.dual)
   
   def compatible(self, history):
      return (hasattr(history, 'populations') and hasattr(history, 'order')
              and history.order() == self.config.order)
   
   def batch(self, popSize):
      if self.dual:
         psize = popSize / 4
      else:
         psize = popSize
      crossoverProb = self.config.crossoverProb
      if crossoverProb < 1e-16:
         return [x for x,s in self.history.populations[0]]
      self.crosser = self.config.crosser(self.config)
      pops = self.history.populations
      newpop = [self.crosser([org for org, s in orgs], crossoverProb)
                for orgs in zip(*pops)]
      if self.dual:
         pop = []
         for x,y in newpop:
            pop.append(x)
            pop.append(y)
         newpop = pop
      return newpop


class Mutation(PopulationDistribution):
   """
      Base class for mutation operators. Provides a method ``mutate`` that performs mutation on a single member of a population.
      
   """
   config = Config(history=MarkovHistory)
   
   def needsScores(self):
      return False
   
   def mutate(self, x):
      """
         Perform mutation on a single organism.
         
         :param x: The organism to mutate.
         :type x: varied
         :returns: The mutated organism.
      """
      return x

   def batch(self, popSize):
      pop = []
      for i, val in enumerate(self.history.lastPopulation()):
         x,s = val
         self.idx = i
         z = self.mutate(x)
         cnt = 0
         while not self.config.space.in_bounds(z):
            cnt += 1
            if cnt > 10000:
               raise RuntimeError("Rejection sampling in mutation failed"
                                  " to generate a point in the space after"
                                  " 10,000 attempts.")
            z = self.mutate(x)
         pop.append(z)
      return pop

   def compatible(self, historyClass):
      return hasattr(historyClass, 'lastPopulation')
 

class Poisson(Mutation):
   """
      Poisson mutation. Randomly mutates a random selection of components
      of a real vector.
      
      Config parameters
      * lmbda -- The percentage of the space to jump, in [0,1]; poisson lambda is
                 computed by multiplying this by the space width in each dim
      * p -- The probability of mutating each component; defaults to 1.0
      
  """
   config = Config(lmbda=.05,
                   p=1.0)
   
   def __init__(self, **kwargs):
      super(Poisson, self).__init__(**kwargs)
      if self.config.space.type != np.ndarray:
         raise ValueError("Space for Gaussian must have type numpy.ndarray")
         
   def lmbda(self):
      sp = self.config.space
      return self.config.lmbda * sp.scale
         
   def mutate(self, x):
      p = np.random.binomial(1, self.config.p, len(x))
      ret = x + p * np.random.poisson(1.0, len(x)) * self.lmbda()
      ret = ret.astype(np.int64)
      if not self.config.space.in_bounds(ret):
         lower = self.config.space.lower
         upper = self.config.space.upper
         ret = np.maximum(np.minimum(ret, upper),lower)
      return ret

class Gaussian(Mutation):
   """
      Gaussian mutation. Randomly mutates a random selection of components
      of a real vector.
      
      Config parameters
      * sd -- The standard deviation to use. 
      * p -- The probability of mutating each component; defaults to 1.0
      
  """
   config = Config(sd=0.01,
                   p=1.0)
   
   def __init__(self, **kwargs):
      super(Gaussian, self).__init__(**kwargs)
      if self.config.space.type != np.ndarray:
         raise ValueError("Space for Gaussian must have type numpy.ndarray")
         
   def sd(self):
      return self.config.sd * self.config.space.scale
         
   def mutate(self, x):
      p = np.random.binomial(1, self.config.p, len(x))
      ret = x + p * np.random.randn(len(x)) * self.sd()
      if not self.config.space.in_bounds(ret):
         scale = self.config.space.scale
         center = self.config.space.center
         ret = np.maximum(np.minimum(ret, scale+center),center-scale)
      return ret
      
   def density(self, center, point):
      sd = self.sd()
      return np.exp(-(1. / (2. * sd * sd)) *
                    ((point - center) ** 2).sum()) / sd / np.sqrt(2*np.pi)


class DecayedGaussian(Gaussian):
   """
      Gaussian mutation with a variance that decreases over time according to fixed formula.
      
      Config parameters
      * sd -- A scale factor on the standard deviation.
      * decay -- A multiplicative decay factor.
      * decayExp -- An exponential decay factor.
      
      Standard deviation is determined by::
      
         sd = sd0 * exp(-(generation * varDecay) ** varExp)
      
   """
   config = Config(decay=1.0,
                   decayExp=1.0)
   
   def sd(self):
      n = self.history.updates
      decay = self.config.decay
      decayExp = self.config.decayExp
      return self.config.sd * np.exp(-(n * decay) ** decayExp) 


class AreaSensitivePoisson(Poisson):
   """
      Poisson mutation with a lambda that depends on the size of a
      partition of the space (see :class:`pyec.util.partitions.Partition`).
      
      Config parameters:
      * decay -- A function ``decay(n,config)`` to compute the multiplier
                 that controls the rate of decrease in standard deviation.
                 Faster decay causes faster convergence, but may miss the
                 optimum. Default is ``((1/generations))``
      
      Lambda is determined by::
      
         lmbda = .5 * ((upper-lower) * decay(n)
         
      where ``dim`` is the dimension of the space (``config.dim``) and 
      ``area`` is the volume of the partition region for the object being mutated.
      
   """
   config = Config(jogo2012=False,
                   decay=lambda n,cfg: (1./n)) # **(1./cfg.space.dim))
   
   def __init__(self, **kwargs):
      super(AreaSensitivePoisson, self).__init__(**kwargs)
   
   def compatible(self, history):
      # N.B. we use a Markov history, but assume we are passed the partition nodes
      return hasattr(history, 'lastPopulation')
   
   def mutate(self, x):
      """Apply area-sensitive poisson mutation.
         Assume that the passed value is a :class:`Point` object
         
         :param x: The point to mutate
         :type x: :class:`Point`
         :returns: A numpy.ndarray mutation the ``point`` property of ``x``.
      
      """
      lower = self.config.space.lower
      upper = self.config.space.upper
      
      y, node = x
      y = y.point
      area = node
      lower, upper = area.bounds.extent()
      lmbda = .5 * (upper - lower)
      lmbda *= self.config.decay(self.history.updates, self.config)
      
      ret = y + np.random.poisson(1.0, len(y)) * lmbda
      if not self.config.space.in_bounds(ret):
         ret = np.maximum(np.minimum(ret, upper),lower)
      
      return ret

   def density(self, info, point):
      scale = self.config.space.scale
      center, area = info
      sd = self.config.sd * scale * (area ** (1./len(point)))
      
      return exp(-(1./ (2 * sd * sd)) *
                 ((point - center) ** 2).sum()) / sd / np.sqrt(2*np.pi) 

class AreaSensitiveGaussian(Gaussian):
   """
      Gaussian mutation with a variance that depends on the size of a
      partition of the space (see :class:`pyec.util.partitions.Partition`).
      
      Config parameters:
      * jogo2012 -- Whether to use the methods from Lockett & Miikkulainen, 2012
      * decay -- A function ``decay(n,config)`` to compute the multiplier
                 that controls the rate of decrease in standard deviation.
                 Faster decay causes faster convergence, but may miss the
                 optimum. Default is ``((1/generations))``
      
      Standard deviation is determined by::
      
         sd = .5 * ((upper-lower) * decay(n)
         
      where ``dim`` is the dimension of the space (``config.dim``) and 
      ``area`` is the volume of the partition region for the object being mutated.
      
   """
   config = Config(jogo2012=False,
                   decay=lambda n,cfg: (1./n)) # **(1./cfg.space.dim))
   
   def __init__(self, **kwargs):
      super(AreaSensitiveGaussian, self).__init__(**kwargs)
      if self.config.jogo2012:
         self.config.decay = lambda n,cfg: (1/n)**.5
   
   def compatible(self, history):
      # N.B. we use a Markov history, but assume we are passed the partition nodes
      return hasattr(history, 'lastPopulation')
   
   def mutate(self, x):
      """Apply area-sensitive gaussian mutation.
         Assume that the passed value is a :class:`Point` object
         
         :param x: The point to mutate
         :type x: :class:`Point`
         :returns: A numpy.ndarray mutation the ``point`` property of ``x``.
      
      """
      center = self.config.space.center
      scale = self.config.space.scale
      
      y, node = x
      y = y.point
      if self.config.jogo2012:
         area = node.area
         # sd = 2 * scale / (-log(area))
         if area < 0.0: 
            area = 1e-100
         sd = self.config.sd * scale * (area ** (1./len(y)))
         sd *= self.config.decay(self.history.updates, self.config)
      else:
         area = node
         lower, upper = area.bounds.extent()
         sd = .5 * (upper - lower)
         sd = np.minimum(sd, scale)
         sd *= self.config.decay(self.history.updates, self.config)
      
      ret = y + np.random.randn(len(y)) * sd
      if not self.config.space.in_bounds(ret):
         ret = np.maximum(np.minimum(ret, scale+center),center-scale)
      
      return ret

   def density(self, info, point):
      scale = self.config.space.scale
      center, area = info
      sd = self.config.sd * scale * (area ** (1./len(point)))
      
      return exp(-(1./ (2 * sd * sd)) *
                 ((point - center) ** 2).sum()) / sd / np.sqrt(2*np.pi) 


class UniformArea(Mutation):
   """
      Uniform mutation within a partition region. The region is determined by the organism being mutated.
      
      :param config: The configuration parameters.
      :type config: :class:`Config`
   """
   def mutate(self, x):
      """Apply uniform area mutation.
         Assume that the passed value is a :class:`Point` object
         
         :param x: The point to mutate
         :type x: :class:`Point`
         :returns: A numpy.ndarray mutation the ``point`` property of ``x``.
      
      """
      y, area = x
      return area.bounds.random()


class Cauchy(Mutation):
   """
      Cauchy mutation.
      
      Config parameters
      * sd -- The width parameter of the Cauchy (\pi)
      * p -- The probability of mutating each component, 1.0 by default
      
   """
   config = Config(sd=1.0,
                   p = 1.0)

   def __init__(self, config):
      super(Cauchy, self).__init__(config)

   def mutate(self, x):
      p = np.random.binomial(1, self.config.p, len(x))
      return x + self.config.sd * p * np.random.standard_cauchy(len(x))


class Bernoulli(Mutation):
   """
      Bernoulli mutation on a :class:`TernaryString` object. Randomly flips bits on a fully specified binary string.
      
      Config parameters
      * p -- The probability of flipping each bit.   
 
   """
   config = Config(p=0.01)

   def __init__(self, **kwargs):
      super(Bernoulli, self).__init__(**kwargs)
      if (not inspect.isclass(self.config.space.type) or
          not issubclass(self.config.space.type, TernaryString)):
         raise ValueError("Space must produce points descending "
                          "from TernaryString")
      p = self.config.p
      if ((isinstance(p, np.ndarray) and (p > 1.0).any()) or
          (not isinstance(p, np.ndarray) and p > 1.0)):
         raise ValueError("bit flip probability > 1.0: " + str(p))

   def p(self):
      return self.config.p

   def mutate(self, x):
      """This method uses an iterative algorithm to flip bits in
      a TernaryString with a given probability. The algorithm essentially
      uses the binary representation of the bit-flipping probability in order
      to convert random byte sampling (``p=.5``) to account for any probability,
      with resolution of ``1e-16``.
      
      """
      flipped = TernaryString.bernoulli(self.p(), self.config.space.dim)
      base = x.base & ~flipped.base | ~x.base & flipped.base
      known = x.known & flipped.known
      return TernaryString(base, known, self.config.space.dim)


class DecayedBernoulli(Bernoulli):
   """
      Like :class:`DecayedGaussian`, but with a Bernoulli distribution instead of a Gaussian; the bit flipping probability decays to zero.
      
      Config parameters
      * sd -- A scale factor on the standard deviation.
      * decay -- A multiplicative decay factor.
      * decayExp -- An exponential decay factor.
      
      Bit flip prob is determined by::
      
         p = p0 * exp(-(generation * varDecay) ** varExp)
      
   """
   config = Config(decay=1.0,
                   decayExp=1.0)
   
   def p(self):
      n = self.history.updates
      decay = self.config.decay
      decayExp = self.config.decayExp
      return self.config.p * np.exp(-(n * decay) ** decayExp) 


class AreaSensitiveBernoulli(Bernoulli):
   """
      Like :class:`AreaSensitiveGaussian` but with a Bernoulli.
      
      Because a bit space is finite, there is no need to decay, but
      we still want to make exiting the area less likely to promote
      exploration within the selected area.
      
      Config parameters
      * p -- The bit-flipping prob
      
   """
   _p = None
   
   def p(self):
      return self._p
   
   def density(self, x, z):
      y = x[0].point
      area = x[1]
      bf = self._p
      prod = 1.0
      for i in xrange(self.config.dim):
         if z[i] != y[i]:
            prod *= bf
         else:
            prod *= (1. - bf)
      return prod
      
   def mutate(self, x):
      y = x[0].point
      area = x[1]
      #lower, upper = area.bounds.extent()
      #diff = TernaryString(lower.base ^ upper.base,-1L,lower.length).toArray()
      #logArea = self.config.space.dim - diff.sum()
      #self._p = (diff * self.config.p  +
      #           (1.0 - diff) * self.config.p ** logArea)
      self._p = self.config.p
      return super(AreaSensitiveBernoulli, self).mutate(y)

  
class EndogeneousGaussian(Mutation):
   """
      Gaussian mutation for evolution strategies, where the adaptive mutation
      parameters are encoded with the organism.
    
      NEEDS A SPACE TO HANDLE PULLING OUT THE SIGNIFICANT DIMENSIONS
      
      Config params
      
      * sd -- The standard deviation to start with.
      * baseDim -- The portion of the genotype encoding the point, i.e. the
                   number of dimensions being searched.
      
   """
   config = Config(baseDim=None)
   
   def __init__(self, **kwargs):
      super(EndogeneousGaussian, self).__init__(**kwargs)
      if (not isinstance(self.config.space, EndogeneousProduct) or
          len(self.config.space.spaces) < 2 or
          self.config.space.spaces[0].type != np.ndarray or
          self.config.space.spaces[1].type != np.ndarray):
         raise ValueError("Endogeneous Gaussian expects EndogeneousProduct "
                          "space with two components, each of "
                          "type numpy.ndarray")
      self.sd = self.config.sd
      self.tau = self.sd / np.sqrt(2*np.sqrt(self.config.populationSize))
      self.tau0 = self.sd / np.sqrt(2*self.config.populationSize)
   
   def mutate(self, x):
      y = x[0]
      sig = x[1]
      rnd = self.tau * np.random.randn(len(sig))
            
      sig2 = np.exp(self.tau0 * np.random.randn(1)) * sig * np.exp(rnd) 
      z = y + sig2 * np.random.randn(len(y))
      
      return [z,sig2] 


class CorrelatedEndogeneousGaussian(Mutation):
   """
      Implements an old (1996) version of mutation for the 
      Correlated Matrix Adaption method of Hansen and Ostermeier.
      
      This method is obsolete and is very different from the modern version.
      
      NEEDS A SPACE TO HANDLE PULLING OUT THE SIGNIFICANT DIMENSIONS, see
      :class:`EndogeneousProduct`
      
      1996 paper says to set cmaCumulation to $\frac{1}{\sqrt{n}}$,
      cmaDamping to $\frac{1}{n}$, cmaCorrelation to $\frac{2}{n^2}$.
      Default assumes will be run 1000 generations, n=1000
      
      Config Parameters:
      
      * sd
      * cmaCumulation
      * cmaDamping
      * cmaCorrelation
      
   """
   config = Config(baseDim=None,
                   sd=1.0,
                   cmaCumulation=1./np.sqrt(1000.), # 1/sqrt(n)
                   cmaDamping=.001, # 1/n
                   cmaCorrelation=0.000002) # 2/n^2
   
   def __init__(self, **kwargs):
      super(CorrelatedEndogeneousGaussian,self).__init__(**kwargs)
      if (not isinstance(self.config.space, EndogeneousProduct) or
          len(self.config.space.spaces) < 5 or
          self.config.space.spaces[0].type != np.ndarray or
          self.config.space.spaces[1].type != np.ndarray or
          (self.config.space.spaces[1].dim !=
           (self.config.space.spaces[0].dim *
            (self.config.space.spaces[0].dim + 1) / 2)) or
          self.config.space.spaces[2].type != np.ndarray or
          self.config.space.spaces[2].dim != self.config.space.spaces[0].dim or
          self.config.space.spaces[3].type != np.ndarray or
          self.config.space.spaces[3].dim != self.config.space.spaces[0].dim or
          self.config.space.spaces[4].type != np.ndarray or
          self.config.space.spaces[4].dim != self.config.space.spaces[0].dim):
         raise ValueError("Correlated Endogeneous Gaussian expects "
                          "EndogeneousProduct "
                          "space with five components, each of "
                          "type numpy.ndarray; if the dimension of "
                          "the first is N, the dimension of the second "
                          "must be N(N+1)/2, and the third through fifth are "
                          "both of dimension N")
      
      self.sd = self.config.sd
      self.dim = self.config.space.spaces[0].dim
      self.center = self.config.space.spaces[0].center
      self.scale = self.config.space.spaces[0].scale
      self.cumulation = self.config.cmaCumulation
      self.cu = np.sqrt(self.cumulation * (2 - self.cumulation))
      self.beta = self.config.cmaDamping
      self.chi = (np.sqrt(self.config.populationSize) 
                  * (1. - (1./(4*self.config.populationSize)) 
                  + (1./(21 * (self.config.populationSize ** 2)))))
      self.correlation = self.config.cmaCorrelation
      self.numEndogeneous = self.dim * (self.dim - 1) / 2 + 3*self.dim
    
   def unpack(self, sig):
      """take a N(N+1)/2 array and make a N x N matrix"""
      idx = 0
      mat = []
      for i in xrange(self.dim):
         row = []
         for j in xrange(self.dim):
            if j < i:
                row.append(0)
            else:
                row.append(sig[idx])
                idx += 1
         mat.append(row)
      return np.array(mat)
   
   def pack(self, mat):
      """take a N x N matrix and make a N(N+1)/2 array"""
      idx = 0
      sig = []
      for i in xrange(self.dim):
         for j in xrange(self.dim):
            if j >= i:
                sig.append(mat[i][j])
      return np.array(sig) 
   
   def mutate(self, x):
      y = x[0]
      sig = x[1]
      cum = x[2]
      delta = x[3]
      deltaCum = x[4]
      rot = self.unpack(sig)
      corr = np.dot(rot, rot)
            
      deltaRot = rot / np.outer(rot.sum(axis=1), np.ones(self.dim))
      deltaCum2 = (1 - self.cumulation) * deltaCum \
       + self.cu * np.dot(deltaRot, np.random.randn(self.dim))
      delta2 = delta * np.exp(self.beta * (np.sqrt(deltaCum2 ** 2) - self.chi))
            
      cum2 = (1 - self.cumulation) * cum \
          + self.cu * np.dot(rot, np.random.randn(self.dim))
      corr2 = (1 - self.correlation) * corr \
          + self.correlation * np.outer(cum2, cum2)
      rot2 = np.linalg.cholesky(corr2)
      sig2 = self.pack(rot2)
      
      z = y + delta * np.dot(rot, np.random.randn(len(y)))
      
      self.sd = np.average(sig)      
      return [z, sig2, cum2, delta2, deltaCum2]
