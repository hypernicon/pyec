"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from numpy import *
from pyec.util.TernaryString import TernaryString
import copy, binascii, struct
from pyec.distribution.basic import PopulationDistribution
from pyec.util.partitions import Segment, Partition

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
      if random.random_sample() > prob:
         return orgs[0]
      if isinstance(orgs[0], ndarray):
         rnd = random.random_sample(len(orgs[0])).round()
         return rnd * orgs[0] + (1 - rnd) * orgs[1]
      elif isinstance(orgs[0], TernaryString):
         rnd = random.bytes(len(str(orgs[0])) * 8)
         rnd = long(binascii.hexlify(rnd), 16)
         base = rnd & orgs[0].base | ~rnd & orgs[1].base
         known = rnd & orgs[0].known | ~rnd & orgs[1].known
         return TernaryString(base, known)
      else:
         return None

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
      if random.random_sample() > prob:
         return orgs[0], orgs[1]
      idx = random.random_integers(0, len(orgs[0]) - 1)
      return append(orgs[0][:idx], orgs[1][idx:], axis=0), append(orgs[1][:idx], orgs[0][idx:], axis=0)

class OnePointCrosser(Crosser):
   """
      Cross two organisms at one point and return only one crossed version.
      
      Unlike :class:`OnePointDualCrosser`, this discards the second variant.
   
      See <http://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)#One-point_crossover>.
   """
   def __call__(self, orgs, prob=1.0):
      if random.random_sample() > prob:
         return orgs[0], orgs[1]
      idx = random.random_integers(0, len(orgs[0]) - 1)
      return append(orgs[0][:idx], orgs[1][idx:], axis=0)
      
class TwoPointCrosser(Crosser):
   """
      Cross two organism by wrapping them into a circle, picking two indices, and combining the two rings. 
      
      See <http://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)#Two-point_crossover>.
   """
   def __call__(self, orgs, prob=1.0):
      if random.random_sample() > prob:
         return orgs[0]
      idx1 = random.random_integers(0, len(orgs[0]) - 1)
      idx2 = idx1
      while idx1 == idx2:
         idx2 = random.random_integers(0, len(orgs[0]) - 1)
    
      if idx1 > idx2:
         a = idx1
         idx1 = idx2
         idx2 = a

      return append(append(orgs[0][:idx1], orgs[1][idx1:idx2],axis=0), orgs[0][idx2:], axis=0)


class IntermediateCrosser(Crosser):
   """
      Cross multiple organisms by averaging their values component-wise.
      
      Normally used by evolution strategies.
   """
   def __call__(self, orgs, prob=1.0):
      return array(orgs).sum(axis=0) / len(orgs)
      
class DominantCrosser(Crosser):
   """
      Cross multiple organisms using generalized uniform crossover.
   """
   def __call__(self, orgs, prob=1.0):
      x = []
      for i in xrange(len(orgs[0])):
         idx = random.randint(0, len(orgs) - 1)
         x.append(orgs[idx][i])
      return array(x)
      
class DifferentialCrosser(Crosser):
   def __init__(self, learningRate, crossoverProb):
      self.CR = crossoverProb
      self.F = learningRate

   def __call__(self, orgs, prob=1.0):
      y, a, b, c = orgs
      d = random.randint(0, len(y))
      idx2 = 0
      for yi in y:
         r = random.random_sample()
         if idx2 == d or r < self.CR:
            y[idx2] = a[idx2] + self.F * (b[idx2] - c[idx2]) 
            idx2 += 1
      return y

class Mutation(PopulationDistribution):
   """
      Base class for mutation operators. Provides a method ``mutate`` that performs mutation on a single member of a population.
      
      :param config: The configuration parameters.
      :type config: :class:`Config`
   """
   def __init__(self, config):
      super(Mutation, self).__init__(config)
      self.population = None
   
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
      for i, val in enumerate(self.population):
         x,s = val
         self.idx = i
         z = self.mutate(x)
         if isinstance(z, ndarray) and self.config.bounded:
            while not self.config.in_bounds(z):
               z = self.mutate(x)
         pop.append(z)
      return pop

   def compatible(self, historyClass):
      return hasattr(historyClass, 'lastPopulation')
   
   def update(self, history):
      self.population = history.lastPopulation()

class Crossover(Mutation):
   """
      Performs recombination using a crossover policy given by
      a :class:`Crosser` instance.
   
      :param selector: A selector to use for choosing the "mother" organism(s).
      :type selector: :class:`Selector`
      :param crossFunc: A crossover policy.
      :type config: :class:`Crosser`
      :param order: The number of parents; default is 2.
      :type order: int
   """
   def __init__(self, selector, crossFunc, order=2):
      super(Crossover, self).__init__(selector.config)
      self.selector = selector
      self.crosser = crossFunc
      self.order = order
      self.n = 1
      self.dual = hasattr(crossFunc, 'dual') and crossFunc.dual
   
   def mutate(self, x):
      raise Exception, "Operation not supported"   
            
   def batch(self, popSize):
      if self.dual:
         psize = popSize / 4
      else:
         psize = popSize
      crossoverProb = hasattr(self.config, 'crossoverProb') and self.config.crossoverProb or 1.0
      if crossoverProb < 1e-16:
         return [x for x,s in self.population1]
      if hasattr(self.config, 'passArea') and self.config.passArea:
         pops = [[x[0] for x,s in self.population1]]
         areas = [x[1].area for x,s in self.population1]
         crossoverProb *= array(areas) ** (1. / self.config.dim)
      else:
         pops = [[x for x,s in self.population1]]
         crossoverProb = crossoverProb * ones(psize) * sqrt(1./self.n)
      for i in xrange(self.order - 1):
         if hasattr(self.config, 'passArea') and self.config.passArea:
            pops.append([x[0] for x in self.selector.batch(psize)])
         else:
            pops.append(self.selector.batch(psize))
      pops.append(list(crossoverProb))
      newpop = [self.crosser(orgs[:-1], orgs[-1]) for orgs in zip(*pops)]
      if self.dual:
         pop = []
         for i in xrange(popSize/2):
            pop.append(pops[0][i])
         for x,y in newpop:
            pop.append(x)
            pop.append(y)
         newpop = pop
      if hasattr(self.config, 'passArea') and self.config.passArea:
         return zip(newpop, areas)
      else:
         return newpop
   
   def update(self, generation, population):
      self.n = generation
      self.population1 = population # copy.copy(population)
      # self.selector should already be updated, update is called to give us
      # the selected result
      self.selector.update(generation, population)

class Gaussian(Mutation):
   """
      Gaussian mutation. Randomly mutates a random selection of components
      of a real vector.
      
      Config parameters
      * stddev -- The standard deviation to use. 
      * mutationProb -- The probability of mutating each component; defaults to 1.0
      
      :param config: The configuration parameters.
      :type config: :class:`Config`
   """
   def __init__(self, config):
      super(Gaussian, self).__init__(config)
      self.sd = config.stddev
      
   def mutate(self, x):
      if hasattr(self.config, 'mutationProb'):
         p = random.binomial(1, self.config.mutationProb, len(x))
         return x + p * random.randn(len(x)) * self.sd
      return x + random.randn(len(x)) * self.sd
      
   def density(self, center, point):
      return exp(-(1. / (2. * self.sd * self.sd)) * ((point - center) ** 2).sum()) / self.sd / sqrt(2*pi)

class DecayedGaussian(Gaussian):
   """
      Gaussian mutation with a variance that decreases over time according to fixed formula.
      
      Config parameters
      * varInit -- A scale factor on the standard deviation.
      * varDecay -- A multiplicative decay factor.
      * varExp -- An exponential decay factor.
      
      Standard deviation is determined by::
      
         sd = varInit * exp(-(generation * varDecay) ** varExp)
      
      :param config: The configuration parameters.
      :type config: :class:`Config`
   """
   def update(self, generation, population):
      super(DecayedGaussian, self).update(generation, population)
      self.sd = self.config.varInit * exp(-(generation * self.config.varDecay) ** self.config.varExp) 

class AreaSensitiveGaussian(Mutation):
   """
      Gaussian mutation with a variance that depends on the size of a
      partition of the space (see :class:`pyec.util.partitions.Partition`).
      
      Config parameters:
      * varInit -- Scaling factor for the standard deviation; defaults to 1.0.
      * scale -- The scale of the space (half the width).
      * spaceScale -- A scaling factor for the space size.
      
      Standard deviation is determined by::
      
         sd = (varInit * scale * (area ** (1/dim)) / sqrt(generation)
         
      where ``dim`` is the dimension of the space (``config.dim``) and 
      ``area`` is the volume of the partition region for the object being mutated.
      
      :param config: The configuration parameters.
      :type config: :class:`Config`
   """
   sdavg = 0
   sdcnt = 0
   sdmin = 1e100
   gen = 1
   def mutate(self, x):
      center = self.config.center
      scale = self.config.scale
      if self.config.bounded and hasattr(self.config.in_bounds, 'extent'):
         center, scale = self.config.in_bounds.extent()
      else:
         scale = self.config.spaceScale
      y = x[0]
      if hasattr(self.config, 'jogo2012') and self.config.jogo2012:
         area = x[1].area
         # sd = 2 * scale / (-log(area))
         if area < 0.0: 
            area = 1e-100
         sd = self.config.varInit * scale * (area ** (1./len(y)))
         sd /= self.gen ** .5
      else:
         area = x[1]
         sd = .5 * (area.bounds[1] - area.bounds[0])
         sd *= (1./log(self.gen)) ** (1./self.config.dim)
      
      self.sdavg = (self.sdavg * self.sdcnt + sd) / (self.sdcnt + 1.0)
      self.sdcnt += 1
      self.sdmin = (average(sd) < self.sdmin) and average(sd) or self.sdmin
      ret = y + random.randn(len(y)) * sd
      if self.config.bounded and not self.config.in_bounds(ret):
         ret = maximum(minimum(ret, scale+center),center-scale)
         #y + random.randn(1) * sd
      return ret

   def density(self, info, point):
      scale = self.config.scale
      if self.config.bounded and hasattr(self.config.in_bounds, 'extent'):
         dummy, scale = self.config.in_bounds.extent()
      center, area = info
      sd = self.config.varInit * scale * (area ** (1./len(point)))
      
      return exp(-(1./ (2 * sd * sd)) * ((point - center) ** 2).sum()) / sd / sqrt(2 * pi) 

   def update(self, generation, population):
      super(AreaSensitiveGaussian, self).update(generation, population)
      self.sdcnt = 0
      self.sdavg = 0.0
      self.sd = self.sdmin
      self.sdmin = 1e100
      self.gen = generation

class UniformArea(Mutation):
   """
      Uniform mutation within a partition region. The region is determined by the organism being mutated.
      
      :param config: The configuration parameters.
      :type config: :class:`Config`
   """
   segment = None
   sd = 1.0
   def mutate(self, x):
      node, lower, upper = Partition.objects.traversePoint(self.segment.id, x, self.config)
      self.sd = (node.area < self.sd) and node.area or self.sd
      r = random.random_sample(len(x))
      return lower + r * (upper - lower)
      
   def update(self, n, population):
      super(UniformArea, self).update(n, population)
      if self.segment is None:
         self.segment = Segment.objects.get(name=self.config.segment)

class Cauchy(Mutation):
   """
      Cauchy mutation.
      
      Config parameters
      * stddev -- The width parameter of the Cauchy (\pi)
      
      :param config: The configuration parameters.
      :type config: :class:`Config`
   """
   def __init__(self, config):
      super(Cauchy, self).__init__(config)
      self.sd = config.stddev

   def mutate(self, x):
      return x + random.standard_cauchy(len(x))
      
class OnePointCauchy(Mutation):
   """
      Mutates just one random component of an organism with a Cauchy distribution.
      
      Config parameter
      * stddev -- The width parameter of the Cauchy (\pi)
      * mutationProb -- The probability of applying the mutation.
      
      :param config: The configuration parameters.
      :type config: :class:`Config`
   """
   def __init__(self, config):
      super(OnePointCauchy, self).__init__(config)
      self.sd = config.stddev

   def mutate(self, x):
      if self.idx >= len(self.population) / 2 \
       and random.random_sample() < self.config.mutationProb:
         idx = random.randint(0, len(x))
         x[idx] += .3 * random.standard_cauchy()
      return x

class DecayedCauchy(Cauchy):
   """
      Like :class:`DecayedGaussian`, except with a Cauchy distribution.
     
      :param config: The configuration parameters.
      :type config: :class:`Config`
   """
   def update(self, generation, population):
      super(DecayedGaussian, self).update(generation, population)
      self.sd = self.config.varInit * exp(-(generation * self.config.varDecay) ** self.config.varExp) 
      
class Bernoulli(Mutation):
   """
      Bernoulli mutation on a :class:`TernaryString` object. Randomly flips bits on a fully specified binary string.
      
      Config parameters
      * bitFlipProbs -- The probability of flipping a bit. 
   
      :param config: The configuration parameters.
      :type config: :class:`Config`
   """

   def __init__(self, config):
      super(Bernoulli, self).__init__(config)
      self.bitFlipProbs = .5
      if hasattr(config, 'bitFlipProbs'):
         self.bitFlipProbs = config.bitFlipProbs

   def mutate(self, x):
      numBytes = int(ceil(self.config.dim / 8.0))
      numFull  = self.config.dim / 8
      initial = ''
      if numBytes != numFull:
         extra = self.config.dim % 8
         initMask = 0
         for i in xrange(extra):
            initMask <<= 1
            initMask |= 1
         initial = struct.pack('B',initMask)
      
      start = (1L << (self.config.dim + 1)) - 1
      p = self.bitFlipProbs
      if (isinstance(p, ndarray) and (p > 1.0).any()) or (not isinstance(p, ndarray) and p > 1.0): raise Exception, "bit flip probability > 1.0: " + str(p)
      base = 0L
      active = TernaryString(x.known,x.known)
      while (isinstance(p, ndarray) and (p > 1e-16).any()) or (not isinstance(p, ndarray) and p > 1e-16):
         #print p
         reps = minimum(100, -floor(log2(p)))
         #print p, reps
         q = 2.0 ** -reps
         next = start
         activeReps = TernaryString(active.base, active.known)
         if isinstance(p, ndarray):
            for j, pj in enumerate(p):
               if pj < 1e-16:
                  active[j] = False
            #print "active: ", active.toArray(p.size)
            for i in xrange(int(max(reps))):
               for j,r in enumerate(reps):
                  if i >= r:
                     activeReps[j] = False
               #print "activeReps: ", activeReps.toArray(p.size)
               next &= activeReps.base & long(binascii.hexlify(random.bytes(numBytes)), 16)
         else:
            for i in xrange(int(reps)):
               next &= long(binascii.hexlify(random.bytes(numBytes)), 16) 
         base |= next & active.base
         p = (p - q) / (1.0 - q)
            
      
      base = x.base & ~base | ~x.base & base
      known = x.known# long(binascii.hexlify(initial + '\xff'*numFull), 16)
      return TernaryString(base, known)



class DecayedBernoulli(Bernoulli):
   """
      Like :class:`DecayedGaussian`, but with a Bernoulli distribution instead of a Gaussian; the bit flipping probability decays to zero.
      
      :param config: The configuration parameters.
      :type config: :class:`Config`
   """
   def update(self, generation, population):
      super(DecayedBernoulli, self).update(generation, population)
      self.bitFlipProbs = self.config.varInit * exp(-(generation * self.config.varDecay) ** self.config.varExp) 

class AreaSensitiveBernoulli(Bernoulli):
   """
      Like :class:`AreaSensitiveGaussian` but with a Bernoulli. The bit flipping prob decays with the logarithm of the area of a partition region, i.e.::
      
         bf = self.config.varInit / (-log(area))
      
      where ``area`` is the volume of a partition region.
      
      Config parameters
      * varInit -- The scaling factor for the bit flipping probability.
      
      :param config: The configuration parameters.
      :type config: :class:`Config`
   """
   bfavg = 0
   bfcnt = 0
   bfmin = 1e100
   
   def density(self, x, z):
      y = x[0]
      area = x[1]
      bf = self.config.varInit / (-log(area))
      prod = 1.0
      for i in xrange(self.config.dim):
         if z[i] != y[i]:
            prod *= bf
         else:
            prod *= (1. - bf)
      return prod
      
   def mutate(self, x):
      y = x[0]
      area = x[1]
      bf = self.config.varInit / (-log(area))
      #bf = .2 ** (-log(area)/self.config.rawdim)
      self.bfavg = (self.bfavg * self.bfcnt + bf) / (self.bfcnt + 1.0)
      self.bfcnt += 1
      self.bfmin = (bf < self.bfmin) and bf or self.bfmin
      self.bitFlipProbs = minimum(bf, 1.0)
      ret = super(AreaSensitiveBernoulli, self).mutate(y)
      #print "bit flip prob: ", self.bitFlipProbs, area
      #print "percent flipped: ", abs(y.toArray(self.config.dim) - ret.toArray(self.config.dim)).sum() / self.config.dim
      return ret

   def update(self, generation, population):
      super(AreaSensitiveBernoulli, self).update(generation, population)
      self.bitFlipProbs = self.bfavg
      self.bfmin = 1e100
      self.bfcnt = 0
      self.bfavg = 0.0

class UniformAreaBernoulli(Bernoulli):
   """
      Like :class:`UniformArea`, but in a binary space. Flips the bits on an organism so that after the mutation, the object remains in the same partition region, and is uniformly chosen within the partition region. See :class:`pyec.util.partitions.Partition`.
      
      Config params
      * varInit -- Scaling factor for the bit flipping probability.
      
      :param config: The configuration parameters.
      :type config: :class:`Config`
   """
   segment = None
   bitFlipProbs = 1.0
   def mutate(self, x):
      node, lower, upper = Partition.objects.traversePoint(self.segment.id, x[0], self.config)
      #print node.id, lower, upper
      #bitFlipProbs = (upper - lower) * .5
      self.bitFlipProbs = maximum(((upper - lower) > .75) * .5 * self.config.varInit,self.config.varInit / (-log(x[1]))) 
      #print self.bitFlipProbs
      self.sd = .1/-log(x[1])
      ret = super(UniformAreaBernoulli, self).mutate(x[0])
      #print "x: ", x[0]
      #print "y: ", ret
      return ret
      
   def update(self, n, population):
      super(UniformAreaBernoulli, self).update(n, population)
      if self.segment is None:
         self.segment = Segment.objects.get(name=self.config.segment)      

      
class EndogeneousGaussian(Mutation):
   """
      Gaussian mutation for evolution strategies, where the adaptive mutation parameters are encoded with the organism.
      
      :param config: The configuration parameters.
      :type config: :class:`Config`
   """
   def __init__(self, config):
      super(EndogeneousGaussian, self).__init__(config)
      self.sd = config.varInit
      self.dim = config.dim
      self.scale = config.scale
      self.bounded = config.bounded
      self.tau = self.sd / sqrt(2*sqrt(config.populationSize))
      self.tau0 = self.sd / sqrt(2*config.populationSize)
      self.numEndogeneous = self.dim
   
   def mutate(self, x):
      z = 2 * self.scale * ones(self.dim)
      while (abs(z) > self.scale).any():
         y = x[:self.dim]
         sig = x[self.dim:]
         rnd = self.tau * random.randn(len(sig))
            
         sig2 = exp(self.tau0 * random.randn(1)) * sig * exp(rnd) 
         z = y + sig2 * random.randn(len(y))
         if not self.bounded:
            break
      return append(z, sig2, axis=0) 
      
   

class CorrelatedEndogeneousGaussian(Mutation):
   """
      Implements an old (1996) version of mutation for the 
      Correlated Matrix Adaption method of Hansen and Ostermeier.
      
      This method is obsolete and is very different from the modern version.
   """
   def __init__(self, config):
      super(CorrelatedEndogeneousGaussian,self).__init__(config)
      self.sd = config.varInit
      self.dim = config.dim
      self.center = config.center
      self.scale = config.scale
      self.bounded = config.bounded
      self.cumulation = config.cmaCumulation
      self.cu = sqrt(self.cumulation * (2 - self.cumulation))
      self.beta = config.cmaDamping
      self.chi = sqrt(config.populationSize) \
       * (1. - (1./(4*config.populationSize)) \
          + (1./(21 * (config.populationSize ** 2))))
      self.correlation = config.cmaCorrelation
      self.numEndogeneous = self.dim * (self.dim - 1) / 2 + 3*self.dim
    
   def unpack(self, sig):
      """take a N(N-1)/2 array and make a N x N matrix"""
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
      return array(mat)
   
   def pack(self, mat):
      """take a N x N matrix and make a N(N-1)/2 array"""
      idx = 0
      sig = []
      for i in xrange(self.dim):
         for j in xrange(self.dim):
            if j >= i:
                sig.append(mat[i][j])
      return array(sig) 
   
   def mutate(self, x):
      z = 2 * self.scale * ones(self.dim) + self.center
      y = x[:self.dim]
      sig = x[self.dim:-3*self.dim]
      cum = x[-3*self.dim:-2*self.dim]
      delta = x[-2*self.dim:-self.dim]
      deltaCum = x[-self.dim:]
      rot = self.unpack(sig)
      corr = dot(rot, rot)
            
      deltaRot = rot / outer(rot.sum(axis=1), ones(self.dim))
      while (abs(z - self.center) > self.scale).any():
         deltaCum2 = (1 - self.cumulation) * deltaCum \
          + self.cu * dot(deltaRot, random.randn(self.dim))
         delta2 = delta * exp(self.beta * (sqrt(deltaCum2 ** 2) - self.chi))
            
         cum2 = (1 - self.cumulation) * cum \
          + self.cu * dot(rot, random.randn(self.dim))
         corr2 = (1 - self.correlation) * corr \
          + self.correlation * outer(cum2, cum2)
         rot2 = linalg.cholesky(corr2)
         sig2 = self.pack(rot2)
            
                        
         z = y + delta * dot(rot, random.randn(len(y)))
            
         if not self.bounded:
            break
      z0 = append(append(z, sig2, axis=0), cum2, axis=0)
      z1 = append(append(z0, delta2, axis=0), deltaCum2, axis=0)
      self.sd = average(sig)
      return z1   
            
