"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from numpy import *
import numpy.linalg as la
import binascii, struct
from pyec.config import Config, ConfigBuilder
from pyec.util.registry import BENCHMARKS
from pyec.util.TernaryString import TernaryString

class Distribution(object):
   """
      A Distribution that can be sampled.
   
      :param config: A set of configuration parameters for the distribution.
      :type config: :class:`Config`
   """

   def __init__(self, config):
      super(Distribution, self).__init__()
      self.config = config

   def __call__(self, **kwargs):
      """Get a single sample from the distribution."""
      return self.batch(1)[0]

   def batch(self, sampleSize):
      """
         Get a sample from the distribution.
      
         :param sampleSize: The size of the sample to generate.
         :type sampleSize: int
      """
      pass

class ProposalDistribution(Distribution):
   """
      A proposal distribution for e.g. simulated annealing.
      
   """

   def adjust(self, rate):
      """Given the current acceptance rate, alter the distribution as needed."""
      pass

   def densityRatio(self, x1, x2, i=None):
      """Given two points, give the ratio of the densities p(x1) / p(x2)."""
      pass

class PopulationDistribution(Distribution):
   """
      A distribution governing a population-based optimizer.
   
      This is the parent class for optimizers in PyEC. Its core methods are ``batch``, inherited from :class:`Distribution`, and ``update``, which reports the population and the scores after a generation is complete.
      
      :params config: The configuration parameters.
      :type config: :class:`Config`
   """

   def update(self, n, population):
      """
         Update the state of the :class:`PopulationDistribution` with the latest population and its fitness scores.
          
         :params population: The previous population with its fitness scores. If ``self.config.sorted`` is ``True``, the list will be sorted by descending score by :class:`Trainer`. Outside of a :class:`Trainer` instance, the caller must ensure that the population is sorted (or not sorted) as necessary.
         :type population: list of (point, score) tuples
      """
      pass

   def run(self, segment='test', fitness=None, extraArgs=None, **kwargs):
     """
        Create a :class:`Trainer` object and run this :class:`PopulationDistribution` instance to maximize a fitness function.
        
        After running this method, the property ``PopulationDistribution.trainer`` will contain the :class:`Trainer` object used to optimize the function. 
        
        :param segment: A name for storing in the database. Deprecated.
        :type segment: str
        :param fitness: The fitness function (objective) to be maximized. If ``None``, then the function will be looked up from ``pyec.util.registry.BENCHMARKS`` based on the ``function`` property of the :class:`Config`.
        :type fitness: any callable object
        :param extraArgs: Any extra args to be passed to ``pyec.util.registry.BENCHMARKS``.
        :type extraArgs: list 
        :returns: The maximum score discovered during optimization.
     """
     from pyec.trainer import Trainer
     if fitness is None:
        if extraArgs is None:
           extraArgs = []
        fitness = BENCHMARKS.load(self.config.function, *extraArgs)
     self.config.segment = segment
     #if hasattr(fitness, 'center'):
     #   self.config.center = fitness.center
     #   self.config.scale = fitness.scale
     #if hasattr(fitness, 'numInputs'):
     #   self.config.numInputs = fitness.numInputs
     #   self.config.numOutputs = fitness.numOutputs
     #   self.config.numHidden = fitness.numHidden
     #   self.config.netPattern = fitness.netPattern
     self.config.fitness = fitness
     try:
        fitness.algorithm = self
        fitness.config = self.config
     except:
        pass
     #if hasattr(fitness, 'initial'):
     #   self.config.initialDistribution = fitness.initial
     #   self.initial = self.config.initialDistribution
     trainer = Trainer(fitness, self, **kwargs)
     trainer.train()
     self.trainer = trainer
     return fitness

   def convert(self, x):
      """ 
         Convert a point to a scorable representation. Deprecated. Use ``Config.convert`` instead.
         
        :params x: The candidate solution
      """
      return x

   @classmethod
   def configure(cls, generations, populationSize, dimension=1, function=None):
      """
         Return a Config object
      """
      return cls.configurator().configure(generations, populationSize, dimension, function)
      
   @classmethod
   def configurator(cls):
      """
         Return a ConfigurationBuilder
      """
      return ConfigBuilder()



class Gaussian(ProposalDistribution, PopulationDistribution):
   """ 
   
      A Gaussian Proposal Distribution. Used as an initial distribution
      by several algorithms, and as a proposal distribution by Simulated Annealing.
      
      :params config: The configuration parameters.
      :type config: :class:`Config`
   
   """
   def __init__(self, config):
      super(Gaussian, self).__init__(config)
      self.var = 1.
      if hasattr(config, 'spaceScale') and config.spaceScale is not None:
         self.var = config.spaceScale / 2.
      elif hasattr(config, 'varInit') and config.varInit is not None:
         self.var = config.varInit
      self.varIncr = 1.05
      self.usePrior = hasattr(config, 'usePrior') and config.usePrior or False

   def __call__(self, **kwargs):
      center = zeros(self.config.dim)
      if self.usePrior and kwargs.has_key('prior'):
         center = kwargs['prior']
      # vary the mixture point
      var = self.variance()
      if kwargs.has_key('idx') and hasattr(var, '__len__'):
         var = var[kwargs['idx']]
      varied = random.randn(self.config.dim) * var + center
      
      
      # check bounds; call again if outside bounds
      if self.config.bounded:
         try:
            if not self.config.in_bounds(varied):
               return self.__call__(**kwargs)
         except RuntimeError, msg:
            print "Recursion error: ", varied
            print abs(varied - self.config.center)
            print self.config.scale
      return varied

   def batch(self, popSize):
      return [self.__call__(idx=i) for i in xrange(popSize)]
   
   def density(self, x, center):
      var = self.variance()
      if isinstance(var, ndarray):
         var = var[0]
         #print x
      diff = center - x
      if sqrt((diff * diff).sum()) > 5. * var:
         return 0.0
      covar = var * var * identity(len(x))
      diff = center - x
      pow = -.5 * dot(diff, dot(la.inv(covar), diff))
      d = ((((2*pi)**(len(x))) * la.det(covar)) ** -.5) * exp(pow)
      return d       
                     
   def variance(self):
      return self.var
            
   def adjust(self, rate):
      if not hasattr(rate, '__len__'):
         if rate < .23:
            self.var /= self.varIncr
         else:
            if self.var < self.config.scale / 5.:
               self.var *= self.varIncr
         return
      self.var = self.var * ones(len(rate))
      for i in  xrange(len(rate)):
         if rate[i] < .23:
            self.var[i] /= self.varIncr
         else:
            if self.var[i] < self.config.scale / 5.:
               self.var[i] *= self.varIncr
      #print rate, self.var

   def densityRatio(self, x1, x2, i = None):
      if self.usePrior:
         return 1.
      else:
         if i is None:
            var = self.var
         else:
            var = self.var[i]
         return exp((1./(2*(var**2))) * ((x2 ** 2).sum() - (x1 ** 2).sum()))
         

class Bernoulli(Distribution):
   """
      A Bernoulli distribution over the binary cube. Returns a :class:`numpy.ndarray` with values of 0.0 or 1.0.
      
      Uses ``config.dim`` to determine the number of bits. The probability of bit being on is given by ``config.bitFlipProb``, which defaults to 0.5.
      
      :params config: The configuration parameters.
      :type config: :class:`Config`
   """
   def __init__(self, config):
      super(Bernoulli, self).__init__(config)
      self.dim = config.dim
      self.bitFlipProb = .5

   def __call__(self, **kwargs):
      return random.binomial(1, self.bitFlipProb, self.dim)
      
   def batch(self, popSize):
      return [self.__call__() for i in xrange(popSize)] 

class BernoulliTernary(Distribution):
   """
      A Bernoulli distribution that allows for missing variables.
      Generates a :class:`TernaryString` with all bits filled in randomly.
      The variable ``config.dim`` is used to determine the number of bits.
      The probability of a bit being on is 0.50. More complicated 
      distributions over :class:`TernaryString` are available in 
      :mod:`pyec.distribution.ec.mutators`.
      
      :params config: The configuration parameters.
      :type config: :class:`Config`
   """
   def __init__(self, config):
      super(BernoulliTernary, self).__init__(config)
      self.dim = config.dim

   def __call__(self, **kwargs):
      numBytes = int(ceil(self.dim / 8.0))
      numFull  = self.dim / 8
      initial = ''
      if numBytes != numFull:
         extra = self.dim % 8
         initMask = 0
         for i in xrange(extra):
            initMask <<= 1
            initMask |= 1
         initial = struct.pack('B',initMask)
            
      base = long(binascii.hexlify(random.bytes(numBytes)), 16)
      known = long(binascii.hexlify(initial + '\xff'*numFull), 16)
      return TernaryString(base, known)
      
   def batch(self, popSize):
      return [self.__call__() for i in xrange(popSize)] 




class FixedCube(ProposalDistribution):
   """
      A uniform distribution over a fixed cube in Euclidean space. Used as an 
      initial distribution for constrained optimization problems.
      
      Uses ``config.dim`` to determine the dimension.
      
      Checks ``config.in_bounds`` to verify that the produced points
      are inside of the constrained region. 
      
      Uses ``config.in_bounds.extent`` to determine the size of the
      hypercube to generate points for. If this function does not exist, ``config.center`` and ``config.scale`` are used instead.
      
      :params config: The configuration parameters.
      :type config: :class:`Config`
   """

   def __init__(self, config):
      super(FixedCube, self).__init__(config)
      self.scale = self.config.scale
      self.center = self.config.center
      if hasattr(config, 'in_bounds'):
         if hasattr(config.in_bounds, 'extent'):
            self.center, self.scale = config.in_bounds.extent()
            
   def __call__(self, **kwargs):
      point = (random.random_sample(self.config.dim) - .5) * 2 * self.scale + self.center
      while self.config.bounded and not self.config.in_bounds(point):
         point = (random.random_sample(self.config.dim) - .5) * 2 * self.scale + self.center
      return point

   def batch(self, popSize):
      return [self.__call__() for i in xrange(popSize)] 

   def densityRatio(self, x1, x2):
      return 1.

   def adjust(self, rate):
      pass



