"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from numpy import *
import numpy.linalg as la
import binascii, struct
import inspect
from pyec.config import Config, ConfigBuilder
from pyec.history import History, MarkovHistory, SortedMarkovHistory, DoubleMarkovHistory
from pyec.space import Euclidean, Binary
from pyec.util.registry import BENCHMARKS
from pyec.util.RunStats import RunStats
from pyec.util.TernaryString import TernaryString

class Distribution(object):
   """
      A Distribution that can be sampled.
   
      :param config: A set of configuration parameters for the distribution.
      :type config: :class:`Config`
   """
   config = Config()

   def __init__(self, **kwargs):
      super(Distribution, self).__init__()
      self.config = Distribution.config.merge(Config(**kwargs))

   def __call__(self):
      return self.sample()

   def sample(self, **kwargs):
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


def initStandard(cls, init=None):
     """
        Standard initialization for population distributions, 
        inserted by :class:`PopulationDistributionMeta`. Will
        call the provided initialization method as well.
        
        :param cls: The class being initialized
        :type cls: A class of type :class:`PopulationDistributionMeta`
        :param init: The existing initialization method as defined in the
                     class; keyword arguments will be placed in a config
                     and merged in with the class level config. IT IS STILL
                     THE RESPONSIBILITY OF THE USER TO CALL 
                     ``super().__init__`` IF ``__init__`` 
                     IS OVERRIDDEN! This provides a level of automaticity
                     without preventing the user from doing whatever they
                     want.
                     
     """
     if init is None:
         def __init__(self, **kwargs):
             config = cls.config.merge(Config(**kwargs))
             super(cls, self).__init__(**config.__properties__)
             self.history = None
             self.fitness = None
    
     else:
         def __init__(self, *args, **kwargs):
             config = cls.config.merge(Config(**kwargs))
             self.history = None
             self.fitness = None
             init(self, *args, **config.__properties__)
    
     return __init__


class PopulationDistributionMeta(type):
   """Metaclass for optimizers; defines class operations that generate
   new optimizers before instantiation.
      
   """
   __registry__ = {}
   
   def __new__(cls, name, bases, attrs):
   
      init = attrs.get(u"__init__", None)
      if init:
          del attrs["__init__"]
      ret = super(PopulationDistributionMeta, cls).__new__(cls, 
                                                           name, 
                                                           bases, 
                                                           attrs)
      ret.__init__ = initStandard(ret, init)
      PopulationDistributionMeta.__registry__[name] = ret
      return ret
   
   def _checkName(cls, name):
      """Check whether a generated name is already in use; if so, 
      return the class assigned that name, otherwise return ``None``.
      
      :param name: A name generated by ``cls._makeName()``
      :type name: ``str``
      :returns: The class with the provided name, or ``None``
      
      """
      return PopulationDistributionMeta.__registry__.get(name, None)
   
   def __getitem__(cls, cfg):
      """
         Create a :class:`PopulationDistribution` based on this class, 
         configured to apply certain parameters
         
         :param cfg: A config object to be applied
         :type cfg: :class:`Config`
         :returns: A new population distribution class
      """
      name = "{0}_qqq{1}qqq".format(cls.__name__, hash(cfg))
      cls2 = cls._checkName(name)
      if cls2 is None:
          bases = (cls,)
          attrs = dict(((k,v) for k,v in cls.__dict__.iteritems()))
          attrs["config"] = cfg
          cls2 = PopulationDistributionMeta(name, bases, attrs)
      return cls2

   def __mul__(cls, other):
      """
         Create a population distribution through scalar multiplication 
         by a nonnegative number
     
         :param other: A scalar weight
         :type other: a nonnegative scalar
         :returns: A new population distribution class
         
      """
      val = float(other)
      if val < 0.0:
         raise ValueError("Cannot multiply an optimizer by a negative number")
      
      
      name = "{0}_mult_qqq{1}qqq".format(cls.__name__, val)
      name = name.replace(".","_").replace("-","_")
      cls2 = cls._checkName(name)
      if cls2 is None:
          bases = (cls,)
          attrs = attrs = dict(((k,v) for k,v in cls.__dict__.iteritems()))
          attrs["weight"] = val * cls.weight
          cls2 = PopulationDistributionMeta(name, bases, attrs)
      return cls2
   
   __rmul__ = __mul__
   
   def __add__(cls, other):
      """
         Create a population distribution by adding two optimizers.
     
         :param other: A generator for second optimizer, possibly weighted
         :type other: A class subclassing :class:`PopulationDistribution`
         :returns: A new population distribution class
      """
      if (not inspect.isclass(other) or 
          not issubclass(other, PopulationDistribution)):
          raise ValueError("Optimizer addition requires optimizers")
    
      import pyec.distribution.convex
      cvx = pyec.distribution.convex.Convex
      name = "{0}_add_qqq{1}qqq".format(cls.__name__, other.__name__)
      cls2 = cls._checkName(name)
      if cls2 is None:
          bases = (cvx,)
          def init(self, **kwargs):
              first = cls(**kwargs)
              second = other(**kwargs)
              kwargs.update(second.config.__properties__)
              kwargs.update(first.config.__properties__)
              cvx.__init__(self, (first, second,), **kwargs)
          attrs = dict(((k,v) for k,v in cvx.__dict__.iteritems()))
          attrs["__init__"] = init
          cls2 = PopulationDistributionMeta(name, bases, attrs)
      return cls2


   def __lshift__(cls, other):
      """
         Create a population distribution by convolving two optimizers.
     
         :param other: A generator for second optimizer, possibly weighted
                       or an integer for self convolution
         :type other: A class subclassing :class:`PopulationDistribution`
                      or ``int``
         :returns: A new population distribution class
         
      """
      import pyec.distribution.convolution
      conv = pyec.distribution.convolution.Convolution
      
      if inspect.isclass(other) and issubclass(other, PopulationDistribution):
          name = "{0}_convolve_qqq{1}qqq".format(cls.__name__, other.__name__)
          cls2 = cls._checkName(name)
          if cls2 is None:
              bases = (conv,)
              def init(self, **kwargs):
                  first = cls(**kwargs)
                  second = other(**kwargs)
                  kwargs.update(second.config.__properties__)
                  kwargs.update(first.config.__properties__)
                  conv.__init__(self, 
                                (first, second,), 
                                **kwargs)
              attrs = dict(((k,v) for k,v in conv.__dict__.iteritems())) 
              attrs["__init__"] = init
              cls2 = PopulationDistributionMeta(name, bases, attrs)
          return cls2
      
      try:
          val = int(other)
      except Exception:
          err = "Convolution requires either an integer or an optimizer"
          raise ValueError(err)
      else:
          if val <= 0:
              raise ValueError("Cannot convolve by a negative number")
      
          conv = pyec.distribution.convolution.SelfConvolution
          name = "{0}_convolve_{1}".format(cls.__name__, other)
          cls2 = cls._checkName(name)
          if cls2 is None:
              bases = (conv,)
              def init(self, **kwargs):
                  first = cls(**kwargs)
                  kwargs.update(first.config.__properties__)
                  conv.__init__(self, first, other, **kwargs)
              attrs = dict(((k,v) for k,v in conv.__dict__.iteritems()))
              attrs["__init__"] = init
              cls2 = PopulationDistributionMeta(name, bases, attrs)
          return cls2     

   def __rshift__(cls, other):
      """Create a population distribution by trajectory truncation.
         
         Argument specifies the number of steps to truncate.
         
         Or, argument can be another optimizer, in which case the result
         is a convolution with the right side truncated one step.
      
         :param other: The integer number of steps to truncate, or optimizer
         :type other: ``int`` or :class:`PopulationDistribution`
         :returns: A new population distribution class, truncated or convolved 
                   with a truncated operator
      
      """
      import pyec.distribution.convolution
      import pyec.distribution.truncation
      tt = pyec.distribution.truncation.TrajectoryTruncation
                      
      if inspect.isclass(other) and issubclass(other, PopulationDistribution):
          name = "{0}_truncate_1".format(cls.__name__)
          cls2 = cls._checkName(name)
          if cls2 is None:
              bases = (tt,)
              def init(self, **kwargs):
                  sub = other(**kwargs)
                  kwargs.update(sub.config.__properties__)
                  tt.__init__(self, sub, 1, **kwargs)
              attrs = dict(((k,v) for k,v in tt.__dict__.iteritems()))
              attrs["__init__"] = init
              cls2 = PopulationDistributionMeta(name, bases, attrs)
          
          return cls << cls2
      
      try:
         other = int(other)
      except Exception:
         raise ValueError("Could not coerce truncation argument to integer")
      
      else:
          name = "{0}_truncate_{1}".format(cls.__name__, other)
          cls2 = cls._checkName(name)
          if cls2 is None:
              bases = (tt,)
              def init(self, **kwargs):
                  sub = cls(**kwargs)
                  kwargs.update(sub.config.__properties__)
                  tt.__init__(self, sub, other, **kwargs)
              attrs = dict(((k,v) for k,v in tt.__dict__.iteritems()))
              attrs["__init__"] = init
              cls2 = PopulationDistributionMeta(name, bases, attrs)
          
          return cls2

   def __or__(cls, other):
      """Given two optimizers create a new optimizer that splits the
      population between the two of them, apportioning according to
      the ``weight`` property of the optimizers being combined
      
      """
      if (not inspect.isclass(other) and
          not issubclass(other, PopulationDistribution)):
         raise ValueError("Expected pipe argument "
                          "to be a PopulationDistribution subclass")

      name = "{0}_pipe_qqq{1}qqq".format(cls.__name__, other.__name__)
      cls2 = cls._checkName(name)
      if cls2 is None:
         import pyec.distribution.split
         split = pyec.distribution.split.Splitter
         bases = (split,)
         def init(self, **kwargs):
            first = cls(**kwargs)
            second = other(**kwargs)
            kwargs.update(second.config.__properties__)
            kwargs.update(first.config.__properties__)
            subs = (first, second)
            split.__init__(self, subs, **kwargs)
         attrs = dict(((k,v) for k,v in split.__dict__.iteritems())) 
         attrs["__init__"] = init
         cls2 = PopulationDistributionMeta(name, bases, attrs)
         
      return cls2


class PopulationDistribution(Distribution):
   """
      A distribution governing a population-based optimizer.
   
      This is the parent class for optimizers in PyEC. Its core methods are ``batch``, inherited from :class:`Distribution`, and ``update``, which reports the population and the scores after a generation is complete.
      
      Config parameters
      * populationSize -- The size of each population, default 1
      * history -- The class of the :class:`History` to use, default
                   :class:`SortedMarkovHistory`
      * space -- The :class:`Space` to search, default `None` must be
                 overridden
      * initial -- The initial distribution; a :class:`Distribution`, 
                   a callable object that returns a single solution,
                   or ``None`` to use the space's randomizer
      
      :params config: The configuration parameters.
      :type config: :class:`Config`
   """
   __metaclass__ = PopulationDistributionMeta
   weight = 1.0
   config = Config(populationSize=1, # The size of each population
                   history=SortedMarkovHistory, # The class for the history
                   space=None, # The search space
                   minimize=True,
                   stats = RunStats()
                  )
   
   def run(self, fitness=None, history=None, extraArgs=None, **kwargs):
      """
        Run this :class:`PopulationDistribution` instance to maximize a fitness function.
        
        After running this method, the property ``PopulationDistribution.trainer`` will contain the :class:`Trainer` object used to optimize the function. 
        
        :param fitness: The fitness function (objective) to be maximized. 
                        If ``None``, then the function will be looked up from 
                        ``pyec.util.registry.BENCHMARKS`` based on the 
                        ``function`` property of the :class:`Config`.
        :type fitness: any callable object with a single argument, or ``None``
        :param history: A history object to extend, or ``None`` to create
                        a new history from the class in the ``history``
                        property of the :class:`Config` object.
        :type history: :class:`History` or ``None``
        :param extraArgs: Any extra args to be passed to 
                          ``pyec.util.registry.BENCHMARKS``.
        :type extraArgs: list 
        :returns: The :class:`History for the object
      """
     
      # Set up the fitness if necessary
      if fitness is None:
         if extraArgs is None:
            extraArgs = []
         fitness = BENCHMARKS.load(self.config.function, *extraArgs)
         try:
            fitness.algorithm = self
            fitness.config = self.config
         except Exception:
            pass
         self.config.fitness = fitness
     
      # Set up history
      if history is None:
         history = self.config.history(self.config)
     
      if not self.compatible(history):
         name = history.__class__.__name__
         raise ValueError("Incompatible history class {0}".format(name))
    
      self.update(history, fitness)
        
      # get the sample
      pop = self.population(self.config.populationSize)
     
      # update the history
      self.history.update(pop, fitness, self.config.space)
     
      return self.history
      
   def __call__(self):
      popSize = self.config.populationSize
      if self.history.empty():
          if self.config.initial is None:
              return [self.config.space.random() for i in xrange(popSize)]
          elif hasattr(self.config.initial, 'batch'):
              return self.config.initial.batch(popSize)
          else:
              return [self.config.initial() for i in xrange(popSize)]
      else:
          return self.batch(popSize)

   def __getitem__(self, pair):
      """Call ``update`` by splitting out the pair into a :class:`History` and
      a fitness function. Returns the optimizer for use as a continuation.
      
      :param pair: A tuple with the history and fitness.
      :type pair: A ``tuple`` of a :class:`History` and a callable object
      :returns: This optimizer (``self``)
      
      """
      history, fitness = pair
     
      # Set up history
      if history is None:
         history = self.config.history(self.config)
      
      if not self.compatible(history):
          err = ("Got an incompatible history in __getitem__; "
                 "expected [history,fitness]")
          raise ValueError(err)
          
      # Set up the fitness if necessary
      if fitness is None:
          fitness = self.config.function
      
      if isinstance(fitness, basestring):
          fitness = BENCHMARKS.load(fitness)
          try:
              fitness.algorithm = self
              fitness.config = self.config
          except Exception:
              pass
          self.config.fitness = fitness    
    
      if not inspect.isfunction(fitness) and not hasattr(fitness, '__call__'):
          err = ("Second object in __getitem__ is not a function; "
                 "expected [history,fitness]")
          raise ValueError(err)
    
      self.update(history, fitness)
      return self
      
   def __mul__(self, other):
      """
         Create a population distribution through scalar multiplication 
         by a nonnegative number.
     
         :param other: A scalar weight
         :type other: a nonnegative scalar
         :returns: A new optimizer with the same class but multiplied weight
         
      """
      val = float(other)
      if val < 0.0:
         raise ValueError("Cannot multiply an optimizer by a negative number")
      
      ret = self.__class__(**self.config.__properties__)
      ret.weight = self.weight * val
      return ret

   __rmul__ = __mul__

   def __imul__(self, other):
      """
         Modify this population distribution through scalar multiplication 
         by a nonnegative number.
     
         :param other: A scalar weight
         :type other: a nonnegative scalar
         :returns: This optimizer (``self``)
         
      """
      val = float(other)
      if val < 0.0:
         raise ValueError("Cannot multiply an optimizer by a negative number")
      
      self.weight *= val
      return self
      
   def __add__(self, other):
      """
         Create a population distribution by adding two optimizers.
     
         :param other: A generator for second optimizer, possibly weighted
         :type other: A class subclassing :class:`PopulationDistribution`
         :returns: :class:`Convex`
      """
      if not isinstance(other, PopulationDistribution):
          raise ValueError("Optimizer addition requires optimizers")
    
      import pyec.distribution.convex
      cvx = pyec.distribution.convex.Convex
      return cvx((self, other), **self.config.__properties__)
      
   __iadd__ = __add__

   def __lshift__(self, other):
      """
         Create a population distribution by convolving two optimizers.
     
         :param other: A generator for second optimizer, possibly weighted
                       or an integer for self convolution
         :type other: A class subclassing :class:`PopulationDistribution`
                      or ``int``
         :returns: :class:`Convolution`
         
      """
      import pyec.distribution.convolution
      conv = pyec.distribution.convolution.Convolution
      
      if isinstance(other, PopulationDistribution):
          return conv((self, other), **self.config.__properties__)
      
      try:
          val = int(other)
      except Exception:
          err = "Convolution requires either an integer or an optimizer"
          raise ValueError(err)
      else:
          if val <= 0:
              raise ValueError("Cannot convolve by a negative number")
      
          conv = pyec.distribution.convolution.SelfConvolution
          return conv(self, val, **self.config.__properties__)    

   __ilshift__ = __lshift__

   def __rshift__(self, other):
      """Create a population distribution by trajectory truncation.
         
         Argument specifies the number of steps to truncate.
         
         Or, argument can be another optimizer, in which case the result
         is a convolution with the right side truncated one step.
      
         :param other: The integer number of steps to truncate, or optimizer
         :type other: ``int`` or :class:`PopulationDistribution`
         :returns: A :class:`TrajectoryTruncation` or :class:`Convolution`
      
      """
      import pyec.distribution.convolution
      import pyec.distribution.truncation
      tt = pyec.distribution.truncation.TrajectoryTruncation
                      
      if isinstance(other, PopulationDistribution):
          return self << tt(other, 1, **other.config.__properties__)
      
      try:
         other = int(other)
      except Exception:
         raise ValueError("Could not coerce truncation argument to integer")
      
      else:
          return tt(self, other, **self.config.__properties__)

   __irshift__ = __rshift__

   def __or__(self, other):
      """Given two optimizers create a new optimizer that splits the
      population between the two of them, apportioning according to
      the ``weight`` property of the optimizers being combined
      
      """
      if not isinstance(other, PopulationDistribution):
         raise ValueError("Expected pipe argument "
                          "to be a PopulationDistribution")
      
      import pyec.distribution.split
      split = pyec.distribution.split.Splitter
      return split((self,other), **self.config.__properties__)
   
   __ior__ = __or__

   def update(self, history, fitness):
      """
         Update the state of the :class:`PopulationDistribution` based on the
         history.
          
         :params history: A :class:`History` object with sufficient info
                          to update the state of the optimizer 
         :type history: :class: `History`
         :param fitness: A fitness function
         :type fitness: Any callable
         
      """
      self.history = history
      self.fitness = fitness

   def convert(self, x):
      """ 
         Convert a point to a scorable representation. Deprecated. Use 
         ``Space.convert`` instead.
         
        :params x: The candidate solution (genotype)
        :returns: The converted candidate (phenotype)
        
      """
      return self.config.space.convert(x)

   def compatible(self, history):
      """Check whether the provided history is acceptable for this
      optimizer.
      
      :param history: The history object that will be used to track
                           the progress of the algorithm
      :type history: A :class:`History`
      :returns: ``bool`` -- whether the history class is acceptable
      
      """
      return True
      
   def needsScores(self):
      """Whether the optimizer uses the scores at all; used to 
      prevent evaluation when not necessary in convolutions.
      
      :returns: ``bool`` -- whether this optimizer needs scores
      
      """
      return True


class HistoryMapper(object):
   """Mixin."""
   def __init__(self, **kwargs):
      self.history = None
      self.historyCache = {}
      
   def mapHistory(self, sub, history=None):
      """Find a compatible subhistory for this suboptimizer.
      
      :param sub: One of the suboptimizers for this convolution
      :type sub: :class:`PopulationDistribution`
      :returns: A compatible :class:`History` for the suboptimizer
      
      """
      if history is None:
         history = self.history
      
      if history is not self.history:
         self.historyCache = {}
      
      if sub not in self.historyCache:
         for h in history.histories:
            try:
               if sub.compatible(h):
                  self.historyCache[sub] = h
                  break
            except:
               pass
      
      if sub in self.historyCache:
         return self.historyCache[sub]
      
      c = sub.__class__.__name__  
      raise ValueError("No compatible history found for {0}".format(c ))


class GaussianProposal(ProposalDistribution, PopulationDistribution):
   """ 
   
      A Gaussian Proposal Distribution. Used as an initial distribution
      by several algorithms, and as a proposal distribution by Simulated Annealing.
      
      Config parameters:
      * sd -- The standard deviation for the Gaussian, may be ndarray or float
      * sd_factor -- The factor to multiply / divide when adjusting the std.
                     dev. dynamically
      
      :params config: The configuration parameters.
      :type config: :class:`Config`
   
   """
   config = Config(sd=1.0,   # the initial standard dev for the Gaussian; 
                             # may be float or ndarray
                   sd_factor=1.05, # the factor to multiply / divide
                                     # when adjusting the std. dev. dynamically
                   # use a DoubleMarkovHistory to match simulated annealing
                   # so that convolution doesn't cause the different
                   # histories to disagree on the acceptance rate.
                   # This feels like a hack, and could cause
                   # problems for anyone using this class, and so
                   # a better solution should be devised.
                   history=DoubleMarkovHistory
                  )
   
   def __init__(self, **kwargs):
      super(GaussianProposal, self).__init__(**kwargs)
      if not isinstance(self.config.space, Euclidean):
         raise ValueError("Cannot use Gaussian on a non-Euclidean space")
      self.var = self.config.sd
      if not isinstance(self.var, ndarray):
         self.var = self.var * ones(self.config.space.dim)
      self.varIncr = self.config.sd_factor
      self.last = None
      
   def compatible(self, history):
      return hasattr(history, 'lastPopulation')

   def sample(self, **kwargs):
      center = self.history.lastPopulation()
      if center is None:
         center = zeros(self.config.space.dim)
      else:
         index = "index" in kwargs and kwargs["index"] or 0
         center = center[index][0]
      
      var = self.variance()
      varied = random.randn(self.config.space.dim) * var + center
      
      # check bounds; call again if outside bounds
      try:
         if not self.config.space.in_bounds(varied):
             return self.sample(**kwargs)
      except RuntimeError, msg:
         print "Recursion error: ", varied
         print abs(varied - self.config.space.center)
         print self.config.space.scale
      return varied

   def batch(self, popSize):
      return [self.sample(index=i) for i in xrange(popSize)]
   
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
   
   def update(self, history, fitness):
      super(GaussianProposal, self).update(history,fitness)
      if hasattr(history, 'acceptanceRate') and (history.evals % 100) == 0:
         self.adjust(history.acceptanceRate)
            
   def adjust(self, rate):
      if rate is None:
         return
      elif rate < .23:
         self.var /= self.varIncr
      else:
         self.var *= self.varIncr
         self.var = minimum(self.config.space.scale/5., self.var)

   def densityRatio(self, x1, x2, i = None):
      if self.usePrior:
         return 1.
      else:
         if i is None:
            var = self.var
            center = zeros(self.config.space.dim)
         else:
            var = self.var[i]
            center = self.last[i]
         return exp((1./(2*(var**2))) * (((center - x2) ** 2).sum() 
                                         - ((center - x1) ** 2).sum()))

   def needsScores(self):
      return False         
