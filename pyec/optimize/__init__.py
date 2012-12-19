"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np

from pyec.config import Config
from pyec.distribution.de import DifferentialEvolution as DE
from pyec.distribution.cmaes import Cmaes
from pyec.distribution.neldermead import NelderMead
from pyec.distribution.gss import GeneratingSetSearch
from pyec.distribution.sa import RealSimulatedAnnealing
from pyec.distribution.pso import ParticleSwarmOptimization
from pyec.distribution.ec.evoanneal import RealEvolutionaryAnnealing
from pyec.space import Euclidean, Hyperrectangle

def optimize(optimizer, func, dimension=5, population=25, generations=100,**kwargs):
   """
      Configure and run an optimizer on a function.
      
      By default the function will be minimize, but maximization can be performed by setting the keyword argument *minimize* to ``False``.
      
      Benchmark functions can be optimized by name. The following names are supported:
      
      - ackley -- A checker-board like oscillator, minimum is -13.37 in 5 dimensions.
      - ackley2 -- Exponentiated and centered version of ackley, minimum is 0 at 0.
      - griewank -- Oscillator with large scale, minimum at 0.
      - langerman -- Sparse, rough, multi-modal. Minimum is 0.98 in five dimensions. 
      - rosenbrock -- Standard benchmark.
      - rastrigin -- Oscillator. Minimum at 
      - salomon -- Ring oscillation. Minimum 0 at 0.
      - schwefel -- Deceptive multimodal function. Minimum is -418 on (-512,512).
      - shekel2 -- Shekel's foxholes, modified. Minimum is -10.4 in five dimensions. 
      - sphere -- A spherical paraboloid, minimum is 0 at 0
      - whitley -- Complex, fractal like shape with small relevant area. Minimum is 0.0.
      - weierstrass -- Everywhere continuous, nowhere differentiable; minimum is 0 at 0.
      
      
      :param optimizer: A :class:`PopulationDistribution` subclass
      :type optimizer: ``class``
      :param func: The function to be optimized, or a lookup key for a benchmark.
      :type func: any callable object or str
      :param dimension: The vector dimension in the search domain
      :type dimension: int
      :param population: The population size (sample size) for the optimizer.
      :type population: int
      :param generations: The number of populations to build (number of samples) during optimization.
      :type generations: int
      :returns: A tuple (best solution, best value) where the first element is the *best solution* located during optimization and *best value* is the value of the function at *best solution*.
      
      
      Keyword arguments:
      
      * minimize -- Whether to minimize the function, otherwise maximize; default is True.
      * initial -- A callable (no arguments) that returns random starting points for the initial distribution of the optimizer.
      * display -- Show progress information once every second.
      * constraint -- A :class:`Boundary` object implementing a constraint region (default is unconstrained). 
      
   """
   space = ("constraint" in kwargs and kwargs["constraint"]
            or Euclidean(dim=dimension))
   config = {
      "minimize":True,
      "space":space,
      "populationSize":population
   }
   config.update(kwargs)
   
   if isinstance(func, basestring):
      from pyec.util.registry import BENCHMARKS
      func = BENCHMARKS.load(func)
      #if config["minimize"]:
      #   h = func
      #   func = lambda x: -h(x)
         
   if config["minimize"]:
     optfunc = lambda x: -func(x)
   else:
     optfunc = func
   
   config = Config(**config)
   alg = (optimizer[config] << generations)()
   pop = alg[None, optfunc]()
   alg.history.update(pop, optfunc, space)
   return alg.history.best()

def differential_evolution(func, **kwargs):
   """
      Apply differential evolution (DE) to optimize a function. See <http://en.wikipedia.org/wiki/Differential_evolution>.
      
      Calls :func:`optimize`. 
      
      Extra keyword arguments:
      
      * CR -- The crossover probability for DE, default 0.2.
      * F -- The learning rate for DE, default 0.5.
   
   """
   return optimize(DE, func, **kwargs)

"""
   Synonym for :func:`differential_evolution`.
"""
de = differential_evolution

def cmaes(func, **kwargs):
   """
      Apply Correlated Matrix Adaptation Evolution Strategy (CMA-ES) to optimize a function. See <http://en.wikipedia.org/wiki/CMA-ES>.
      
      Calls :func:`optimize`. 
      
      Extra keyword arguments:
      
      * parents -- The percentage of the population to use as parents, default 0.5.
      * variance -- The standard deviation for CMA-ES to use during initialization, if Gaussian initialization is used (only unconstrained optimization); default is 1.0.
      
   """
   popSize = "population" in kwargs and kwargs["population"] or 25
   if "parents" in kwargs:
      kwargs["mu"] = int(kwargs["parents"] * popSize)
   return optimize(Cmaes, func, **kwargs)
   
def nelder_mead(func, generations=500, population=1, **kwargs):
   """
      Apply Nelder-Mead method to optimize a function. See <http://en.wikipedia.org/wiki/Nelder-Mead_method>.
      
      Calls :func:`optimize`. 
      
      Extra keyword arguments:
      
      * convergence -- The tolerance on the simplex before restarting; default 1e-10.
      * alpha, beta, gamma, delta -- standard parameters for Nelder-Mead.
      
   """
   if "convergence" in kwargs:
      kwargs["tol"] = kwargs["convergence"]
   return optimize(NelderMead, func, generations=generations, population=population, **kwargs)

"""
   Synomnym for :func:`nelder_mead`.
"""
nm = nelder_mead

def generating_set_search(func, generations=500, population=1, **kwargs):
   """
      Apply a basic generating set search to optimize a function. See <http://smartfields.stanford.edu/documents/080403_kolda.pdf>.
      
      Uses no search heuristic, and uses the d+1 size basis in dimension d.
      
      Calls :func:`optimize`. 
      
      Extra keyword arguments:
      
      * convergence -- The tolerance on the simplex before restarting; default 1e-10.
      * penalty_func -- A penalty function for the objective.
      * expansion_factor -- Multiplicative expansion factor to use when a new best solution is found; default is 1.1.
      * contraction_factor -- Multiplicative contraction factor to use when no new best is found; default is 0.95.
      * initial_step -- The initial step.
      
   """
   if kwargs.has_key('convergence'): kwargs["tol"] = kwargs['convergence']
   if kwargs.has_key('expansion_factor'): kwargs["expand"] = kwargs['expansion_factor']
   if kwargs.has_key('contraction_factor'): kwargs["contract"] = kwargs['contraction_factor']
   if kwargs.has_key('initial_step'): kwargs["step"] = kwargs['initial_step']
   return optimize(GeneratingSetSearch, func, generations=generations, population=population, **kwargs)

"""
   Synonym for :func:`generating_set_search`.
"""
gss = generating_set_search   

def simulated_annealing(func, generations=1000, population=1, **kwargs):
   """
      Apply simulated annealing to optimize a function. See <http://en.wikipedia.org/wiki/Simulated_annealing>.
      
      Calls :func:`optimize`. 
      
      Extra keyword arguments:
      
      * schedule -- One of (log, linear) for a logarithmic or linear cooling schedule, or a function T(n) to return the temperature at time n.
      * learning_rate -- The temperature will be divided by the learning rate is a logarithmic or linear schedule is used.
      * restart_prob -- A probability to restart simulated annealing; 0.001 by default.
      
   """
   if kwargs.has_key('variance'): kwargs["sd"] = kwargs['variance']
   if kwargs.has_key('learning_rate'): kwargs["learningRate"] = kwargs['learning_rate']
   if kwargs.has_key('schedule_divisor'): kwargs["divisor"] = kwargs['schedule_divisor']
   if kwargs.has_key('restart_prob'): kwargs["restart"] = kwargs['restart_prob']
   return optimize(RealSimulatedAnnealing, func, generations=generations, population=population, **kwargs)

"""
   Synonym for :func:`simulated_annealing`.
"""
sa = simulated_annealing

def particle_swarm_optimization(func, generations=100, population=25, **kwargs):
   """
      Apply particle swarm optmization to optimize a function. See <http://en.wikipedia.org/wiki/Particle_swarm_optimization>.
      
      Calls :func:`optimize`. 
      
      Extra keyword arguments:
      
      * omega -- The velocity decay.
      * phi_g -- The global best influence parameter.
      * phi_p -- The local best influence parameter.
      
   """
   if kwargs.has_key('omega'): kwargs["omega"] = kwargs['omega']
   if kwargs.has_key('phi_g'): kwargs["phig"] = kwargs['phi_g']
   if kwargs.has_key('phi_p'): kwargs["phip"] = kwargs['phi_p']
   return optimize(ParticleSwarmOptimization, func, generations=generations, population=population, **kwargs)

"""
   Synonym for :func:`particle_swarm_optimization`.
"""
pso = particle_swarm_optimization

def evolutionary_annealing(func, **kwargs):
   """
      Apply evolutionary annealing to optimize a function. See Chapter 11 of <http://nn.cs.utexas.edu/downloads/papers/lockett.thesis.pdf>.
      
      Calls :func:`optimize`. 
      
      Extra keyword arguments:
      
      * learning_rate -- A scaling factor controlling the temperature schedule; smaller numbers search more slowly and thoroughly, larger numbers search faster and less thoroughly. 
      * variance -- The initial standard deviation of the Gaussian mutation distribution, i.e. how locally the search is spaced. Defaults to 1.0, does not need to be changed.
      * jogo2012 -- Use the parameters from Lockett and Miikkulainen in JOGO   
   """
   if kwargs.has_key('learning_rate'): kwargs["learningRate"] = kwargs['learning_rate']
   if kwargs.has_key('variance'): kwargs["sd"] = kwargs['variance']
   if kwargs.has_key('jogo2012'):
      from pyec.util.partitions import VectorSeparationAlgorithm
      kwargs["jogo2012"] = True
      kwargs["schedule"] = "log"
      kwargs["separator"] = VectorSeparationAlgorithm

   return optimize(RealEvolutionaryAnnealing, func, **kwargs)

"""
   Synonym for :func:`evolutionary_annealing`.
"""
evoanneal = evolutionary_annealing


   