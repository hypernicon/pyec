"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np

from pyec.distribution.basic import FixedCube
from pyec.distribution.de import DEConfigurator
from pyec.distribution.cmaes import CmaesConfigurator
from pyec.distribution.neldermead import NMConfigurator
from pyec.distribution.gss import GSSConfigurator
from pyec.distribution.sa import SAConfigurator
from pyec.distribution.pso import PSOConfigurator
from pyec.distribution.ec.evoanneal import REAConfigurator

def optimize(configurator, func, dimension=5, population=25, generations=100,**kwargs):
   """
      Configure and run an optimizer on a function using a :class:`ConfigBuilder` object in order to instantiate the optimizer.
      
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
      
      
      :param configurator: A builder used to generate a parameterized optimizer.
      :type configurator: :class:`ConfigBuilder`
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
   configurator.cfg.printOut = False
   minimize = True
   if kwargs.has_key('minimize'):
      minimize = kwargs['minimize']
   if type(func) == type("") or type(func) == type(u""):
      from pyec.util.registry import BENCHMARKS
      func = BENCHMARKS.load(func)
      if minimize:
         h = func
         func = lambda x: -h(x)
   if kwargs.has_key('display'):
      configurator.cfg.printOut = kwargs['display']
   configurator.cfg.save = kwargs.has_key('save') and kwargs['save'] or False
   configurator.cfg.dim = dimension
   configurator.cfg.bounded = False
   if kwargs.has_key('constraint'):
      configurator.cfg.in_bounds = kwargs['constraint']
      configurator.cfg.bounded = True
   if kwargs.has_key('initial'):
      configurator.cfg.initialDistribution = kwargs['initial']
   elif configurator.cfg.bounded:
      configurator.cfg.initialDistribution = FixedCube(configurator.cfg)
   if minimize:
     optfunc = lambda x: -func(x)
   else:
     optfunc = func
   alg = configurator.configure(generations, population, dimension, optfunc)
   alg.run("",optfunc)
   trainer = alg.trainer
   return trainer.maxOrg, minimize and -trainer.maxScore or trainer.maxScore

def differential_evolution(func, **kwargs):
   """
      Apply differential evolution (DE) to optimize a function. See <http://en.wikipedia.org/wiki/Differential_evolution>.
      
      Calls :func:`optimize`. 
      
      Extra keyword arguments:
      
      * CR -- The crossover probability for DE, default 0.2.
      * F -- The learning rate for DE, default 0.5.
      * variance -- The standard deviation for DE to use during unconstrained initialization, default 1.0.
   
   """
   configurator = DEConfigurator()
   configurator.cfg.crossoverProb = kwargs.has_key('CR') and kwargs["CR"] or .2
   configurator.cfg.learningRate = kwargs.has_key('F') and kwargs["F"] or .5
   configurator.cfg.varInit = kwargs.has_key('variance') and kwargs["variance"] or 1.0
   return optimize(configurator, func, **kwargs)

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
   
   configurator = CmaesConfigurator()
   configurator.cfg.varInit = kwargs.has_key('variance') and kwargs["variance"] or 1.0
   configurator.cfg.muProportion = kwargs.has_key('parents') and kwargs['parents'] or .5
   return optimize(configurator, func, **kwargs)
   
def nelder_mead(func, generations=500, population=1, **kwargs):
   """
      Apply Nelder-Mead method to optimize a function. See <http://en.wikipedia.org/wiki/Nelder-Mead_method>.
      
      Calls :func:`optimize`. 
      
      Extra keyword arguments:
      
      * convergence -- The tolerance on the simplex before restarting; default 1e-10.
      * variance -- The standard deviation for CMA-ES to use during initialization; default is 1.0 (unconstrained spaces).
      * alpha, beta, gamma, delta -- standard parameters for Nelder-Mead.
      
   """
   configurator = NMConfigurator()
   configurator.cfg.varInit = kwargs.has_key('variance') and kwargs["variance"] or 1.0
   if kwargs.has_key('convergence'): configurator.cfg.restartTolerance = kwargs['convergence']
   if kwargs.has_key('alpha'): configurator.cfg.alpha = kwargs['alpha']
   if kwargs.has_key('beta'): configurator.cfg.beta = kwargs['beta']
   if kwargs.has_key('gamma'): configurator.cfg.gamma = kwargs['gamma']
   if kwargs.has_key('delta'): configurator.cfg.delta = kwargs['delta']
   return optimize(configurator, func, generations=generations, population=population, **kwargs)

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
   configurator = GSSConfigurator()
   if kwargs.has_key('convergence'): configurator.cfg.tolerance = kwargs['convergence']
   if kwargs.has_key('penalty_func'): configurator.cfg.penalty = kwargs['penalty_func']
   if kwargs.has_key('expansion_factor'): configurator.cfg.expandStep = kwargs['expansion_factor']
   if kwargs.has_key('contraction_factor'): configurator.cfg.contractStep = kwargs['contraction_factor']
   if kwargs.has_key('initial_step'): configurator.cfg.stepInit = kwargs['initial_step']
   return optimize(configurator, func, generations=generations, population=population, **kwargs)

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
      * proposal -- The proposal distribution; a Gaussian with adaptive variance is used by default.
      * variance -- Initial standard deviation for the default Gaussian.
      * restart_prob -- A probability to restart simulated annealing; 0.001 by default.
      
   """
   configurator = SAConfigurator()
   if kwargs.has_key('schedule'): configurator.cfg.schedule = kwargs['schedule']
   if kwargs.has_key('proposal'): configurator.cfg.proposal = kwargs['proposal']
   if kwargs.has_key('learning_rate'): configurator.cfg.learningRate = kwargs['learning_rate']
   if kwargs.has_key('schedule_divisor'): configurator.cfg.divisor = kwargs['schedule_divisor']
   if kwargs.has_key('variance'): configurator.cfg.varInit = kwargs['variance']
   if kwargs.has_key('restart_prob'): configurator.cfg.restartProb = kwargs['restart_prob']
   return optimize(configurator, func, generations=generations, population=population, **kwargs)

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
   configurator = PSOConfigurator()
   if kwargs.has_key('omega'): configurator.cfg.omega = kwargs['omega']
   if kwargs.has_key('phi_g'): configurator.cfg.phig = kwargs['phi_g']
   if kwargs.has_key('phi_p'): configurator.cfg.phip = kwargs['phi_p']
   return optimize(configurator, func, generations=generations, population=population, **kwargs)

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
      * space_scale -- For unconstrained optimization, the standard deviation of the initial Gaussian and of the Gaussian warping in the space. Should match a general notion of the size or scale of the function in the search domain.
      
   """
   
   configurator = REAConfigurator()
   if kwargs.has_key('learning_rate'): configurator.cfg.learningRate = kwargs['learning_rate']
   if kwargs.has_key('variance'): configurator.cfg.varInit = kwargs['variance']
   configurator.cfg.spaceScale = kwargs.has_key('space_scale') and kwargs['space_scale'] or 10.0
   if kwargs.has_key('jogo2012'):
      configurator.cfg.jogo2012 = True
      configurator.cfg.anneal_log = True
      configurator.cfg.partitionLongest = False
   kwargs['save'] = True
   return optimize(configurator, func, **kwargs)

"""
   Synonym for :func:`evolutionary_annealing`.
"""
evoanneal = evolutionary_annealing


   