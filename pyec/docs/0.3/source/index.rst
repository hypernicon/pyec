.. PyEC documentation master file, created by
   sphinx-quickstart on Thu Jul 12 11:03:55 2012.

Introduction to PyEC
================================

PyEC provides an implementation of several well-known optimization methods with a particular focus on methods of evolutionary computation or natural computation. If you are just interested in optimizing functions, see the section on `Basic Usage` below.

For the time being, this code is in an experimental stage and will likely change substantially with the next few minor versions.

Who is PyEC for?
-------------------
PyEC is intended for three main groups of people:

**Anyone who needs to optimize functions.** PyEC can be used as a drop-in module to perform optimization quickly and accurately. The most common optimization methods have been packaged in an easily accessible form for this purpose, as described in `Basic Usage` below.

**Researchers in Evolutionary Computation.** PyEC contains tools to construct, modify, and parameterize optimization methods. It can be used as a toolbox for the researcher in evolutionary computation to construct evolutionary algorithms on complex search domains, or to test new genetic operators or new combinations of existing evolutionary components. These features are not well documented yet, but the class-specific `Documentation` should help you get started.

**Anyone interested in Stochastic Optimization.** PyEC efficiently implements the most common methods in stochastic optimization. It can be used to test and experiment how different algorithms work, and what effects different parameter settings can have.

Installation
-------------------

PyEC is distributed to PyPi and can be downloaded with **setuptools** using the name ``pyec``, like so::

   $ easy_install pyec

PyEC has dependencies on the linear algebra package `NumPy<http://numpy.scipy.org/>` and the scientific computing package `SciPy<http://www.scipy.org/>`. 


.. toctree::
   :maxdepth: 2

Basic Usage
-------------------
PyEC provides various optimization methods. If you just want to optimize, the following examples should help. PyEC provides easily accessible optimization routines for Evolutionary Annealing, Differential Evolution, Nelder-Mead, Generating Set Search, CMA-ES, Particle Swarm Optimization, and Simulated Annealing.

Start a Python terminal session as follows::

   Python 2.6.5 (r265:79359, Mar 24 2010, 01:32:55) 
   [GCC 4.0.1 (Apple Inc. build 5493)] on darwin
   Type "help", "copyright", "credits" or "license" for more information.
   >>> import pyec.optimize


To start with, we are going to optimize Branin's function. This is a real function in two dimensions. It is defined as

.. math::
   
   f(x,y) = \left(y-\frac{5.1}{4\pi^2}x^2+\frac{5}{\pi}x-6\right)^2 + 10\left(1-\frac{1}{8\pi}\right)\cos(x) + 10

and it has three global optima with :math:`f(-\pi, 12.275) = f(\pi, 2.275) = f(9.42478, 2.475) = 0.39788735`. Usually, the optimization is constrained by :math:`-5 < x < 10` and :math:`0 < y < 15`::

   >>> from numpy import *
   >>> def branin(x):
   ...    return (x[1] - (5.1/(4*(pi**2)))*(x[0]**2) + (5./pi)*x[0] - 6.)**2 + 10 * (1. - 1./(8.*pi))*cos(x[0])+10.
   ...

Here are some examples of PyEC on Branin's function with no constraints, on about 2,500 function evaluations (default)::

   >>> pyec.optimize.evolutionary_annealing(branin, dimension=2)
   (array([ 3.14159266,  2.27500002]), 0.39788735772973816)
   >>> pyec.optimize.differential_evolution(branin, dimension=2)
   (array([ 3.14100091,  2.25785439]), 0.39819904998361366)
   >>> pyec.optimize.cmaes(branin, dimension=2)
   (array([ 3.14159266,  2.27500001]), 0.39788735772973816)
   >>> pyec.optimize.nelder_mead(branin, dimension=2)
   (array([ 3.14159308,  2.27499873]), 0.39788735773149675)
   >>> pyec.optimize.generating_set_search(branin, dimension=2)
   (array([ 3.12552433,  2.29660443]), 0.39920864246465015)
   >>> pyec.optimize.particle_swarm_optimization(branin, dimension=2)
   (array([ 3.1282098 ,  2.26735578]), 0.39907497646047574)

In these examples, all we had to do was to pass our function to one of PyEC's optimizers along with the dimension (Branin's has 2 inputs variables), and these methods used default configurations in order to locate the global minimum (:func:`evolutionary_annealing` requires the ``space_scale`` parameter for unconstrained optimization). The return values are a tuple with two items. The first element is the best solution found so far, and the second element is the function value at that solution. In this case, many of the optimizers were not particularly accurate. Some methods missed the global minimum by an error on the order of 0.01, which is much larger than we would prefer, especially for this simple problem. Most of these methods can be made more accurate by replacing some of the default parameters::
   
   >>> pyec.optimize.differential_evolution(branin, dimension=2, generations=250, population=50, CR=.2, F=.2)
   (array([ 3.14159085,  2.27498739]), 0.39788735794177654)
   >>> pyec.optimize.generating_set_search(branin, dimension=2, generations=5000, expansion_factor=1., initial_step=0.5)
   (array([ 3.1386125 ,  2.27810758]), 0.3979306095569175)
   >>> pyec.optimize.particle_swarm_optimization(branin, dimension=2, omega=-0.5, phi_p=0.0)
   (array([ 3.14159258,  2.27500003]), 0.39788735772976125)

Now all but one of the methods have found the global optimum accurately up to eight decimal places. 

If we had wished to find the maximum values rather than the minimum values of the function, we could have passed the parameter ``minimize=False`` to any of these optimizers::

   >>> pyec.optimize.differential_evolution(branin, dimension=2, minimize=False)

In general, a function :math:`f(x)` can be maximized by minimizing the alternate function :math:`-f(x)` instead, which is what PyEC does internally when ``minimize=False``.

Branin's function is relatively easy to optimize; we would like to try a harder function. PyEC ships with several benchmark multimodal functions, many of which are defined in higher dimensions as well. These benchmarks can be referenced by name when calling PyEC's optimizers. One example is Rastrigin's function (see <http://en.wikipedia.org/wiki/Rastrigin_function>)::

   def rastrigin(x):
      return 10 * len(x) + ((x ** 2) - 10 * cos(2 * pi * x)).sum()  


Rastrigin's has a minimum at the origin, where its function value is 0. PyEC's optimizers can find this minimum with a little tweaking::

   >>> pyec.optimize.differential_evolution("rastrigin", dimension=10, generations=2500, population=100, CR=.2, F=.2)
   (array([ -3.09226981e-05,   2.19169568e-05,   4.46486498e-06,
            -1.50452001e-05,   6.03987807e-05,  -6.17905562e-06,
             4.82476074e-05,  -1.02580314e-05,  -2.07212921e-05,
             9.15748483e-06]), 1.6496982624403245e-06)

PyEC's optimizers are stochastic; that is, they search the space randomly. No two runs of any optimizer are the same. You can get widely varying results from different runs of the same algorithm, so it's best to run an algorithm a few times if you're not satisfied with the results::

   >>> pyec.optimize.cmaes("rastrigin", dimension=10, generations=2500,population=250)
   (array([  1.48258949e-03,   2.25335429e-04,  -5.35427662e-04,
            -2.74244483e-03,   3.20044246e-03,  -4.59549462e-03,
            -2.09654701e-03,   9.93491865e-01,   8.95951435e-04,
            -9.95219709e-01]), 1.9996060669315057)
   >>> pyec.optimize.cmaes("rastrigin", dimension=10, generations=2500,population=250)
   (array([ -4.26366209e-04,  -7.29513508e-04,   5.97365406e-04,
            -9.93842635e-01,   4.47482962e-04,  -3.32484925e-03,
            -3.98886672e-03,   4.06692711e-04,  -1.49134732e-03,
             3.80257643e-03]), 1.0041502969750979)
   >>> pyec.optimize.cmaes("rastrigin", dimension=10, generations=2500,population=250)
   (array([ -4.24651080e-04,   7.78200373e-04,   4.80037528e-03,
             1.06188871e-03,   2.50392639e-04,  -3.00255770e-03,
            -9.91998151e-01,   5.52063421e-03,  -3.44827888e-03,
            -9.97582491e-01]), 2.0081777356006967)

Every optimizer performs best on some set of problems, and performs worse on others. Here, :func:`differential_evolution` performs well on rastrigin, whereas CMA-ES is less reliable on this function. The opposite situation can hold true for other optimization problems.


Constrained Optimization
-----------------------------

PyEC supports constrained optimization as well. :class:`pyec.optimize.Hyperrectangle` boundaries can be implemented by::

   >>> pyec.optimize.evolutionary_annealing(lambda x: x[0]-x[1], dimension=2, constraint=pyec.optimize.Hyperrectangle(center=5.0,scale=2.5))
   (array([ 2.5,  7.5]), -5.0)

Other constraints can be implemented by creating a subclass of :class:`pyec.optimize.Constraint` and passing it to the optimizer via the keyword argument *constraint*.


Release Notes
-----------------------------

Version 0.3 has substantially changed the internal representation of optimizers with the following goals:

* Each optimizer should publish its configuration parameter requirements in a code-accessible form.
* Standardize configuration parameters across components.
* Better documentation of each method, especially the componential construction of optimizers in PyEC.
* Basic support for neural networks and neuroevolution.
* Algebraic operators for optimizers, e.g. ``SimpleGeneticAlgorithm = ProportionalSelection * -ProportionalSelection * OnePointCrossover * GaussianMutation[p=-0.1]``.
* New training framework derived from self-convolution
* Better support for different search spaces.
* Better support for complex constraint regions.

Future versions will offer support for features such as:

* Automatic function specific optimizer configuration
* Improved documentation

Documentation
=====================

If you're just interested in optimization, then the methods in :mod:`pyec.optimize` should be sufficient. But if you are interested in developing new stochastic iterative optimization algorithms or augmenting those provided by PyEC, then you'll need to understand the architecture of PyEC. 

Basic Concepts
---------------------

As discussed in the introduction, an optimization problem consists of a function to be optimized (the *objective function*, *fitness function*, or *cost function*) and a set of inputs to the function that are acceptable as optima (the search domain). 

In PyEC, the search domain is an instance of :class:`pyec.Space`. This object provides information about the type of the function inputs. Much work on optimization distinguishes between the search space (the set of inputs on which the objective function is defined) and the search domain (the portion of the search space that is acceptable as a solution), but PyEC does not make this distinction, so that an instance of :class:`pyec.Space` both specifies the space and as well as the feasible constraints. Theoretically, PyEC is intended to work on topological spaces, and the search domain is the topological restriction of the search space to a particular subset. Practically, when a restricted domain is required, then a subclass of the general search space provides the appropriate methods, such as determining whether a point is within the feasible region.

From a theoretical perspective, PyEC regards an optimization method as a conditional probability distribution that probabilistically generates a batch sample of possible solutions (the "population") given the previously evaluated samples and their scores on the objective function. All optimization methods inherit from :class:`pyec.distribution.basic.PopulationDistribution`. 

Configuration
--------------------

Most optimization methods are depend on various parameters for their optimization. In PyEC, these parameters are wrapped in a :class:`pyec.Config` object. Every optimization method in PyEC provides its own defaults on a class variable named ``config``. For example, the default :class:`DifferentialEvolution` has a crossover rate of 0.2 (``CR = 0.2``) by default. PyEC provides a means to modify these configuration parameters before instantiation using the `[]` operator, e.g.::

   >>> from pyec.config import Config
   >>> from pyec.distribution.de import DifferentialEvolution
   >>> DifferentialEvolution.config.CR
   0.2
   >>> DE_CR_05 = DifferentialEvolution[Config(CR=0.5)]
   >>> DE_CR_05.config.CR
   0.5

In this way, it is easy to configure algorithms to run as needed. Under the hood, ``DE_CR_05`` is a new class with its own :class:`Config` object, distinct from the ``config`` property of :class:`DifferentialEvolution`. These are both in turn distinct from the config on :class:`PopulationDistribution`. So, for example::

   >>> PopulationDistribution.config.__properties__
   {'stats': <pyec.util.RunStats.RunStats object at 0x10375b610>, 
    'observer': None, 'space': None, 'minimize': True, 'populationSize': 1, 
    'history': <class 'pyec.history.SortedMarkovHistory'>}
   >>> DifferentialEvolution.config.__properties__
   {'F': 0.5, 'space': <pyec.space.Euclidean object at 0x10375bdd0>, 
    'initial': None, 'CR': 0.2, 'populationSize': 100, 
    'history': <class 'pyec.history.LocalBestHistory'>}
   >>> DE_CR_05.config.__properties__
   {'CR': 0.5} 

In this case, ``DE_CR_05`` is a subclass of :class:`DifferentialEvolution`, which is a subclass of :class:`PopulationDistribution`. So what happens when an operator is instantiated?::

   >>> DifferentialEvolution().config.__properties__
   {'F': 0.5, 'stats': <pyec.util.RunStats.RunStats object at 0x10375b610>, 
    'observer': None, 'space': <pyec.space.Euclidean object at 0x10375bdd0>, 
    'minimize': True, 'initial': None, 'CR': 0.2, 'populationSize': 100, 
    'history': <class 'pyec.history.LocalBestHistory'>}
   >>> DE_CR_05().config.__properties__
   {'F': 0.5, 'stats': <pyec.util.RunStats.RunStats object at 0x10375b610>, 
    'observer': None, 'space': <pyec.space.Euclidean object at 0x10375bdd0>, 
    'minimize': True, 'initial': None, 'CR': 0.5, 'populationSize': 100, 
    'history': <class 'pyec.history.LocalBestHistory'>}

Once the algorithm is instantiated, it has its own ``config``, and this ``config`` has properties from each of its parent classes. The instance `config` takes on its default value from the lowest parent class in the chain that defines it. So, ``CR=0.5`` in ``DE_CR_05`` overrides ``CR=0.2`` in :class:`DifferentialEvolution`. 

Running an Optimizer
---------------------
Once an optimizer has been configured, it can be run. Running an iterative optimizer means that the optimizer is used to generate a sequence of samples (populations) in the search domain. The optimizer uses the results of previous samples and their scores in order to generate the next sample. 

Operationally, PyEC separates the state of an optimization run from the logic used to generate new points for evaluation. The state of a particular optimization run is maintained in a :class:`History` object. The :class:`History` object should contain sufficient information for the algorithm to regenerate its state, and the :class:`PopulationDistribution` defining the optimizer should not contain any state outside the :class:`History` object that cannot be rebuilt from the :class:`History`. Each  algorithm must specify the :class:`History` class that it normally uses in its ``config`` object. Additionally, each optimizer may override the method ``compatible`` that is used to check whether a :class:`History` instance is compatible with the optimizer. The default history is a :class:`MarkovHistory`, which stores only the last set of samples and their scores. :class:`MarkovHistory` is sufficient to implement many genetic algorithms. 

A PyEC optimizer is first updated with a history and a fitness function, and then sampled to select new evaluation points to test. These two steps are repeated indefinitely, so that the optimizer is run like so::

   >>> de = DifferentialEvolution()
   >>> f = lambda x: sum(x ** 2)
   >>> pop = de[None, f]()
   >>> history = de.history
   >>> for i in xrange(100):
   ...    history.update(pop, f, space, de)
   ...    pop = de[history, f]()
   ...
   >>> history.minimal()
   (array([  8.33013471e-24]), 6.9391144322723016e-47)
   
In this code, an instance of DifferentialEvolution is created to minimize a basic parabola in one dimension. A blank history of the default class is created by passing ``None`` along with the objective function into the ``[]`` operator, and then the optimizer is immediately called to generate the initial population. The ``history`` is extracted, and then updated with the latest population, the objective function, the search domain, and the optimizer. The actual function evaluation (if necessary) occurs in the call to ``history.update``. Finally, the optimizer is updated with the history, and a new population is generated. These last two steps are repeated, and at the end, the minimal point observed during the process is extracted by a call to ``history.minimal``.

There is a simpler way to run an optimizer, like so::

   >>> de100 = (DifferentialEvolution << 100)()
   >>> lastPop = de100[None, (lambda x: sum(x**2))]()
   >>> de100.history.minimal()
   (array([  1.77413093e-23]), 3.1475405527093268e-46)

In this second example, the notation ``DifferentialEvolution << 100`` represents an operation called *self-convolution*, which is equivalent to running :class:`DifferentialEvolution` for 100 generations and then return the final sample. Self-convolution is an operation on optimization algorithms. Behind the scenes, the operation ``<<`` produces a new optimizer class that inherits from :class:`SelfConvolution`. PyEC offers several operations for creating new operators beyond self-convolution. In fact, PyEC provides an algebra for optimizers, discussed next.

Optimizer Algebra
------------------ 

Viewed from a certain perspective, the space of all possible optimizers is a normed vector space, similar to the space of real numbers or the space of continuous functions (see Lockett and Miikkulainen, "Measure-Theoretic Analysis of Stochastic Optimization", 2013). Thus, one can meaningfully speak of adding optimizers, or multiplying them by scalar values. There are also other operations that are useful.

PyEC support several optimizer operations at both the class and the instance level. The main operations currently implemented are: (1) addition, (2) scalar multiplication, (3) convolution, (4) trajectory truncation, and (5) population splitting.

Addition of two optimizer indicates a probabilistic choice. Thus if we have $\mathcal{A} + \mathcal{B}$, the result is a pseudo-optimizer that uses $\mathcal{A}$ to produce the next generation/sample half the time, and uses $\mathcal{B}$ the other half of the time. This was described as a "pseudo-optimizer" because the result is not a probability distribution. To get a probability distribution, we have to normalize the coefficients, that is, $1 + 1 = 2$, so we choose each optimizer with probability $\frac{1}[2}$. PyEC keeps track of additive coefficients and normalizes automatically, so that every possible addition is a real optimizer.

Scalar multiplication simply changes the coefficients. If there is only one optimizer involved, this has no effect. But if there are multiple optimizers, then the coefficients express the frequency with which each optimizer is used to generate a distribution. PyEC makes one choice per generation. Note that this is different from the formal theory of Lockett and Miikkulainen, which allows one choice per individual.

So::

   >>> from pyec.distribution.de import DifferentialEvolution
   >>> from pyec.distribution.cmaes import Cmaes
   >>> DE_Plus_CMAES = DifferentialEvolution + Cmaes
   <class 'pyec.distribution.convex.DifferentialEvolution_add_qqqCmaesqqq'>
   >>> Half_DE_Plus_Half_CMAES = .5 * DifferentialEvolution + .5 * Cmaes
   <class 'pyec.distribution.convex.DifferentialEvolution_mult_qqq0_5qqq_add_qqqCmaes_mult_qqq0_5qqqqqq'>
   >>> import pyec.optimize
   >>> sum([pyec.optimize.optimize(DE_Plus_CMAES, lambda x: -(x**2).sum())[1] for i in xrange(10)])/10
   2.3966419053038709e-20
   >>> sum([pyec.optimize.optimize(Half_DE_Plus_Half_CMAES, lambda x: -(x**2).sum())[1] for i in xrange(10)])/10
   2.898181656813492e-20   

If, rather than probabilistically choosing one optimizer or the other to generate the population, one wishes instead to have a certain percentage of the next population generated by a particular algorithm, then one can use the ``|`` operator to split the population::

   >>> split_Half_DE_Half_CMAES = .5 * DifferentialEvolution | .5 * Cmaes
   >>> sum([pyec.optimize.optimize(split_Half_DE_Half_CMAES, lambda x: -(x**2).sum())[1] for i in xrange(10)])/10

An important concept for the theory of genetic algorithms is the convolution operator. A convolution of optimizers is a process where a population is chosen first according to one optimizer, and then chosen population is fed into a second optimizer as part of its history. Genetic algorithms can be defined using convolutions, and this is done in PyEC using the "<<" for convolution, e.g.::

   SimpleGeneticAlgorithm = (
      Proportional << ((Proportional >> 1) <<
                        Crossover[_(crosser=OnePointDualCrosser)])
      << Bernoulli
   )[_(space=BinaryReal(realDim=5))] 

When the second argument to ``<<`` is an integer, then self-convolution is implied, and the optimizer is convolved with itself the specified number of times, as shown above. 

Trajectory truncation is denoted by ``>>``. If the second argument is an integer, the specified number of populations are truncated off of the population history. If the second argument is an optimizer, then the first optimizer is convolved with the second, but the second optimizer is trajectory truncated by one. Thus the following two are equivalent::

    >>> OptimizerA >> OptimizerB
    >>> OptimizerA << (OptimizerB >> 1)

Each operation creates a new class. These classes are assigned a name that matches the operations that generated them. PyEC tries to reuse classes where possible.

In general, combining optimizers with these operations will not automatically create good optimizers, although in the case of addition and scalar multiplication, the performance change can be smooth as the coefficients are varied (see Lockett, "Measure-Theoretic Analysis of Performance in Evolutionary Algorithms"). However, these constructive operations can save a lot of time on coding and testing if used properly to implement new optimizers.


Optimization
------------------

.. automodule:: pyec.optimize
   :members:


Basics
------------------

.. automodule:: pyec
   :members:

Configuration
++++++++++++++++++++

.. automodule:: pyec.config
   :members:

Trainer
++++++++++++++++++++
.. automodule:: pyec.trainer
   :members:

Distributions
------------------

.. automodule:: pyec.distribution
   :members:

.. automodule:: pyec.distribution.basic
   :members:

Evolutionary Computation
++++++++++++++++++++++++++++++++

.. automodule:: pyec.distribution.ec
   :members:

.. automodule:: pyec.distribution.ec.ga
   :members:

.. automodule:: pyec.distribution.ec.es
   :members:

.. automodule:: pyec.distribution.ec.evoanneal
   :members:

Evolutionary Operators
++++++++++++++++++++++++++

.. automodule:: pyec.distribution.ec.selectors
   :members:

.. automodule:: pyec.distribution.ec.mutators
   :members:

Natural Computation
+++++++++++++++++++++++++++

.. automodule:: pyec.distribution.de
   :members:

.. automodule:: pyec.distribution.cmaes
   :members:

.. automodule:: pyec.distribution.pso
   :members:

.. automodule:: pyec.distribution.eda
   :members:

Direct Search
+++++++++++++++++++++++++

.. automodule:: pyec.distribution.neldermead
   :members:

.. automodule:: pyec.distribution.gss
   :members:

Bayesian Networks
+++++++++++++++++++++++++++

.. automodule:: pyec.distribution.bayes
   :members:

.. automodule:: pyec.distribution.bayes.net
   :members:

.. automodule:: pyec.distribution.bayes.variables
   :members:

.. automodule:: pyec.distribution.bayes.sample
   :members:

.. automodule:: pyec.distribution.bayes.score
   :members:

.. automodule:: pyec.distribution.bayes.structure
   :members:

.. automodule:: pyec.distribution.bayes.structure.greedy
   :members:

.. automodule:: pyec.distribution.bayes.structure.proposal
   :members:


Boltzmann Machines
+++++++++++++++++++++++

.. automodule:: pyec.distribution.boltzmann
   :members:

.. automodule:: pyec.distribution.boltzmann.bm
   :members:

.. automodule:: pyec.distribution.boltzmann.rbm
   :members:

.. automodule:: pyec.distribution.boltzmann.sample
   :members:


Utilities
------------------

.. automodule:: pyec.util
   :members:

.. automodule:: pyec.util.TernaryString
   :members:

.. automodule:: pyec.util.cache
   :members:

.. automodule:: pyec.util.partitions
   :members:

.. automodule:: pyec.util.registry
   :members:

.. automodule:: pyec.util.benchmark
   :members:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

