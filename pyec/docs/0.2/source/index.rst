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

Version 0.2 has changed substantially from version 0.1. Here are some of the changes:

* Thinned out experimental code
* Added more documentation
* Replaced the 1996 (obsolete) version of CMA-ES with the modern version
* Improved evolutionary annealing; added :class:`pyec.distributions.ec.selectors.TournamentAnnealing`
* Removed dependency on Django
* Removed database support for algorithm results
* Added :mod:`pyec.optimize` to provide easy access to optimization routines
* Removed numerous command line scripts
* Improved support for Bayesian networks and greedy structure searching
* Support for Bayesian structure search with Evolutionary Annealing and Simulated Annealing

The following changes are planned for Version 0.3:

* Each optimizer should publish its configuration parameter requirements in a code-accessible form.
* Standardize configuration parameters across components.
* Better documentation of each method, especially the componential construction of optimizers in PyEC.
* Basic support for neural networks and neuroevolution.
* Algebraic operators for optimizers, e.g. ``SimpleGeneticAlgorithm = ProportionalSelection * -ProportionalSelection * OnePointCrossover * GaussianMutation[p=-0.1]``.
* New training framework derived from self-convolution
* Better support for different search spaces.
* Better support for complex constraint regions.
* Automatic function-specific configuration of some parameters for the main algorithms. 

Release 0.2 is a transitional release, and substantial changes will be made to the internal components for the next few releases.

Documentation
=====================


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

