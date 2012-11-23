"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
from pyec.util.cache import LRUCache

class History(object):
    """A History used to track the progress of an optimization algorithm.
    
    Different algorithms should extend this class in order to define
    the minimal amount of history that needs to be stored in order for the
    algorthm to operate.
    
    """
    sorted = False
    useCache = True
    
    def __init__(self):
        super(History, self).__init__()
        self.evals = 0
        self.minSolution = None
        self.minScore = 1e300
        self.maxSolution = None
        self.maxScore = -1e300 
        self._empty = True
        self.cache = LRUCache() # 10,000 items by default
        self.updates = 0
        self.printEvery = 1000000000000L #how often to print generation report
    
    def empty(self):
        return self._empty
    
    def minimal(self):
        """Get the minimal solution and its score.
        
        :returns: A tuple of two item with the solution object first
                  and the score of that item second 
                  
        """
        return self.minSolution, self.minScore
        
    def maximal(self):
        """Get the maximal solution and its score.
        
        :returns: A tuple of two item with the solution object first
                  and the score of that item second 
                  
        """
        return self.maxSolution, self.maxScore
        
    def num_evaluations(self):
        """Get the number of function evaluations performed in this 
        history object.
        
        :returns: An integer representing the number of times the fitness
                  function or objective has been evaluated
                  
        """
        return self.evals
        
    def update(self, population, fitness, space):
        """
         Update the state of the :class:`History` with the latest population 
         and its fitness scores. Subclasses should probably override
         ``internalUpdate`` rather than ``update``, unless they want to
         change how the min/max are tracked.
         
         Returns the history for use in continuations.
         
         If ``population`` is ``None``, then this method does nothing; this
         is so that you can set up a loop like to run an optimizer like::
         
         p = None
         f = lambda x: 
         t = History()
         s = some_optimizer.config.space
         for i in xrange(generations):
             p = some_optimizer[t.update(p,f,s), f]()
         t.update(p,f,s)
          
         :params population: The previous population.
         :type population: list of points in the search domain
         :params fitness: The fitness / cost / objective function
         :type fitness: Any callable object
         :params space: The search domain
         :type space: :class:`Space`
         :returns: The history (``self``), for continuations
         
        """
        if population is None:
            return
        
        self._empty = False
        self.evals += len(population)
        self.updates += 1
        
        # score the sample
        pop  = population
        scored = [(x, self.score(x, fitness, space)) for x in pop]
        
        for x,s in scored:
            if s > self.maxScore:
                 if (self.maxSolution is not None and 
                     self.maxSolution is not self.minSolution):
                     del self.maxSolution
                 self.maxScore = s
                 self.maxSolution = x
               
            if s < self.minScore:
                 if (self.minSolution is not None and 
                     self.minSolution is not self.maxSolution):
                     del self.minSolution
                 self.minScore = s
                 self.minSolution = x
        
        if not (self.updates % self.printEvery):
           genmin = min([s for x,s in scored])
           genmax = max([s for x,s in scored])
           genavg = np.average([s for x,s in scored])
           print self.updates, ": min", self.minScore, " max", self.maxScore,
           print " this generation (min, avg, max): ", genmin, genavg, genmax    
               
        self.internalUpdate(scored)
        
        return self
        
    def internalUpdate(self, population):
        """
         Update the state of the :class:`History` with the latest population 
         and its fitness scores. This is an internal call intended for
         overridden by subclasses. One of the important functions is to
         delete points no longer needed by the history.
          
         :params population: The previous population with its fitness scores. 
         :type population: list of (point, score) tuples
         
        """    
        
        for x,s in population:
            if x is not self.minSolution and x is not self.maxSolution:
                del x

    def score(self, point, fitness, space):
        """Get the fitness score, caching where possible.
      
         :param point: A valid point in the space
         :type point: Must match ``space.type``
         :param fitness: The fitness function
         :type fitness: Any callable
         :param space: The space to which the point belongs
         :type space: :class:`Space`
         :returns: The fitness value, cached if possible
         
        """
   
        if fitness is None: 
            return None
            
        if self.useCache: 
            try:
                hashed = space.hash(point)
                if self.cache.has_key(hashed):
                    return self.cache[hashed]
            except Exception:
                pass
        
        if not space.in_bounds(point):
           # use NaN so that the result is less than nor greater than
           # any other score, and therefore NEVER optimal
           s = 1e400 / 1e400
        else:
           s = fitness(space.convert(point))
        
        if self.useCache:
            try:
                hashed = space.hash(point)
                self.cache[hashed] = s
            except Exception:
                pass
        
        return s


class MarkovHistory(History):
     """A :class:`History` that stores the last population only."""
     
     def __init__(self):
         super(MarkovHistory, self).__init__()
         self.population = None
         
     def internalUpdate(self, population):
         """Overrides ``internalUpdate`` in :class:`History`"""
         if self.population is not None:
             for x,s in self.population:
                 if x is not self.minSolution and x is not self.maxSolution:
                     del x
         
         self.population = population
         
     def lastPopulation(self):
         return self.population
        
        
class SortedMarkovHistory(History):
    """A :class:`History` that stores the last population only, sorted.
       Default sorting is by score from least to greatest, for minimization.
       To sort differently, set the ``history`` in :class:`Config` to a 
       generator that provides a different sorter, e.g.
       
       ``config.history = lambda h: SortedMarkovHistory(lambda p: -p[1])``
       
       :params sorter: A comparator passed to the ``key`` argument of the 
                       built-in function ``sorted``.
       :type sorter: A callable object with one argument, passed a tuple
                     of type ``(solution,score)``.
                     
    """
    sorted = True
    
    def __init__(self, sorter=None):
        super(SortedMarkovHistory, self).__init__()
        self.sorter = lambda p: p[1]
        
    def internalUpdate(self, population):
        """Overrides ``internalUpdate`` in :class:`History`"""
        super(SortedMarkovHistory, self).internalUpdate(population)
         
        self.population = sorted(self.population, key=self.sorter)

class LocalMinimumHistory(History):
     """A :class:`History` that stores the best members of the
     population at each index. Best is assumed to mean fitness-minimal.
     
     See :class:`LocalMaximumHistory` for maximization
     
     """
     
     def __init__(self):
         super(LocalMinimumHistory, self).__init__()
         self.localBestPop = None
         
     def internalUpdate(self, population):
         """Overrides ``internalUpdate`` in :class:`History`"""
         if self.localBestPop is None:
             self.localBestPop = population
         else:
             idx = 0
             for y,s2 in population:
                 x,s = self.localBestPop[idx]
                 if s is None or self.better(s2, s):
                     self.localBestPop[idx] = y,s2
                     if (x is not self.minSolution and 
                         x is not self.maxSolution):
                         del x
                 else:
                     
                     if (y is not self.minSolution and 
                         y is not self.maxSolution):
                         del y
                 idx += 1     
         
     def localBest(self):
         return self.localBestPop

     def better(self, x, y):
         """Determine which of two scores is better. Default is less than.
         
         :param x: The first score
         :type x: ``float``
         :param y: The second score
         :type y: ``float``
         :returns: ``bool`` -- ``True`` if ``x < y``, ``False`` otherwise
         
         """
         return x < y

class LocalMaximumHistory(LocalMinimumHistory):
     """A :class:`History` that stores the maximal members of the
     population at each index."""
     
     def better(self, x, y):
         """Determine which of two scores is better. For this history,
         higher is better.
         
         :param x: The first score
         :type x: ``float``
         :param y: The second score
         :type y: ``float``
         :returns: ``bool`` -- ``True`` if ``x > y``, ``False`` otherwise
         
         """
         return x > y


class CheckpointedHistory(History):
    """A :class:`History` suitable for use in a self-convolution.
    
    Keeps a stack of states for a history; push with ``checkpoint``
    and roll back with ``pop`` 
    
       :param historyClass: A constructors for :class:`History` objects
       :type historyClass: :class:`History` constructor
        
    """
    def __init__(self, historyClass):
        super(CheckpointedHistory, self).__init__()
        self.history = historyClass() 
        self.states = []
        
        
    def internalUpdate(self, population):
        """Overrides ``internalUpdate`` in :class:`History`"""
        self.history.internalUpdate(population)
    
    def checkpoint(self):
        self.states.append = self.history.__getstate__()
        
    def rollback(self):
        if not len(self.states):
            raise ValueError("Cannot rollback initial state!")
        state = self.states[-1]
        self.history.__setstate__(state)
        self.states = self.states[:-1]


class MultipleHistory(History):
    """A :class:`History` suitable for use in a convex optimizer.
    Keeps multiple histories and updates them all.
    
       :param historyClasses: A list of constructors for :class:`History`
                              objects
       :type historyClasses: list of :class:`History` constructors
        
    """
    def __init__(self, *historyClasses):
        super(MultipleHistory, self).__init__()
        if not len(historyClasses):
            err = "MultipleHistory needs at least 1 history"
            raise ValueError(err)
        self.histories = [h() for h in set(historyClasses)]
    
    def internalUpdate(self, population):
        """Overrides ``internalUpdate`` in :class:`History`"""
        for h in self.histories:
            h.internalUpdate(population)


class CheckpointedMultipleHistory(MultipleHistory):
    """A :class:`History` suitable for use in a convolution.
    
    Keeps a stack of states for each subhistory; push with ``checkpoint``
    and roll back with ``pop`` 
    
       :param historyClasses: A list of constructors for :class:`History`
                              objects
       :type historyClasses: list of :class:`History` constructors
        
    """
    def __init__(self, *historyClasses):
        super(CheckpointedMultipleHistory, self).__init__(*historyClasses)
        self.states = []
    
    def checkpoint(self):
        self.states.append = [h.__getstate__() for h in self.histories]
        
    def rollback(self):
        if not len(self.states):
            raise ValueError("Cannot rollback initial state!")
        states = self.states[-1]
        for h,st in zip(self.histories, states):
            h.__setstate__(st)
        self.states = self.states[:-1]
        
        
class DelayedHistory(History):
    """A :class:`History` for trajectory truncation.
    
    Keeps a subordinate history, and updates it with a lag from a queue.
    
    :param history: A :class:`History` to be delayed
    :type history: :class:`History`
    :param delay: The number of steps to delay
    :type delay: ``int``
    
    """
    def __init__(self, history, delay=1):
        super(DelayedHistory, self).__init__()
        self.history = history
        if delay < 1:
            err = "Delay must be a positive integer, not {0}".format(delay)
            raise ValueError(err)
        self.delay = delay
        self.queue = []
        
    def update(self, population):
        """Stack up the new population, and push the delayed 
        populations to the subordinate history.
        
        """
        if len(self.queue) == self.delay:
            self.history.update(self.queue[-1])
            self.queue = self.queue[:-1]
    
        self.queue.append(population)
    
    