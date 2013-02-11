"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import copy
import numpy as np
from pyec.util.cache import LRUCache

class History(object):
    """A History used to track the progress of an optimization algorithm.
    
    Different algorithms should extend this class in order to define
    the minimal amount of history that needs to be stored in order for the
    algorthm to operate.
    
    """
    useCache = True
    attrs = set()
    sorted = False
    root = True # whether the current history is the top level root for
                # all histories of this algorithm
    
    def __init__(self, config):
        super(History, self).__init__()
        self.config = config
        self.evals = 0
        self.minSolution = None
        self.minScore = np.inf
        self.maxSolution = None
        self.maxScore = -np.inf 
        self._empty = True
        if not hasattr(self, 'cache'):
            self.cache = LRUCache() # 10,000 items by default
        self.updates = 0
        #how often to print generation report
        self.printEvery = config.printEvery or 1000000000000L 
        self.attrs = set(["evals","minSolution","minScore","maxScore","attrs",
                          "minSolution", "maxSolution","_empty","cache",
                          "updates","printEvery", "useCache"])
    
    def __getstate__(self):
        """Used by :class:`CheckpointedHistory`
        and :class:`CheckpointedMultipleHistory` to checkpoint a history
        so it can be rolled back after updates.
        
        Should return all objects in the history in a dictionary, sensitive
        to the fact that object references may need to be copied.
        
        :returns: A dictionary with the state of the history
        
        """
        state = {}
      
        for attr in self.attrs:
            val = getattr(self, attr)
            if isinstance(val, list):
                val = [x for x in val]
            state[attr] = val
         
        state['cfg'] = self.config.__properties__
        return state

    def __setstate__(self, state):
        for attr in self.attrs:
            val = state[attr]
            setattr(self, attr, val)
         
        import pyec.config
        self.config = pyec.config.Config(**state['cfg'])  
    
    def empty(self):
        """Whether the history has been used or not."""
        return self._empty
    
    def better(self, score1, score2):
        """Return whether one score is better than another.
        
        Uses ``config.minimize`` to decide whether lesser or greater
        numbers are better.
        
        :param score1: the score in question (floating point number)
        :type score1: ``float``
        :param score2: the score being compared
        :type score2: ``float``
        :returns: whether ``score1`` is better than ``score2``
        """
        if self.config.minimize:
            return score1 < score2
        else:
            return score1 > score2
    
    def best(self):
        """Get the best solution, whether minimizing or maximizing.
        
        Same as ``optimal``
        
        """
        if self.config.minimize:
            return self.minimal()
        else:
            return self.maximal()
    
    optimal = best
    
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
        
    def update(self, population, fitness, space, opt):
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
         o = some_optimizer
         s = o.config.space
         for i in xrange(generations):
             p = some_optimizer[t.update(p,f,s,0), f]()
         t.update(p,f,s,o)
          
         :params population: The previous population.
         :type population: list of points in the search domain
         :params fitness: The fitness / cost / objective function
         :type fitness: Any callable object
         :params space: The search domain
         :type space: :class:`Space`
         :params opt: The optimizer reporting this population
         :type opt: :class:`PopulationDistribution`
         :returns: The history (``self``), for continuations
         
        """
        if population is None:
            return
        
        #self.config.stats.start(repr(self) + "history.update.all")
        self._empty = False
        self.evals += len(population)
        self.updates += 1
        
        #self.config.stats.start(repr(self) + "history.update.scoreall")
        # score the sample
        pop  = population
        scored = [(x, self.score(x, fitness, space)) for x in pop]
        #self.config.stats.stop(repr(self) + "history.update.scoreall")
        #self.config.stats.start(repr(self) + "history.update.findbest")
        
        if self.root and self.config.observer is not None:
            self.config.observer.report(opt, scored)
        
        for x,s in scored:
            if s > self.maxScore:
                 self.maxScore = s
                 self.maxSolution = x
               
            if s < self.minScore:
                 self.minScore = s
                 self.minSolution = x
        #self.config.stats.stop(repr(self) + "history.update.findbest")
        
        if not (self.updates % self.printEvery):
            genmin = min([s for x,s in scored])
            genmax = max([s for x,s in scored])
            genavg = np.average([s for x,s in scored])
            print self.updates, ": min", self.minScore, " max", self.maxScore,
            print " this generation (min, avg, max): ", genmin, genavg, genmax
        
        #self.config.stats.start(repr(self) + "history.update.internal")       
        self.internalUpdate(scored)
        #self.config.stats.stop(repr(self) + "history.update.internal")
        #self.config.stats.stop(repr(self) + "history.update.all")
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
        pass

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
        
        #self.config.stats.start("history.score")
            
        if self.useCache: 
            try:
                hashed = space.hash(point)
                if self.cache.has_key(hashed):
                    ret = self.cache[hashed]
                    #self.config.stats.stop("history.score")
                    return ret
            except Exception:
                pass
        
        if not space.in_bounds(point):
            # use NaN so that the result is less than nor greater than
            # any other score, and therefore NEVER optimal
            s = np.inf - np.inf
        else:
            try:
                s = fitness(space.convert(point))
            except ValueError:
                s = np.inf - np.inf
        
        if self.useCache:
            try:
                hashed = space.hash(point)
                self.cache[hashed] = s
            except Exception:
                pass
        #self.config.stats.stop("history.score")
        return s
    
    def setCache(self, cache):
        self.cache = cache


class MarkovHistory(History):
     """A :class:`History` that stores the last population only."""
     
     def __init__(self, config):
         super(MarkovHistory, self).__init__(config)
         self.population = None
         self.acceptanceRate = None
         self.tempAcceptanceRate = None # intentionally not in attrs; transient
         self.attrs |= set(["population", "acceptanceRate"])
         
     def internalUpdate(self, population):
         """Overrides ``internalUpdate`` in :class:`History`"""
         self.population = population
         if self.tempAcceptanceRate is not None:
            if self.acceptanceRate is None:
                self.acceptanceRate = self.tempAcceptanceRate
            else:
                self.acceptanceRate *= (self.updates - 1.0) / self.updates
                self.acceptanceRate += self.tempAcceptanceRate / self.updates
         
     def lastPopulation(self):
         return self.population
        
     def reportAcceptance(self, newRate):
         """Record the current acceptance rate, to be added into the existing
         acceptance rate on update. Used by simulated annealing, evolution
         strategies to update a proposal distribution.
         
         :param newRate: The lastest acceptance percentange
         :type newRate: ``float``
         
         """
         self.tempAcceptanceRate = newRate
     
class DoubleMarkovHistory(MarkovHistory):
     """Like :class:`MarkovHistory`, but stores the last two populations.
     
     """
     def __init__(self, config):
         super(DoubleMarkovHistory, self).__init__(config)
         self._penultimate = None
         self.attrs |= set(["_penultimate"])
         
     def internalUpdate(self, population):
         self._penultimate = self.population
         super(DoubleMarkovHistory, self).internalUpdate(population)
         
     def penultimate(self):
         return self._penultimate
     
        
class MultiStepMarkovHistory(History):
    """A :class:`History` that stores a small fixed number of prior populations.
    
    """
    def __init__(self, config):
        super(MultiStepMarkovHistory, self).__init__(config)
        self.populations = []
        self.attrs |= set(["populations"])
        
    def internalUpdate(self, population):
        self.populations.append(population)
        if len(self.populations) > self.config.order:
            self.populations = self.populations[-self.config.order:]
    
    def order(self):
        return self.config.order
     
        
class SortedMarkovHistory(MarkovHistory):
    """A :class:`History` that stores the last population only, sorted.
       Default sorting is by score from least to greatest, for minimization.
       To sort differently, set the ``history`` in :class:`Config` to a 
       generator that provides a different sorter, e.g.
       
       ``config.history = lambda h: SortedMarkovHistory(lambda p: -p[1])``
                     
    """
    sorted = True
    
    def __init__(self, config):
        super(SortedMarkovHistory, self).__init__(config)
        
    def sorter(self, x):
        """
        Sorting routine for this history, passed as key to
        python ``sorted`` built-in. Will be pass a ``(solution, score)``
        tuple.
        
        Uses ``config.minimize`` to determine whether to sort low to high
        or high to low
        
        """
        if self.config.minimize:
            return x[1] or np.inf
        else:
            return -(x[1] or np.inf)
    
    def internalUpdate(self, population):
        """Overrides ``internalUpdate`` in :class:`History`"""
        super(SortedMarkovHistory, self).internalUpdate(population)
         
        self.population = sorted(self.population, key=self.sorter)
        

class LocalBestHistory(History):
     """A :class:`History` that stores the best members of the
     population at each index. Best is interpreted according to the
     value of ``config.minimize``.
     
     """
     
     def __init__(self, config):
         super(LocalBestHistory, self).__init__(config)
         self.localBestPop = None
         self.attrs |= set(["localBestPop"])
         
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
                 idx += 1     
         
     def localBest(self):
         return self.localBestPop


class CheckpointedHistory(History):
    """A :class:`History` suitable for use in a self-convolution.
    
    Keeps a stack of states for a history; push with ``checkpoint``
    and roll back with ``pop`` 
    
       :param historyClass: A constructors for :class:`History` objects
       :type historyClass: :class:`History` constructor
        
    """
    def __init__(self, config, historyClass):
        super(CheckpointedHistory, self).__init__(config)
        self.history = historyClass(self.config)
        self.history.root = False
        self.history.printEvery = 100000000L
        self.cache = self.history.cache
        self.useCache = self.history.useCache
        self.states = []
        self.attrs |= set(["states"])
     
    def setCache(self, cache):
        self.cache = cache
        self.history.setCache(cache)
        
    def internalUpdate(self, population):
        """Overrides ``internalUpdate`` in :class:`History`"""
        pass
        
    def update(self, population, fitness, space, opt):
        if population is None:
            return self
        super(CheckpointedHistory, self).update(population, fitness, space, opt)
        self.history.update(population, fitness, space, opt)
        return self
    
    def checkpoint(self):
        self.states.append = self.history.__getstate__()
        
    def rollback(self):
        if not len(self.states):
            raise ValueError("Cannot rollback initial state!")
        state = self.states[-1]
        self.history.__setstate__(state)
        self.evals = self.history.evals
        self._empty = self.history._empty
        self.updates = self.history.updates
        self.states = self.states[:-1]
        self.attrs |= set(["states"])

    def __getstate__(self):
        start = super(CheckpointedHistory, self).__getstate__()
        start.update({"states":self.states,
                      "history":self.history.__getstate__()})
        return start
        
    def __setstate__(self, state):
        self.history.__setstate__(state["history"])
        del state["history"]
        super(CheckpointedHistory, self).__setstate__(state)


class MultipleHistory(History):
    """A :class:`History` suitable for use in a convex optimizer.
    Keeps multiple histories and updates them all.
    
       :param historyClasses: A list of constructors for :class:`History`
                              objects
       :type historyClasses: list of :class:`History` constructors
        
    """
    def __init__(self, config, *historyClasses):
        super(MultipleHistory, self).__init__(config)
        if not len(historyClasses):
            err = "MultipleHistory needs at least 1 history"
            raise ValueError(err)
        self.histories = [h(self.config) for h in set(historyClasses)]
        useCache = False
        for h in self.histories:
            h.root = False
            h.printEvery = 1000000000L
            h.setCache(self.cache)
            useCache = useCache or h.useCache
        self.useCache = useCache
        
    def setCache(self, cache):
        self.cache = cache
        for h in self.histories:
            h.setCache(self.cache)
    
    def update(self, population, fitness, space, opt):
        """Overrides ``internalUpdate`` in :class:`History`"""
        if population is None:
            return self
        super(MultipleHistory, self).update(population, fitness, space, opt)
        for h in self.histories:
            h.update(population, fitness, space, opt)
        return self

    def internalUpdate(self, population):
        pass

    def __getstate__(self):
        start = super(MultipleHistory, self).__getstate__()
        histories = [h.__getstate__() for h in self.histories]
        start.update({"histories":histories})
        return start
        
    def __setstate__(self, state):
        for st,h in zip(state["histories"], self.histories):
            h.__setstate__(st)
        super(MultipleHistory, self).__setstate__(state)
    

class CheckpointedMultipleHistory(MultipleHistory):
    """A :class:`History` suitable for use in a convolution.
    
    Keeps a stack of states for each subhistory; push with ``checkpoint``
    and roll back with ``pop`` 
    
       :param history: A list of constructors for :class:`History`
                              objects
       :type history: list of :class:`History` constructors
        
    """
    def __init__(self, config, *historyClasses):
        super(CheckpointedMultipleHistory, self).__init__(config,
                                                          *historyClasses)
        self.states = []
        self.attrs |= set(["states"])
    
    def checkpoint(self):
        self.states.append([h.__getstate__() for h in self.histories])
        assert len(self.states) <= 2
        
    def rollback(self):
        if not len(self.states):
            raise ValueError("Cannot rollback initial state!")
        states = self.states[-1]
        for h,st in zip(self.histories, states):
            h.__setstate__(st)
        self.evals = self.histories[0].evals
        self._empty = self.histories[0]._empty
        self.updates = self.histories[0].updates
        self.states = self.states[:-1]
        
        
class DelayedHistory(History):
    """A :class:`History` for trajectory truncation.
    
    Keeps a subordinate history, and updates it with a lag from a queue.
    
    :param history: A :class:`History` to be delayed
    :type history: :class:`History`
    :param delay: The number of steps to delay
    :type delay: ``int``
    
    """
    def __init__(self, config, history, delay=1):
        super(DelayedHistory, self).__init__(config)
        self.history = history
        self.history.root = False
        self.history.printEvery = 100000000L
        if delay < 1:
            err = "Delay must be a positive integer, not {0}".format(delay)
            raise ValueError(err)
        self.delay = delay
        self.cache = self.history.cache
        self.useCache = self.history.useCache
        self.queue = []
    
    def setCache(self, cache):
        self.cache = cache
        self.history.setCache(cache)

    def update(self, population, fitness, space, opt):
        """Stack up the new population, and push the delayed 
        populations to the subordinate history.
        
        """
        if population is None:
            return self
        
        if len(self.queue) == self.delay:
            self.evals += len(self.queue[-1])
            self.updates += 1
            self._empty = False
            self.history.update(self.queue[-1], fitness, space, opt)
            self.queue = self.queue[:-1]
    
        self.queue.append(population)
        return self
    
    def internalUpdate(self, population):
        pass
        
            
    def __getstate__(self):
        start = super(DelayedHistory, self).__getstate__()
        start.update({
           "delay":self.delay,
           "history":self.history.__getstate__(),
           "queue": [p for p in self.queue],
        })
        return start
        
    def __setstate__(self, state):
        self.history.__setstate__(state["history"])
        self.delay = state["delay"]
        self.queue = state["queue"]
        super(DelayedHistory, self).__setstate__(state)

    