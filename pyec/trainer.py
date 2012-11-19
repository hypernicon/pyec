"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from numpy import *
import gc, traceback, sys
from time import time
import logging
log = logging.getLogger(__file__)

from pyec.util.TernaryString import TernaryString
from pyec.util.partitions import Segment, Point, Partition, ScoreTree




class Trainer(object):
   """
      Tool to run an population-based optimizer on a problem.
      
      :param fitness: A function to be maximized; the fitness function or objective.
      :type fitness: any callable object
      :param optimizer: An optimizer to use to optimize the function.
      :type optimizer: :class:`PopulationDistribution`
   """
   
   """
      A list of tuples storing the (group number, group average, running maximum, group maximum) for each set of `groupby` solutions. If `groupby` equals `config.populationSize`, then this list represents the progress of the optimizer broken down into generations.  
   """
   data = []
   
   
   """
      An int indicating how to group the data.
   """
   groupby = 50

   def __init__(self, fitness, optimizer, **kwargs):
      self.fitness = fitness
      self.algorithm = optimizer
      self.config = optimizer.config
      self.sort = True
      self.segment = None
      self.save = True
      self.data = []
      if hasattr(self.config, 'groupby'):
         self.groupby = self.config.groupby
      self.since = 0
      self.groupCount = 0
      self.maxOrg = None
      self.maxScore = None
      if kwargs.has_key("save"):
         self.save = kwargs["save"]
      else:
         if hasattr(self.config, "save"):
            self.save = self.config.save
      config = self.config
      if self.save:
         self.segment = Segment(name=config.segment, config=config)

   def train(self):
      """
         Run the optimizer for `optimizer.config.generations` generations, each with population size `optimizer.config.populationSize`.
         
         :returns: A tuple with two elements containing the best solution found and the maximal fitness (objective value).
      """
      trainStart = time()
      stats = RunStats()
      stats.recording = self.config.recording
      maxScore = -1e100
      maxOrg = None
      gens = self.config.generations
      if gens < 1:
         return maxScore, maxOrg
      successfulMutations = 0
      successImprovement = 0
      lastTime = time()
      for idx in xrange(gens):
         startTime = time()
         stats.start("generation")
         i = idx
         population = []
         start = time()
         stats.start("sample")
         self.config.selectedScores = []
         total = 0.
         count = 0.

         for w, x in enumerate(self.algorithm.batch(self.config.populationSize)):
            stats.stop("sample")
            stats.start("score")
            if not hasattr(self.config, 'convert') or self.config.convert:
               z = self.algorithm.convert(x)
               score = float(self.fitness(self.algorithm.convert(x)))
            else:
               z = x
               score = float(self.fitness(x))
            if self.config.bounded and not self.config.in_bounds(z):
               score = -1e300
            if hasattr(self.fitness, 'statistics'):
               fitnessStats = self.fitness.statistics
            else:
               fitnessStats = None
            if score != score:
               score = -1e300
            total += score
            count += 1
            self.since += 1
            stats.stop("score")
            population.append((x,score, fitnessStats))
            stats.start("sample")
            if len(self.config.selectedScores) > w:
               baseScore = self.config.selectedScores[w]
               if score > baseScore:
                  successfulMutations += 1.0
                  successImprovement += score - baseScore
         genavg = total / count
         attempts = (idx + 1) * self.config.populationSize
         success = ((successfulMutations + 0.) / attempts)
         avgImprove = (successImprovement / (successfulMutations + 1e-10))
         if self.sort:
            population = sorted(population, key=lambda x: x[1], reverse=True)   
            genmax = population[0][1]
            genorg = population[0][0]
            genstat = population[0][2]
         else:
            genmax = max([s for x,s,f in population])
            for x,s,f in population:
               if s == genmax:
                  genorg = x
                  genstat = f
                  break
         
         if genmax > maxScore:
            del maxOrg
            del maxScore
            maxScore = genmax
            maxOrg = genorg
            #print str(self.config.encode(genorg))
            #print genstat
            if hasattr(maxOrg, 'computeEdgeStatistics'):
               maxOrg.computeEdgeStatistics()
               print maxOrg.edges
         else:
            del genorg
            
         while self.since >= self.groupby:
            self.since -= self.groupby
            self.groupCount += 1
            self.data.append((self.groupCount, genavg, maxScore, genmax))
            
         cnt = 0
         pop2 = []
         gps = []
         for point, score, fitnessStats in population:
            stats.start("point")
            pop2.append((point, score))
            if self.save:
             try:
               pt = None
               bn = None
               bit = None
               other = None
               if isinstance(point, ndarray):
                  pt = maximum(1e-30 * ones(len(point)), abs(point))
                  pt *= sign(point)
                  
               elif isinstance(point, TernaryString):
                  bit = point
               elif hasattr(point, 'computeEdgeStatistics'):
                  bn = point
                  # bn.computeEdgeStatistics()
                  # print bn.edges
               else:
                  other = point
                  
               gp = Point(point=pt, bayes=bn, binary=bit, other=other, statistics=fitnessStats, score=score, count=1, segment=self.segment)
               gps.append(gp)
             except:
               raise
            stats.stop("point")
         
         if self.save:
            stats.start("save")
            Point.objects.bulkSave(gps, stats)
            stats.stop("save")
            
         population = pop2
         stats.start("update")
         self.algorithm.update(i+2,population)
         stats.stop("update")
         stats.stop("generation")
         del population
         
         
         if self.save:
            gc.collect()
         if True: # (time() - lastTime) > 1.0:
            lastTime = time()
            if self.config.printOut:
               if self.config.recording:
                  print stats
               if hasattr(self.algorithm, 'var'):
                  print i, ": ", time() - startTime, self.algorithm.var, '%.16f' % genmax, '%.16f' % maxScore  
               else:
                  print i, ": ", time() - startTime, genmax, maxScore
            
         if maxScore >= self.config.stopAt:
            break
      if self.config.printOut:
         print "total time: ", time() - trainStart
         print "best score: ", maxScore
      self.maxScore = maxScore
      self.maxOrg = maxOrg
      return maxScore, maxOrg


class BadAlgorithm(Exception):
   """
      An Exception to indicate that an optimizer provided as an argument
      cannot be run by :class:`Trainer`.
   """
   pass
   
class RunStats(object):
   """
      A simple stats recording tool that can be used to aggregate teh amount of time spent in various methods of an optimizer. Keeps track of variables by key names and outputs the time spent between `start` and `stop` for each key. For a recorded key, the average time spent between `start` and `stop` can be retrieved by [], like so::
      
         def checkStats():     
            stats = RunStats()
            # start the timer
            stats.start("test")
            .. .
            # stop the timer
            stats.stop("test")
            # print just the key "test"
            print stats["test"]
            # print all
            print stats


   """
   totals = {}
   times = {}
   counts = {}
   recording = True
   
   def start(self, key):
      """
         Start recording time for `key`.
         
         :param key: A name for the key.
         :type key: str
      """
      if not self.recording: return
      if not self.totals.has_key(key):
         self.totals[key] = 0.0
         self.counts[key] = 0
      self.times[key] = time()

   def stop(self, key):
      """
         Stop recording time for `key`.
         
         :param key: The previously started key that is to be stopped.
         :type key: str
      """
      if not self.recording: return
      now = time()
      self.totals[key] += now - self.times[key]
      del self.times[key]
      self.counts[key] += 1
      
   def __getitem__(self, key):
      return self.totals[key] / self.counts[key]
      
   def __str__(self):
      ret = ""
      for key,val in sorted(self.totals.items(), key=lambda x: x[0]):
         ret += "%s: %.9f\n" % (key, self[key])
      return ret