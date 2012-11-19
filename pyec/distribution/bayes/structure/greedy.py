"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from numpy import *
from time import time
import traceback
from pyec.trainer import RunStats

from basic import StructureSearch

class GreedyStructureSearch(StructureSearch):
   def __init__(self, branchFactor, scorer):
      self.branchFactor = branchFactor
      self.scorer = scorer

   def __call__(self, network, data):
      self.network = network
      stats = RunStats()
   
      # clear the variables
      for variable in network.variables:
         variable.parents = {}
         variable.update(data)
      
      self.network.dirty = True
      self.network.computeEdgeStatistics()
      
      changes = 1
      score = -1e300
      start = time()
      self.network.changed = {}
      self.network.likelihood(data)
      while True:
         start = time()
         bestType = "none"
         bestIdx1 = 0
         bestIdx2 = 0
         bestScore = score
      
         # try to remove an edge
         for i1, variable in enumerate(network.variables):
            for idx, v2 in variable.parents.iteritems():
               try:
                  toRemove = variable.parents[idx]
                  undo = self.removeEdge(idx, variable, data)
                  self.network.changed = {variable.index:True, toRemove.index:True}
                  score2 = self.attempt(lambda: self.scorer(network, data), undo)
                  if score2 > bestScore:
                     bestScore = score2
                     bestType = "remove"
                     bestIdx1 = variable.index
                     bestIdx2 = toRemove.index
                  undo()
               except:   
                  pass
                  
         
         # try to reverse an edge
         for i1, variable in enumerate(network.variables):
            for idx, v2 in variable.parents.iteritems():
               toReverse = variable.parents[idx]
               if len(toReverse.parents) >= self.branchFactor:
                  continue
               if not self.canReverse(toReverse, variable):
                  continue
               
               
               try:
                  undo = self.reverseEdge(idx, variable, data)
                  self.network.changed = {variable.index:True, toReverse.index:True}
                  score2 = self.attempt(lambda: self.scorer(network, data), undo)
                  self.network.changed = {}
                  
                  if score2 > bestScore:
                     bestScore = score2
                     bestType = "reverse"
                     bestIdx1 = variable.index
                     bestIdx2 = toReverse.index
                  undo()
               except:
                  pass  
               
         
         # try to add an edge
         for i1, variable in enumerate(network.variables):
            for i2, variable2 in enumerate(network.variables):
               if self.admissibleEdge(variable, variable2):
                  
                  if len(variable.parents) >= self.branchFactor:
                     continue
               
                  #stats.start("add.inner")
                  try:
                     #stats.start("add.add")
                     undo = self.addEdge(variable, variable2, data)
                     #stats.stop("add.add")
                     #stats.start("add.score")
                     self.network.changed = {variable.index:True, variable2.index:True}
                     score2 = self.attempt(lambda: self.scorer(network, data), undo)
                     self.network.changed = {}
                     #stats.stop("add.score")
                     #print "add ", variable.index, variable2.index, score2, bestScore
                     if score2 > bestScore:
                        bestScore = score2
                        bestType = "add"
                        bestIdx1 = variable.index
                        bestIdx2 = variable2.index
                     undo()
                  except:
                     pass
                  #stats.start("add.compute")
                  network.computeEdgeStatistics()
                  #stats.stop("add.compute")
                  #stats.stop("add.inner")
         if bestType == "none":
            break
         elif bestType == "remove":
            try:
               elem = None
               for var in network.variables:
                  if var.index == bestIdx1:
                     elem = var
                     break
               undo = self.removeEdge(bestIdx2, elem, data)
            except:   
               pass
         elif bestType == "reverse":
            try:
               elem = None
               for var in network.variables:
                  if var.index == bestIdx1:
                     elem = var
                     break
               undo = self.reverseEdge(bestIdx2, elem, data)
            except:   
               pass
         elif bestType == "add":
            try:
               v1 = None
               v2 = None
               for var in network.variables:
                  if var.index == bestIdx1:
                     v1 = var
                  if var.index == bestIdx2:
                     v2 = var
               undo = self.addEdge(v1, v2, data)
            except:
               print "exception during add"
               traceback.print_exc()   
               pass
         changes += 1
         score = bestScore
         self.network.changed = {}
         self.network.likelihood(data)
      
      return score
         
