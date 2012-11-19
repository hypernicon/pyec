"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from numpy import *

class CyclicException(Exception):
   pass

class DuplicateEdgeException(Exception):
   pass
   
class IrreversibleEdgeException(Exception):
   pass   

class StructureSearch(object):
   def __init__(self, scorer, autocommit=False):
      self.scorer = scorer
      self.autocommit = autocommit
      self.network = None
      
   def canReverse(self, newChild, newParent):
      """
         check to ensure reverse link is not already present
         (In a DAG, it should not be)
      """
      if newChild.parents.has_key(newParent.index):
         return False
      return True
   
   def admissibleEdge(self, var1, var2):
      """Is edge admissible in a DAG?"""
      if var1.index == var2.index:
         return False
      if var1.parents.has_key(var2.index):
         return False
      if var2.parents.has_key(var1.index):
         return False
      return True
   
   def merge(self, net, other, data, allowCyclic=False):
      """add the edges from other to self, preventing cycles if asked"""
      self.network = net
      net.computeEdgeStatistics()
      other.computeEdgeStatistics()
      indexMap = dict([(v.index, v) for v in net.variables])
      undoList = []
      
      def undo(update=True):
         for undo2 in reversed(undoList):
            undo2(False)
         
      for frm, to in other.edges:
         try:
            frm2 = indexMap[frm.index]
            to2 = indexMap[to.index]
            undo2 = self.addEdge(to2, frm2, data, allowCyclic)
            frm2.children = None
            undoList.append(undo2)
         except Exception, msg:
            pass
      
      return undo
   
   def cross(self, net, other, data, allowCyclic=False):
      self.network = net
      net.computeEdgeStatistics()
      other.computeEdgeStatistics()
      indexMap = dict([(v.index, v) for v in net.variables])
      indexMap2 = dict([(v.index, v) for v in other.variables])
      undoList = []
      if len(net.edges) == 0: return other
      if len(other.edges) == 0: return net
      if len(net.edges) < net.numVariables / 2 and len(other.edges) < other.numVariables / 2:
         return net
      
      
      def undo(update=True):
         for undo2 in reversed(undoList):
            undo2(False)
            
      for variable in net.variables:
         # pick a parent
         if random.random_sample < 0.5:
            # Add relationships from other, avoiding cycles
            ps = len(variable.parents)
            for idx, parent in variable.parents.iteritems():
               undoList.append(self.removeEdge(idx, variable, allowCyclic))
               parent.children = None
            
            for idx, parent2 in v2.parents.iteritems():
               try:
                  parent = indexMap[parent.index]
                  undoList.append(self.addEdge(variable, parent, data, allowCyclic))
                  parent.children = None
               except Exception, msg:
                  pass
      net.computeEdgeStatistics()
      return undo
   
   def removeEdge(self, i, variable, data=None):
      self.network.computeEdgeStatistics()
      oldstate = self.network.getComputedState()
      toRemove = variable.parents[i]
      variable.removeParent(toRemove)
      toRemove.children = None
      self.network.dirty = True
            
      def undo(update=True):
         variable.addParent(toRemove)
         toRemove.children = None
         self.network.restoreComputedState(oldstate)
            
      try:
         self.network.updateVar(variable, data)
      except:
         undo()
         raise
      
      return undo      
      
      
   def addEdge(self, child, parent, data = None, allowCyclic = False):
      self.network.computeEdgeStatistics()
      oldstate = self.network.getComputedState()
      if child.parents.has_key(parent.index):
         raise DuplicateEdgeException, "Edge already exists"
      child.addParent(parent)
      parent.children = None
      self.network.dirty = True
      
      def undo(update=True):
         parent.children = None
         child.removeParent(parent)
         self.network.restoreComputedState(oldstate)
      
      if (not allowCyclic) and not self.network.isAcyclic():
         undo()
         raise CyclicException, "Adding an edge makes network cyclic"
                  
      try:
         self.network.updateVar(child, data)
      except:
         undo()
         raise            
      return undo 
      
   def reverseEdge(self, i, variable, data=None, allowCyclic = False):
      """toReverse is new child, variable is new parent"""
      self.network.computeEdgeStatistics()
      oldstate = self.network.getComputedState()
      toReverse = variable.parents[i]
      if not self.canReverse(toReverse, variable):
         raise IrreversibleEdgeException, "Edge reversal disallowed"
      variable.removeParent(toReverse)
      toReverse.addParent(variable)
      variable.children = None
      toReverse.children = None
      self.network.dirty = True
      
      def undo(update=True):
         variable.addParent(toReverse)
         toReverse.removeParent(variable)
         self.network.restoreComputedState(oldstate)
            
      if (not allowCyclic) and not self.network.isAcyclic():
         undo()
         raise CyclicException, "Reversing an edge makes nework cyclic"
      
      try:
         self.network.updateVar(variable, data)
         self.network.updateVar(toReverse, data)
      except:
         undo()
         raise
      return undo
      
   def attempt(self, fn, exc):
      try:
         return fn()
      except:
         exc()
         raise