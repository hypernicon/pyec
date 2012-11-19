"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import cPickle
from cStringIO import StringIO
from sample import *
from structure import *
from variables import *
from pyec.distribution.basic import Distribution
from pyec.util.cache import LinkedList, LRUCache

def checkDeferred(handler):
      def handle(*args, **kwargs):
         if args[0].deferred:
            args[0].initialize()   
         return handler(*args, **kwargs)
      return handle

def checkDeferredWeights(handler):
      def handle(*args, **kwargs):
         if args[0].deferredWeights:
            args[0].estimate()   
         return handler(*args, **kwargs)
      return handle

class BayesNet(Distribution):
   cfg = None
   densityCache = LRUCache()
   weightCache = LRUCache()
   likelihoodCache = LRUCache()

   def __init__(self, config = None):
      BayesNet.cfg = config
      super(BayesNet, self).__init__(config)
      self.config = config
      self.numVariables = config.numVariables
      self.variableGenerator = config.variableGenerator
      self.structureGenerator = config.structureGenerator
      self.randomizer = config.randomizer
      self.sampler = config.sampler
   
      self.variables = []
      for i in xrange(self.numVariables):
         self.variables.append(self.variableGenerator(i))
      self.decay = 1
      self.dirty = False
      self.acyclic = True
      self.edges = []
      self.edgeRatio = 0.0
      self.edgeTuples = None
      self.cacheKeys = dict([(v.index, v.cacheKey) for v in self.variables])
      self.edgeMap = {}
      self.binary = zeros(len(self.variables)**2)
      self.deferred = False
      self.deferredWeights = False
      self.edgeRep = None
      self.densityStored = None
      self.cacheHits = 0
      self.cacheTries = 0
      self.changed = {}
      self.last = {}
   
   @checkDeferred
   def get(self, index):
      for var in self.variables:
         if var.index == index:
            return var
      return None
   
   @checkDeferred
   def sort(self):
      self.variables = sorted(self.variables, key=lambda x: len(x.parents))
      visited = {}
      skipped = {}
      #print "before: ", [v.index for v in self.variables]
      vars = LinkedList()
      for var in self.variables:
         vars.append(var)
      current = vars.first
      while current is not None:
         v = current.value
         advance = True
         for idx, p in v.parents.iteritems():
            if not visited.has_key(idx):
               before = current.before
               vars.remove(current)
               vars.append(v)
               if before is not None:
                  current = before.after
               advance = False
               skipped[v.index] = True
               break
         if not advance and skipped.has_key(current.value.index):
            break
         if advance:
            current = current.after
            skipped = {}
            visited[v.index] = True
         #print i, visited, v.index, v.parents
      current = vars.first
      self.variables = []
      while current is not None:
         self.variables.append(current.value)
         current = current.after
      #print "after: ", [v.index for v in self.variables]
      return len(skipped) == 0
   
   @checkDeferred
   def decompose(self):      
      acyclic = self.sort()
      #self.variables is now ordered in a way to allow sampling
      clusters = []
      for var in self.variables:
         # is the parent in one of the clusters?
         member = False
         for c in clusters:
            for v in c:
               if var.parents.has_key(v):
                  member = True
                  break
            if member:
               c.append(var.index)
               break
         if not member:
            clusters.append([var.index])
      return clusters               
         
   @checkDeferredWeights
   def distribution(self, index, x):
      var = None
      for v in self.variables:
         if v.index == index:
            var = v
      return var.distribution(x)
         
   @checkDeferredWeights   
   def randomize(self):
      return self.randomizer(self)  
   
   @checkDeferredWeights
   def conditionalLikelihood(self, index, data):
      sum = 0.0
      var = None
      for v in self.variables:
         if v.index == index:
            var = v
            break
      for x in data:
         key = self.cacheKeys[var.index]
         key += str(x[var.index])
         if not BayesNet.densityCache.has_key(key):
            BayesNet.densityCache[key] = var.density(x)
         sum += log(BayesNet.densityCache[key])
      return sum
            
   @checkDeferredWeights   
   def likelihood(self, data):
      prod = 0.0
      for v in self.variables:
         vprod = 0.0
         for x in data:
            key = self.cacheKeys[v.index]
            key += str(x[v.index])
            if not BayesNet.densityCache.has_key(key):
                  BayesNet.densityCache[key] = v.density(x)
            vprod += log(BayesNet.densityCache[key])
         self.last[v.index] = vprod
         prod += vprod
      self.densityStored = prod
      return prod
   
   @checkDeferredWeights   
   def likelihoodChanged(self, data):
      diff = 0.0
      olddiff = 0.0
      total = 0.0
      for v in self.variables:
         if self.changed.has_key(v.index) and self.changed[v.index]:
            for x in data:
               key = self.cacheKeys[v.index]
               key += str(x[v.index])
               if not BayesNet.densityCache.has_key(key):
                  BayesNet.densityCache[key] = v.density(x)
               total += log(BayesNet.densityCache[key])
         else:
            total += self.last[v.index]
      return total
   
   @checkDeferredWeights
   def marginal(self, cmpIdx, data):
      var = None
      for v in self.variables:
         if v.index == cmpIdx:
            var = v
            break
      total = 0.0
      for t in data:
         total += var.marginalDensity(t, self)
      return total / len(data)

   @checkDeferredWeights
   def map(self, cmpIdx, data):
      var = None
      for v in self.variables:
         if v.index == cmpIdx:
            var = v
            break
      total = 0.0
      for t in data:
         z = var.map(t)
         if z[cmpIdx] == t[cmpIdx]:
            total += 1.0
      return total / len(data)


                     
   @checkDeferredWeights   
   def density(self, x):
      self.computeEdgeStatistics()
      prod = 1.0
      for variable in self.variables:
         key = self.cacheKeys[variable.index]
         key += str(x[variable.index])
         if not BayesNet.densityCache.has_key(key):
            BayesNet.densityCache[key] = variable.density(x)
         #print key, " - ", BayesNet.densityCache[key], " - ", variable.density(x)
         prod *= BayesNet.densityCache[key]
      return prod
      
   @checkDeferredWeights   
   def __call__(self):
      """sample the network"""
      return self.sampler(self)

   @checkDeferredWeights   
   def batch(self, num):
      return [self.__call__() for i in xrange(num)]

   @checkDeferred   
   def numFreeParameters(self):
      total = 0
      for variable in self.variables:
         total += variable.numFreeParameters()
      return total

   @checkDeferred   
   def update(self, epoch, data):
      self.computeEdgeStatistics()
      for variable in self.variables:
         self.updateVar(variable, data)
   
   @checkDeferred
   def updateVar(self, variable, data):
      self.computeEdgeStatistics()
      key = self.cacheKeys[variable.index]
      if not BayesNet.weightCache.has_key(key):
         variable.update(data)
         BayesNet.weightCache[key] = variable.getComputedState()
      else:
         variable.restoreComputedState(BayesNet.weightCache[key])
   
   @checkDeferred   
   def merge(self, other, data):
      return self.structureGenerator.merge(self, other, data)                  

   @checkDeferred   
   def cross(self, other, data):
      return self.structureGenerator.cross(self, other, data) 

   @checkDeferred   
   def hasEdge(self, frm, t):
      """has an edge from the parent with index 'from'
         to the child with index 'to'
         
         TODO: improve efficiency; current implementation N^2
         can be made constant
      """
      try:
         toNode = [variable for variable in self.variables if variable.index == t][0]
         fromNode = [parent for l,parent in fromNode.parents.iteritems() if parent.index == frm][0]
         return True
      except:
         return False

   @checkDeferred   
   def isAcyclic(self):
      """Is the network a DAG?"""
      if self.dirty: 
         self.computeEdgeStatistics()
      return self.acyclic
      
      """
      tested = set([])
      for variable in self.variables:
         if len(set(variable.parents) - tested) > 0:
            self.acyclic = False
            return False
         tested = set(list(tested) + [variable])
      self.acyclic = True   
      return True   
      """   
      
      """
      for variable in self.variables:
         if variable in variable.parents:
            return False
            
      tested = set([])
      while len(tested) < len(self.variables):
        added = False
        for variable in self.variables:
            if variable in tested:
               continue
            if len(set(variable.parents) - tested) == 0:
               tested = set(list(tested) + [variable])
               added = True
               break
        if not added:
           return False
      return True
      """

   @checkDeferred   
   def structureSearch(self, data):
      return self.structureGenerator(self, data)
   
   @checkDeferredWeights
   def getComputedState(self):
      state = {}
      self.computeEdgeStatistics()
      state['acyclic'] = self.acyclic
      state['edges'] = self.edges
      state['edgeRatio'] = self.edgeRatio
      state['edgeTuples'] = self.edgeTuples
      state['cacheKeys'] = self.cacheKeys
      varstate = {}
      varorder = {}
      for i,v in enumerate(self.variables):
         varstate[v.index] = v.getComputedState()
         varorder[i] = v.index
      state['varstate'] = varstate
      state['varorder'] = varorder
      return state

   def restoreComputedState(self, state):
      self.dirty = False
      self.acyclic = state['acyclic']
      self.edges = state['edges']
      self.edgeRatio = state['edgeRatio']
      self.edgeTuples = state['edgeTuples']
      self.cacheKeys = state['cacheKeys']
      varorder = state['varorder']
      varstate = state['varstate']
      self.variables = sorted(self.variables, key=lambda v: varorder[v.index])
      for v in self.variables:
         v.restoreComputedState(varstate[v.index])
            
   @checkDeferred   
   def computeEdgeStatistics(self):
      if not self.dirty: return
      
      self.acyclic = self.sort()
      if self.edges is not None: del self.edges
      self.edges = []
      for variable in self.variables:
         for l, variable2 in variable.parents.iteritems():
            self.edges.append((variable2, variable))
      self.edges = sorted(self.edges, key=lambda e: (e[0].index,e[1].index))
      self.edgeRatio = len(self.edges) / (1e-10 + (len(self.variables) ** 2))
      self.edgeBinary()
      self.edgeTuples = [(frm.index, to.index) for frm,to in self.edges]
      self.cacheKeys = dict([(v.index, v.cacheKey) for v in self.variables])
      self.dirty = False
      self.densityStored = None
   
      
   @checkDeferred   
   def getChildren(self, variable):
      children = []
      self.computeEdgeStatistics()
      for variable2 in self.variables:
         if variable2.parents.has_key(variable.index):
             children.append(variable2)
      return children
   
   @checkDeferred   
   def updateVariables(self, data):
      for variable in self.variables:
         variable.update(data)
   
   @checkDeferred   
   def __getitem__(self, index):
      if self.edgeMap.has_key(index):
         return self.edgeMap[index]
      frmidx = index % self.numVariables
      toidx = index / self.numVariables
      for l, v in self.variables[toidx].parents.iteritems():
         if v.index == frmidx:
            self.edgeMap[index] = True
            return True
      self.edgeMap[index] = False
      return False
      
   def __len__(self):
      return self.numVariables ** 2
   
   @checkDeferred   
   def edgeBinary(self):
      if not self.dirty: return self.binary
      if self.edgeMap is not None:
         del self.edgeMap
      self.edgeMap = {}
         
      ret = self
      #ret = zeros(len(self.variables)**2)
      #for frm,t in self.edges:
      #   idx = frm.index + len(self.variables) * t.index
      #   ret[idx] = 1
      self.binary = ret
      return ret
            
   def __getstate__(self):
      return {
         'v': self.variables,
         'r': self.randomizer,
         's': self.sampler,
         'sg': self.structureGenerator,
      }
      
   def __setstate__(self, state):
      self.dirty = True
      self.variables = state['v']
      self.numVariables = len(self.variables)
      self.randomizer = state['r']
      self.structureGenerator = state['sg']
      self.sampler = state['s']
      indexMap = {}
      for variable in self.variables:
         indexMap[variable.index] = variable
      for variable in self.variables:
         variable.parents = {}
         for i in variable.parentIndices:
            variable.addParent(indexMap[i])
            
      self.changed = {}    
      self.last = {}  
      self.deferred = False
      self.deferredWeights = True
      self.edgeRep = None
      self.edgeMap = {}
      self.edgeTuples = None
      self.edges = None
      self.binary = None
      self.densityStored = None
      self.computeEdgeStatistics()
      
   @checkDeferred   
   def __str__(self):
      """pickle the object"""
      self.computeEdgeStatistics()
      return cPickle.dumps(len(self.variables)) + cPickle.dumps(self.edgeTuples)
      
   def initialize(net):
      for frm,to in net.edgeRep:
         net.variables[to].addParent(net.variables[frm])
      net.dirty = True
      net.deferred = False
      net.edgeRep = None
      net.computeEdgeStatistics()   
   
   @checkDeferred
   def estimate(net):
      cfg = BayesNet.__dict__['cfg']
      for variable in net.variables:
         net.updateVar(variable, cfg.data) 
      net.deferredWeights = False
            
   @classmethod
   def parse(cls, rep):
      io = StringIO(rep)
      numVars = cPickle.load(io)
      cfg = BayesNet.__dict__['cfg']
      net = cls(cfg)
      edges = cPickle.load(io)
      net.edgeRep = edges
      net.deferred = True
      net.deferredWeights = True
      
      return net
