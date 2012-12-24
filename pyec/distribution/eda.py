"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from numpy import *
import numpy.linalg

from pyec.distribution.basic import PopulationDistribution
from pyec.distribution.bayes.net import BayesNet
from pyec.distribution.bayes.sample import DAGSampler
from pyec.distribution.bayes.structure.greedy import GreedyStructureSearch
from pyec.distribution.bayes.structure.proposal import StructureProposal
from pyec.distribution.bayes.score import BayesianDirichletScorer, BayesianInformationCriterion
from pyec.distribution.bayes.variables import BinaryVariable, GaussianVariable
from pyec.history import SortedMarkovHistory
from pyec.space import Binary, BinaryReal, Euclidean
from pyec.util.TernaryString import TernaryString

from pyec.config import Config

import scipy.cluster.vq as vq

seterr(divide="ignore")


class Boa(PopulationDistribution):
   """The Bayesian Optimization Algorithm of Martin Pelikan. """
   config = Config(variableGenerator=BinaryVariable,
                   branchFactor=3,
                   structureGenerator=GreedyStructureSearch(3, BayesianInformationCriterion()),
                   randomizer = lambda net: TernaryString(0L, -1L, net.numVariables),
                   sampler = DAGSampler(),
                   data = None,
                   truncate = 0.60, # percentage of the population to keep
                   space = BinaryReal(realDim=5),
                   history=SortedMarkovHistory)

   def __init__(self, **kwargs):
      super(Boa, self).__init__(**kwargs)
      self.network = BayesNet(numVariables=self.config.space.dim,
                              **self.config.__properties__)
      self.network = StructureProposal(**self.config.__properties__).search(self.network)
      self.trained = False

   def compatible(self, history):
      return hasattr(history, "lastPopulation") and history.sorted

   def sample(self):
      x = self.network.__call__()
      cnt = 0
      while not self.config.space.in_bounds(x):
         if cnt > 10000:
            raise ValueError("Rejection sampling failed after 10,000 attempts in BOA")
         x = self.network.__call__()
         cnt += 1
      return x

   def batch(self, num):
      return [self.sample() for i in xrange(num)]

   def update(self, history, fitness):
      super(Boa, self).update(history,fitness)
      if history.lastPopulation() is None:
         return
      
      population = [x for x,s in history.lastPopulation()]
      selected = int(self.config.truncate * len(population))
      self.network = BayesNet(numVariables=self.config.space.dim,
                              **self.config.__properties__)
      self.network.config.data = None
      self.network = StructureProposal(**self.network.config.__properties__).search(self.network)
      self.network.config.data = population
      self.network.structureSearch(population)
      self.network.update(self.history.updates, population)


class RBoa(Boa):
   """The Real-coded Boa algorithm of Ahn et al."""
   config = Config(variableGenerator=GaussianVariable,
                   randomizer = lambda net: zeros(net.numVariables),
                   space=Euclidean(dim=5))
   
   def sample(self):
      # set up the clustering
      for w, subset in enumerate(self.decomp):
         # pick a cluster
         r = random.random_sample()
         idx = None
         for i, c in enumerate(self.clusterCoeffs[w]):
            if c >= r:
               idx = i
            
         # set the mean, sd, and sdinv to match the cluster choice
         for j in subset:
            var = self.network.get(j)
            var.mu = self.clusterMeans[w][idx][j]
            var.sd = self.clusterSds[w][idx][j]
            var.sdinv = self.clusterSdInvs[w][idx][j]
      
      return super(RBoa, self).sample()
   
   def update(self, history, fitness):
      super(RBoa,self).update(history, fitness)
      if history.lastPopulation() is None:
         return
      
      population = [x for x,s in history.lastPopulation()]
      selected = int(self.config.truncate * len(population))
      data = population[:selected]
      
      decomp = self.network.decompose()
      
      self.decomp = decomp
      self.assignments = []
      self.numClusters = {}
      self.clusterCoeffs = {}
      self.clusterMeans = {}
      self.clusterSds = {}
      self.clusterSdInvs = {}
      #cluster within decomp
      for w, subset in enumerate(decomp):
         data2 = self.project(data, subset)
         distortion = 1e300
         means = None
         for k in xrange(selected / 5):
            ms, distort = vq.kmeans(data2, k+1)
            if distort < distortion or means is None:
               means = ms
               self.numClusters[w] = k+1
         
         # assign data to closest cluster
         assign = {}
         coeffs = zeros(self.numClusters[w])
         assigned = [[] for i in xrange(self.numClusters[w])]
         for i, d in enumerate(data2):
            closest = 1e300
            for j, m in enumerate(means):
               dist = ((d - m) ** 2).sum()
               if not assign.has_key(i) or dist < closest:
                  assign[i] = j
                  closest = dist
            coeffs[assign[i]] += 1. / selected
            assigned[assign[i]].append(data[i])
         
         # print w, self.numClusters[w], " clusters"
         coeffs2 = zeros(self.numClusters[w])
         total = 0.0
         for idx, c in enumerate(coeffs):
            total += c
            coeffs2[idx] = total
         self.assignments.append(assign)
         # print coeffs2
         self.clusterCoeffs[w] = coeffs2
         
         self.clusterMeans[w] = {}
         self.clusterSds[w] = {}
         self.clusterSdInvs[w] = {}
         for cnum, subdata in enumerate(assigned):
             # compute mean and sd for the cluster
             subdata = array(subdata)
             self.clusterMeans[w][cnum] = {}
             self.clusterSds[w][cnum] = {}
             self.clusterSdInvs[w][cnum] = {}
             for idx in subset:
                # get variable idx
                var = self.network.get(idx)
                # set variable params 
                var.update(subdata)
                # store variable params
                self.clusterMeans[w][cnum][idx] = var.mu
                self.clusterSds[w][cnum][idx] = var.sd
                self.clusterSdInvs[w][cnum][idx] = var.sdinv
        
   def project(self, data, subset):
      data2 = []
      for d in data:
         nd = []
         for idx in subset:
            nd.append(d[idx])
         data2.append(nd)
      return array(data2)


