"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from numpy import *
import numpy.linalg

from pyec.distribution.basic import *
from pyec.distribution.bayes.net import BayesNet
from pyec.distribution.bayes.sample import DAGSampler
from pyec.distribution.bayes.structure.greedy import GreedyStructureSearch
from pyec.distribution.bayes.structure.proposal import StructureProposal
from pyec.distribution.bayes.score import BayesianDirichletScorer, BayesianInformationCriterion
from pyec.distribution.bayes.variables import *
from pyec.util.TernaryString import TernaryString

from pyec.config import ConfigBuilder

import scipy.cluster.vq as vq

seterr(divide="ignore")

class RBoaConfigurator(ConfigBuilder):
   
   def __init__(self, *args):
      super(RBoaConfigurator, self).__init__(Boa)
      self.cfg.toSelect = .25
      self.cfg.space = 'real'
      self.cfg.branchFactor = 3

   def postConfigure(self, cfg):
      cfg.numVariables = cfg.dim

class BoaConfigurator(RBoaConfigurator):
   def __init__(self, *args):
      super(BoaConfigurator, self).__init__(*args)
      self.cfg.space = 'binary'
      self.cfg.branchFactor = 10
      self.cfg.binaryDepth = 16

   def postConfigure(self, cfg):
      cfg.rawdim = cfg.dim
      cfg.rawscale = cfg.scale
      cfg.rawcenter = cfg.center
      cfg.dim = cfg.dim * cfg.binaryDepth
      cfg.center = .5
      cfg.scale = .5 
      cfg.numVariables = cfg.dim

class Boa(PopulationDistribution):
   unsorted = False

   def __init__(self, cfg):
      super(Boa, self).__init__(cfg)
      self.dim = cfg.dim
      self.selected = cfg.toSelect
      if cfg.space == 'real':
         cfg.variableGenerator = lambda i: GaussianVariable(i, cfg.dim, cfg.scale)
         cfg.structureGenerator = GreedyStructureSearch(cfg.branchFactor, BayesianInformationCriterion())
         cfg.randomizer = lambda x: zeros(cfg.dim)
         cfg.sampler = DAGSampler()
         self.network = BayesNet(cfg)
         self.config.data = None
         self.network = StructureProposal(self.config)(self.network)
         self.initial = FixedCube(cfg)
      else:
         cfg.variableGenerator = BinaryVariable
         cfg.structureGenerator = GreedyStructureSearch(cfg.branchFactor, BayesianDirichletScorer())
         cfg.randomizer = lambda x: TernaryString(0L,0L)
         cfg.sampler = DAGSampler()
         self.network = BayesNet(cfg)
         self.config.data = None
         self.network = StructureProposal(self.config)(self.network)
         self.initial = BernoulliTernary(cfg)
      self.trained = False
      self.cfg = cfg
      self.elitist = False
      if hasattr(cfg, 'elitist') and cfg.elitist:   
         self.elitist = True
      self.maxScore = 0
      self.maxOrg = None

   def __call__(self):
      if not self.trained:
         return self.initial.__call__()
      
      if self.cfg.space == "real":
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
      
      x = self.network.__call__()
      if self.cfg.space == "binary":
         x = self.decodeData(x)
      elif self.cfg.bounded and not self.cfg.in_bounds(x):
         return self.__call__()
      return x

   def batch(self, num):
      if hasattr(self, 'newpop'):
         return self.newpop
      else:
         return [self.__call__() for i in xrange(num)]

   def update(self, generation, population):
      self.trained = True
      if self.maxOrg is None or self.maxScore <= population[0][1]:
         self.maxOrg = population[0][0]
         self.maxScore = population[0][1]
      selected = int(self.selected * len(population))
      if self.elitist:
         data = [self.convertData(self.maxOrg), self.maxScore] + [self.convertData(x) for x,s in population[:selected]]
      else:
         data = [self.convertData(x) for x,s in population[:selected]]
      self.network = BayesNet(self.config)
      self.config.data = None
      self.network = StructureProposal(self.config)(self.network)
      self.config.data = data
      self.network.structureSearch(data)
      self.network.update(generation, data)
      
      if self.config.space != "real":
         return
      
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
             
      self.newpop = data + [self.__call__() for i in xrange(len(population) - selected)]
         
        
   def project(self, data, subset):
      data2 = []
      for d in data:
         nd = []
         for idx in subset:
            nd.append(d[idx])
         data2.append(nd)
      return array(data2)
   
   def convert(self, x):
      if self.config.space == "binary":
         ns = array([i+1 for i in xrange(self.config.binaryDepth)] * self.config.rawdim)
         ms = .5 ** ns
         y = reshape(x.__mult__(ms), (self.config.binaryDepth, self.config.rawdim))
         y = y.sum(axis=0).reshape(self.config.rawdim)
         return y * self.config.rawscale + self.config.rawcenter
      return x
   
   def convertData(self, x):
      return x
   
   def decodeData(self, x):
      return y
   
   @classmethod
   def configurator(cls):
      return BoaConfigurator(cls)   
 
