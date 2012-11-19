"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from numpy import *
from pyec.distribution.basic import Distribution
from pyec.util.TernaryString import TernaryString
from time import time
from sample import *
import numpy.linalg
import gc

class rbm(Distribution):
   """ a binary rbm """
   center = 0.5
   scale = 0.5
   
   def __init__(self, vsize, hsize, lr=0.01, mo=0.9):
      self.vsize = vsize
      self.hsize = hsize
      self.dim = vsize + hsize
      self.rate = lr
      self.momentum = mo
      self.w = random.standard_cauchy((vsize, hsize)) / vsize / hsize
      self.bv = random.standard_cauchy(vsize) / vsize / hsize
      self.bh = random.standard_cauchy(hsize) / hsize / vsize
      self.wc = zeros((vsize,hsize))
      self.bvc = zeros(vsize)
      self.bhc = zeros(hsize)
      self.wg = self.w.copy()
      self.bvg = self.bv.copy()
      self.bhg = self.bh.copy()
      self.wcg = zeros((vsize,hsize))
      self.bvcg = zeros(vsize)
      self.bhcg = zeros(hsize)
      self.sampler = RBMSimulatedTempering(1000)
      self.samplerAlt = RBMGibbsSampler(1000, 100)
      self.epoch = 0
      self.batchSize = 100
      
      
   def __call__(self, x):
      return -self.energy(x)   
      
   def energy(self, x, useAlternate=False):
      """ compute the energy function """
      if useAlternate: return self.energy2(x)
      v = x[:self.vsize].toArray(self.vsize)
      h = x[self.vsize:self.vsize+self.hsize].toArray(self.hsize)
      ret = -(dot(v, dot(self.w, h)) + dot(v, self.bv) + dot(h, self.bh))
      return ret
   
   def energy2(self, x):
      """ compute the energy function """
      v = x[:self.vsize].toArray(self.vsize)
      h = x[self.vsize:self.vsize+self.hsize].toArray(self.hsize)
      ret = -(dot(v, dot(self.wg, h)) + dot(v, self.bvg) + dot(h, self.bhg))
      return ret
   
   def partition(self):
      """ compute the partition function - only for small dimension !!! """
      from pyec.util.TernaryString import TernaryString
      total = 0
      vsize = self.vsize
      hsize = self.hsize
      all = (1L << (vsize+hsize)) - 1L
      for i in xrange(1 << (vsize+hsize)):
         total += exp(self.__call__(TernaryString(long(i), all)))
      return total
   
   def scoreSample(self, sample, Z=1.0):
      return [(x, exp(-self.__call_(x))/Z) for x in sample]
   
   def batch(self, size):
      return self.sampler(self, size)
      

                           
   def bucket(self, sample, Z=1.0):
      """build a dictionary containing a histogram"""
      d = {}
      size = len(sample)
      incr = 1.0 / size
      for x in sample:
         y = str(x)
         if d.has_key(y):
            d[y][0] += incr
         else:
            d[y] = [incr, exp(-self.__call__(x))/Z]
      return d

   def complete(self, data, sample=True):
      completed = []
      for v in data:
         x = zeros(self.vsize + self.hsize)
         x[:self.vsize] = v
         h = dot(v, self.w) + self.bh
         if sample:
            x[self.vsize:] = random.binomial(1, 1. / (1. + exp(-h)), self.hsize)
            completed.append(TernaryString.fromArray(x))
         else:
            x[self.vsize:] = 1. / (1. + exp(-h))
            completed.append(x)
      return completed
   
   def completeV(self, data, sample=True):
      completed = []
      for h in data:
         h2 = h.toArray(self.bh.size)
         x = zeros(self.vsize + self.hsize)
         x[self.vsize:] = h2
         v = dot(self.w, h2) + self.bv
         if sample:
            x[:self.vsize] = random.binomial(1, 1. / (1. + exp(-v)), self.vsize)
            completed.append(TernaryString.fromArray(x))
         else:
            x[:self.vsize] = 1. / (1. + exp(-v))
            completed.append(x)
      return completed
   
   def complete2(self, data):
      completed = []
      for v in data:
         x = zeros(self.vsize + self.hsize)
         x[:self.vsize] = v
         h = dot(v, self.wg) + self.bhg
         x[self.vsize:] = 1. / (1. + exp(-h))
         completed.append(TernaryString.fromArray(x))
      return completed

   def logistic(self, x):
      x = minimum(maximum(x, -10.), 10.)
      return 1. / (1. + exp(-x))  
         
   def correlate(self, data):
      ws = zeros((self.vsize, self.hsize))
      vs = zeros(self.vsize)
      hs = zeros(self.hsize)
      for d in data:
         x = d.toArray(self.vsize + self.hsize)
         v = x[:self.vsize]
         h = x[self.vsize:]
         ws += outer(v,h) / len(data)
         vs += v / len(data)
         hs += h / len(data)
      return ws, vs, hs

   def train(self, n):
      from pyec.util.partitions import ScoreTree, Partition, Point
      from pyec.trainer import RunStats
      stats = RunStats()
      stats.recording = False
      numBatches = len(self.data) / self.batchSize
      lr = 0.001 / ((n/numBatches+1.)**2)
      current = self.epoch % numBatches
      start = current * self.batchSize
      end = start + self.batchSize
      data = self.data[start:end]
      completed = self.complete(data)
      energy = sum([self.energy(d) for d in completed]) / len(completed)
      print "Energy of data: ", energy#, " v ", energy2, "\n\n"
      sampled = self.sampler.batch(self.batchSize)
      energys = sum([self.energy(d) for d in sampled]) / len(sampled)
      print "Energy of sample: ", energys
      for point in completed:
         gp = Point(point=None, bayes=None, binary=point, score=-self.energy(point), count=1, segment=self.sampler.selectors[-1].segment)
         gp.save()
         try:
            Partition.objects.separate(gp, self.sampler.config, stats)
            ScoreTree.objects.insert(gp, self.sampler.config, stats)
         except:
            gp.alive = False
            gp.save()
      wcb = abs(self.w).sum()
      dw, dv, dh = self.correlate(completed)
      mw, mv, mh = self.correlate(sampled)
      diffw = dw - mw
      diffv = dv - mv
      diffh = dh - mh
      self.wc += (1 - self.momentum) * lr * diffw
      self.bvc += (1 - self.momentum) * lr * diffv
      self.bhc += (1 - self.momentum) * lr * diffh
      self.w += self.wc
      self.bv += self.bvc
      self.bh += self.bhc
      self.wc *= self.momentum
      self.bhc *= self.momentum
      self.bvc *= self.momentum
    
      
      print "scale of deriv: ", average(diffw), average(diffv), average(diffh)
      
      """
      gw, gv, gh = self.correlate(g)
      dw, dv, dh = self.correlate(c2)
      self.wcg += (1 - self.momentum) * self.rate * (dw - gw)
      self.bvcg += (1 - self.momentum) * self.rate * (dv - gv)
      self.bhcg += (1 - self.momentum) * self.rate * (dh - gh)
      self.wg += self.wc
      self.bvg += self.bvc
      self.bhg += self.bhc
      self.wcg *= self.momentum
      self.bhcg *= self.momentum
      self.bvcg *= self.momentum
      """
      self.sampler.completeTraining(self, n)
        
   def meanFieldUp(self, n, vs):
      wt = self.w.transpose()
      hs = random.random_sample((n,self.hsize))
      for j in xrange(n):
         for i in xrange(25):
            hs[j] = self.logistic(dot(wt, vs[j]) + self.bh)
      gc.collect()
      return hs   
   
   def meanFieldDown(self, n, hs):
      vs = random.random_sample((n,self.vsize))
      for j in xrange(n):
         for i in xrange(25):
            vs[j] = self.logistic(dot(self.w, hs[j]) + self.bv)
      gc.collect()
      return vs  
   
   def updateChains(self, nchains, vchains, hchains, nsteps):
      wt = self.w.transpose()
      for i in xrange(nchains):
      
         for j in xrange(nsteps):
            nvp = self.bv.copy()
            nvp += dot(self.w,hchains[i])
            nvp = self.logistic(nvp)
            vchains[i] = random.binomial(1, nvp, self.vsize)
            nhp = self.bh.copy()
            nhp += dot(wt, vchains[i])
            nhp = self.logistic(nhp)
            hchains[i] = random.binomial(1, nhp, self.hsize)  
      return vchains, hchains
   
   def postTrain(self, vs, hs):
      pass
      
   def trainAutonomous(self, data, epochs, nchains=100, nsteps=1):
      """
         data is 3-d
           d1 = batch num
           d2 = example num
           d3 = input index
      """
      hchains = random.random_sample((nchains, self.hsize)).round()
      vchains = random.random_sample((nchains, self.vsize)).round()
      vchains, hchains = self.updateChains(nchains, vchains, hchains, 1000) 
      mo = 0.5
      for i in xrange(epochs):
         if i > 5:
            mo = 0.9
         lr = self.rate / (i+1.)
         err = 0.0
         for k, vs in enumerate(data):
            n = shape(vs)[0]
            
            hs = self.meanFieldUp(n, vs)
            #gvs = self.meanFieldDown(n, hs)
            #err += sqrt((abs(vs - gvs) ** 2).sum(axis=1)).sum() / len(data)
            sn = sqrt(n)
            vsn = vs / sn
            hsn = hs / sn
            
            # compute correlation matrices
            ws = tensordot(vsn, hsn, axes=(0,0))
            bvs = vs.sum(axis=0) / n
            bhs = hs.sum(axis=0) / n
            
            # update the gibbs chains
            vchains, hchains = self.updateChains(nchains, vchains, hchains, nsteps) 
                  
     
            # compute sample correlations
            snchains = sqrt(nchains)
            vchainsn = vchains / snchains
            hchainsn = hchains / snchains
            ws2 = tensordot(vchainsn, hchainsn, axes=(0,0))
            bvs2 = vchains.sum(axis=0) / nchains
            bhs2 = hchains.sum(axis=0) / nchains
            
            # compute gradient
            dw = ws - ws2
            self.wc += lr * dw
            self.bvc += lr * (bvs -  bvs2)
            self.bhc += lr * (bhs - bhs2)
            self.w += self.wc
            self.bv += self.bvc
            self.bh += self.bhc
            self.wc *= mo
            self.bv *= mo
            self.bh *= mo
            
            err += (abs(dw).sum() / self.vsize / self.hsize) / 3.
            
            self.postTrain(vs, hs)
            
            gc.collect()
            print "\tBatch: ", i, k, ": ", err / (k+1.)
         print "Epoch ", i, ": ", err / len(data)

class rbmGL(rbm):
   def __init__(self, vsize, hsize):
      super(rbmGL, self).__init__(vsize, hsize)
      self.vsig = 0.5 * ones(vsize)

      
   def energy(self, x, useAlternate=False):
      """ compute the energy function """
      v = x[:self.vsize].toArray(self.vsize)
      h = x[self.vsize:self.vsize+self.hsize].toArray(self.hsize)
      extra = ((self.vsig * v) ** 2).sum()
      ret = -(extra + dot(v, dot(self.w, h)) + dot(v, self.bv) + dot(h, self.bh))
      return ret
      
   def meanFieldDown(self, n, hs):
      vs = random.random_sample((n,self.vsize))
      for j in xrange(n):
         for i in xrange(25):
            vs[j] = dot(self.w, hs[j]) + self.bv
            vs[j] = vs[j] * (self.vsig ** 2) + random.randn(self.vsize) * self.vsig
      gc.collect()
      return vs  
   
   def updateChains(self, nchains, vchains, hchains, nsteps):
      wt = self.w.transpose()
      for i in xrange(nchains):
      
         for j in xrange(nsteps):
            nvp = self.bv.copy()
            nvp += dot(self.w,hchains[i])
            nvp *= self.vsig ** 2
            vchains[i] = nvp + random.randn(self.vsize) * self.vsig
            nhp = self.bh.copy()
            nhp += dot(wt, vchains[i])
            nhp = self.logistic(nhp)
            hchains[i] = random.binomial(1, nhp, self.hsize)  
      return vchains, hchains

   def postTrain(self, vs, hs):
      # update variance
      gvs = self.meanFieldDown(shape(vs)[0], hs)
      var = sqrt(((vs - gvs) ** 2).sum(axis=0) / shape(vs)[0])
      self.vsig = 0.95 * self.vsig + 0.05 * var

class DeepBeliefNet(object):
   def __init__(self, sizes):
      self.sizes = sizes
      self.stack = []
      
   def wrap(self, data):
      wrapped = []
      for vs in data:
         for i, r in enumerate(self.stack):
            nvs = r.meanFieldUp(shape(vs)[0], vs)
            if i != 0:
               del vs
            vs = nvs
         wrapped.append(vs)
      return wrapped

   def train(self, depth, data, epochs, nchains=100, nsteps=5):
      while len(self.stack) < depth:
         print "Training at depth ", len(self.stack)
         vsize = self.sizes[len(self.stack)]
         hsize = self.sizes[len(self.stack)+1]
         if len(self.stack) > 0:
            lsize = self.sizes[len(self.stack)-1]
            r = rbm(vsize, hsize)
            # need to invert (lsize,vsize) to (vsize,hsize)
            # either remove columns or pad with 0s
            r.bv = self.stack[-1].bh
            if lsize < hsize:
               r.w = zeros((vsize, hsize))
               r.bh = zeros(hsize)
               r.w[:, :lsize] = self.stack[-1].w.transpose()
               r.bh[:lsize] = self.stack[-1].bv
            else:
               #take the middle
               low = (lsize - hsize) / 2
               high = low + hsize
               r.w = self.stack[-1].w.transpose()[:,low:high]
               r.bh = self.stack[-1].bv[low:high]
         else:
            r = rbm(vsize, hsize) 
         wrapped = self.wrap(data)
         r.trainAutonomous(wrapped, epochs, nchains, nsteps)
         del wrapped
         gc.collect()
         self.stack.append(r)
      print "Trained ", depth, "levels"
      
