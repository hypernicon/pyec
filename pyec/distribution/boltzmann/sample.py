"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from numpy import *
from pyec.util.TernaryString import TernaryString
from pyec.distribution.basic import BernoulliTernary
import struct, binascii
from time import time

class BoltzmannSampler(object):
   def batch(self, rbm, size, initial, useAlternate=False):
      return self.__call__(rbm, size, initial, useAlternate)
   
   
class RBMGibbsSampler(BoltzmannSampler):
   def __init__(self, burn, chains, binary=True, meanField = False):
      self.burn = burn
      self.chains = chains
      self.binary = binary
      self.meanField = meanField
      
   def __call__(self, rbm, size=1, initial=None, useAlternate=False, clampIdx=None):
      from pyec.util.TernaryString import TernaryString
      sample = []
      vsize = rbm.vsize
      hsize = rbm.hsize
      if useAlternate:
         w = rbm.wg
         bv = rbm.bvg
         bh = rbm.bhg
      else:
         w = rbm.w
         bv = rbm.bv
         bh = rbm.bh
      if clampIdx is None:
         clampIdx = 0
      if initial is not None:
         initialArr = initial.toArray(rbm.dim)
      for i in xrange(self.chains):
         if initial is not None:
            x = initialArr
         else:
            x = random.random_sample(vsize+hsize).round()
            h = dot(x[:vsize], w) + bh
            h = 1. / (1. + exp(-h))
            if self.meanField:
               x[vsize:] = h
            else:
               x[vsize:] = random.binomial(1, h, hsize)
         for j in xrange(self.burn + size / self.chains):
            print "Gibbs: chain ", i, j
            v = dot(w, x[vsize:]) + bv
            if self.meanField:
               x[clampIdx:vsize] = (1. / (1. + exp(-v)))[clampIdx:]
            else:
               x[clampIdx:vsize] = random.binomial(1, 1. / (1. + exp(-v)), vsize)[clampIdx:]
            h = dot(x[:vsize], w) + bh
            x[vsize:] = 1. / (1. + exp(-h))
            if self.binary and not self.meanField:
               x[vsize:] = random.binomial(1, x[vsize:], hsize)
            if j >= self.burn:
               if self.binary:
                  y = x
                  if self.meanField:
                     y = random.binomial(1, x, vsize+hsize)
                  sample.append(TernaryString.fromArray(y))
               else:
                  sample.append(x)
      return sample

class RBMHeatBathSampler(BoltzmannSampler):
   def __init__(self, burn, chains, binary=True, meanField = False):
      self.burn = burn
      self.chains = chains
      self.binary = binary
      self.meanField = meanField
      
   def __call__(self, rbm, size=1, initial=None, useAlternate=False, clampIdx=None):
      from pyec.util.TernaryString import TernaryString
      sample = []
      vsize = rbm.vsize
      hsize = rbm.hsize
      if useAlternate:
         w = rbm.wg
         bv = rbm.bvg
         bh = rbm.bhg
      else:
         w = rbm.w
         bv = rbm.bv
         bh = rbm.bh
      if clampIdx is None:
         clampIdx = 0
      if initial is not None:
         initialArr = initial.toArray(rbm.dim)
      for i in xrange(self.chains):
         if initial is not None:
            x = initialArr
         else:
            T0 = 1. / 4
            x = random.random_sample(vsize+hsize).round()
            h = dot(x[:vsize], w) + bh
            h = 1. / (1. + exp(-h*T0))
            if self.meanField:
               x[vsize:] = h
            else:
               x[vsize:] = random.binomial(1, h, hsize)
         for j in xrange(self.burn + size / self.chains):
            print "Gibbs: chain ", i, j
            T = maximum(( 3*self.burn + j ) / float(4*self.burn), 1.0)
            v = dot(w, x[vsize:]) + bv
            if self.meanField:
               x[clampIdx:vsize] = (1. / (1. + exp(-v*T)))[clampIdx:]
            else:
               x[clampIdx:vsize] = random.binomial(1, 1. / (1. + exp(-v*T)), vsize)[clampIdx:]
            h = dot(x[:vsize], w) + bh
            x[vsize:] = 1. / (1. + exp(-h*T))
            if self.binary and not self.meanField:
               x[vsize:] = random.binomial(1, x[vsize:], hsize)
            if j >= self.burn:
               if self.binary:
                  y = x
                  if self.meanField:
                     y = random.binomial(1, x, vsize+hsize)
                  sample.append(TernaryString.fromArray(y))
               else:
                  sample.append(x)
      return sample

class BMGibbsSampler(BoltzmannSampler):
   def __init__(self, burn, chains, binary=True, meanField = False):
      self.burn = burn
      self.chains = chains
      self.binary = binary
      self.meanField = meanField
      
   def __call__(self, rbm, size=1, initial=None, useAlternate=False, clampIdx=None):
      from pyec.util.TernaryString import TernaryString
      sample = []
      vsize = rbm.vsize
      hsize = rbm.hsize
      if useAlternate:
         w = rbm.wg
         bv = rbm.bvg
         bh = rbm.bhg
      else:
         w = rbm.w
         bv = rbm.bv
         bh = rbm.bh
      lv = rbm.lv
      lh = rbm.lh
      if clampIdx is None:
         clampIdx = 0
      if initial is not None:
         initialArr = initial.toArray(rbm.dim)
      for i in xrange(self.chains):
         if initial is not None:
            x = initialArr
         else:
            x = random.random_sample(vsize+hsize).round()
            h = dot(x[:vsize], w) + dot(lh, x[vsize:]) + bh
            h = 1. / (1. + exp(-h))
            if self.meanField:
               x[vsize:] = h
            else:
               x[vsize:] = random.binomial(1, h, hsize)
         for j in xrange(self.burn + size / self.chains):
            print "Gibbs: chain ", i, j
            #down -> up
            #T = maximum((self.burn + 0.) / self.burn - j, 1.0)
            v = dot(w, x[vsize:]) + dot(lv, x[:vsize]) + bv
            if self.meanField:
               x[clampIdx:vsize] = (1. / (1. + exp(-v)))[clampIdx:]
            else:
               x[clampIdx:vsize] = random.binomial(1, 1. / (1. + exp(-v)), vsize)[clampIdx:]
            h = dot(x[:vsize], w) + dot(lh, x[vsize:]) + bh
            x[vsize:] = 1. / (1. + exp(-h))
            if self.binary and not self.meanField:
               x[vsize:] = random.binomial(1, x[vsize:], hsize)
               
            # left -> right
            v = dot(w, x[vsize:]) + dot(x[:vsize], lv) + bv
            h = dot(x[:vsize], w) + dot(x[vsize:], lh) + bh
            if self.meanField:
               x[clampIdx:vsize] = (1. / (1. + exp(-v)))[clampIdx:]
               x[vsize:] = (1. / (1. + exp(-h)))
            else:
               x[clampIdx:vsize] = random.binomial(1, 1. / (1. + exp(-v)), vsize)[clampIdx:]
               x[vsize:] = random.binomial(1, 1. / (1. + exp(-h)), hsize)
            
            v = dot(w, x[vsize:]) + dot(lv, x[:vsize]) + bv
            h = dot(x[:vsize], w) + dot(lh, x[vsize:]) + bh
            if self.meanField:
               x[clampIdx:vsize] = (1. / (1. + exp(-v)))[clampIdx:]
               x[vsize:] = (1. / (1. + exp(-h)))[vsize:]
            else:
               x[clampIdx:vsize] = random.binomial(1, 1. / (1. + exp(-v)), vsize)[clampIdx:]
               x[vsize:] = random.binomial(1, 1. / (1. + exp(-h)), hsize)
            

            
                        
            if j >= self.burn:
               if self.binary:
                  y = x
                  if self.meanField:
                     y = random.binomial(1, x, vsize+hsize)
                  sample.append(TernaryString.fromArray(y))
               else:
                  sample.append(x)
      return sample


class RBMSwendsenWangSampler(BoltzmannSampler):
   def __init__(self, burn, chains, binary = True):
      self.burn = burn
      self.chains = chains
      self.binary = binary
      
   def __call__(self, rbm, size=1, initial = None, useAlternate=False):
      from pyec.util.TernaryString import TernaryString
      sample = []
      vsize = rbm.vsize
      hsize = rbm.hsize
      if useAlternate:
         w = rbm.wg
         bv = rbm.bvg
         bh = rbm.bhg
      else:
         w = rbm.w
         bv = rbm.bv
         bh = rbm.bh
      
      biases = append(bv, bh, axis=0) 
      wp = 1. - exp(-2 * abs(w) )
     
           
      if initial is not None:
         initialArr = initial.toArray(rbm.dim)
      start = time()
      #gibbs = RBMGibbsSampler(1000,1,False)
      for n in xrange(self.chains):
         start = time()
         if initial is not None:
            x = initialArr
         else:
            #x = gibbs(rbm, 1)[0].round()
            x = random.random_sample(vsize + hsize).round()
            h = dot(x[:vsize], w) + bh
            x[vsize:] = random.binomial(1, 1. / (1. + exp(-h)), hsize)
         last = copy(x)
         repeats = 0
         
         for l in xrange(self.burn + size / self.chains):
            print "SW: chain ", n, l
            startInner = time()
            clusters = array([i for i in xrange(vsize + hsize)], dtype=int)
            r = random.random_sample((vsize, hsize))
            for i in xrange(vsize):
               for j in xrange(hsize):
                  i2 = j + vsize
                  if (w[i,j] < 0.0 and x[i] != x[i2] and r[i,j] < wp[i,j]) or \
                   (w[i,j] > 0.0 and x[i] == x[i2] and r[i,j] < wp[i,j]):
                  #if d[i,j] > 0.0:
                     if clusters[i] != clusters[i2]:
                        # take the lesser number
                        if clusters[i] < clusters[i2]:
                           c1 = clusters[i]
                           c2 = clusters[i2]
                        else:
                           c1 = clusters[i2]
                           c2 = clusters[i]
                        ifi2 = maximum(1 - abs(clusters - c2), 0)
                        noti2 = 1 - ifi2
                        clusters = ifi2 * c1 + noti2 * clusters
            
            computed = zeros(x.size)
            for k in xrange(x.size):
               if computed[k] > 0.0:
                  continue
               
               incluster = maximum(1 - abs(clusters - clusters[k]), 0)
               bias = (incluster * biases * x).sum()
               if self.binary:
                  state = float(random.binomial(1, 1. / (1. + exp(-bias)), 1))
               else:
                  state = float(1. / (1. + exp(-bias)))
               computed += incluster 
               x = incluster * state + (1-incluster) * x
                  
            if l >= self.burn:
               if self.binary:
                  sample.append(TernaryString.fromArray(x))
               else:
                  sample.append(x)
                  x = random.binomial(1, state, x.size)
         print "SW sample ", n, ": ", time() - start   
         
      return sample    

class BMSwendsenWangSampler(BoltzmannSampler):
   def __init__(self, burn, chains, binary = True):
      self.burn = burn
      self.chains = chains
      self.binary = binary
      
   def __call__(self, rbm, size=1, initial = None, useAlternate=False):
      from pyec.util.TernaryString import TernaryString
      sample = []
      vsize = rbm.vsize
      hsize = rbm.hsize
      if useAlternate:
         w = rbm.wg
         bv = rbm.bvg
         bh = rbm.bhg
      else:
         w = rbm.w
         bv = rbm.bv
         bh = rbm.bh
      lv = rbm.lv
      lh = rbm.lh
      
      biases = append(bv, bh, axis=0) 
      wp = 1. - exp(-2 * w )
      wn =  1. - exp(2 * w )
      lvp = 1. - exp(-2 * lv)
      lvn = 1. - exp(2 * lv)
      lhp = 1. - exp(-2 * lh)
      lhn = 1. - exp(2 * lh)
      
            
      if initial is not None:
         initialArr = initial.toArray(rbm.dim)
      start = time()
      for n in xrange(self.chains):
         start = time()
         if initial is not None:
            x = initialArr
         else:
            x = random.random_sample(vsize + hsize).round()
            h = dot(x[:vsize], w) + bh
            x[vsize:] = random.binomial(1, 1. / (1. + exp(-h)), hsize)
         last = copy(x)
         repeats = 0
         for l in xrange(self.burn + size / self.chains):
            print "SW: chain ", n, l
            startInner = time()
            clusters = array([i for i in xrange(vsize + hsize)], dtype=int)
            r = random.random_sample((vsize, hsize))
            rh = random.random_sample((hsize, hsize))
            rv = random.random_sample((vsize, vsize))
            for i in xrange(vsize):
               for j in xrange(hsize):
                  i2 = j + vsize
                  if (w[i,j] < 0.0 and x[i] != x[i2] and r[i,j] < wn[i,j]) or \
                   (w[i,j] > 0.0 and x[i] == x[i2] and r[i,j] < wp[i,j]):
                  #if d[i,j] > 0.0:
                     if clusters[i] != clusters[i2]:
                        # take the lesser number
                        if clusters[i] < clusters[i2]:
                           c1 = clusters[i]
                           c2 = clusters[i2]
                        else:
                           c1 = clusters[i2]
                           c2 = clusters[i]
                        ifi2 = maximum(1 - abs(clusters - c2), 0)
                        noti2 = 1 - ifi2
                        clusters = ifi2 * c1 + noti2 * clusters
            
            for i in xrange(vsize):
               for j in xrange(vsize):
                  if i == j: continue
                  i2 = j
                  if (lv[i,j] < 0.0 and x[i] != x[i2] and rv[i,j] < lvn[i,j]) or \
                   (lv[i,j] > 0.0 and x[i] == x[i2] and rv[i,j] < lvp[i,j]):
                  #if d[i,j] > 0.0:
                     if clusters[i] != clusters[i2]:
                        # take the lesser number
                        if clusters[i] < clusters[i2]:
                           c1 = clusters[i]
                           c2 = clusters[i2]
                        else:
                           c1 = clusters[i2]
                           c2 = clusters[i]
                        ifi2 = maximum(1 - abs(clusters - c2), 0)
                        noti2 = 1 - ifi2
                        clusters = ifi2 * c1 + noti2 * clusters
            
            for i in xrange(hsize):
               for j in xrange(hsize):
                  if i == j: continue
                  i2 = j
                  if (lh[i,j] < 0.0 and x[i] != x[i2] and rh[i,j] < lhn[i,j]) or \
                   (lh[i,j] > 0.0 and x[i] == x[i2] and rh[i,j] < lhp[i,j]):
                  #if d[i,j] > 0.0:
                     if clusters[i] != clusters[i2]:
                        # take the lesser number
                        if clusters[i] < clusters[i2]:
                           c1 = clusters[i]
                           c2 = clusters[i2]
                        else:
                           c1 = clusters[i2]
                           c2 = clusters[i]
                        ifi2 = maximum(1 - abs(clusters - c2), 0)
                        noti2 = 1 - ifi2
                        clusters = ifi2 * c1 + noti2 * clusters
            
            computed = zeros(x.size)
            for k in xrange(x.size):
               if computed[k] > 0.0:
                  continue
               
               
               incluster = maximum(1 - abs(clusters - clusters[k]), 0)
               bias = (incluster * biases).sum()
               if self.binary:
                  state = float(random.binomial(1, 1. / (1. + exp(-bias)), 1))
               else:
                  state = float(1. / (1. + exp(-bias)))
               computed += incluster 
               x = incluster * state + (1-incluster) * x
                  
            if l >= self.burn:
               if self.binary:
                  sample.append(TernaryString.fromArray(x))
               else:
                  sample.append(x)
                  x = random.binomial(1, state, x.size)
         print "SW sample ", n, ": ", time() - start   
         
      return sample    
      
class RBMSimulatedTempering(BoltzmannSampler):
   def __init__(self, burn, chains = 1, binary=True, temps = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.5,  15., 17.5, 20., 22.5, 25., 27.5, 30., 32.5, 35., 37.5, 40., 42.5, 45., 47.5, 50., 55., 60., 65., 70., 75., 80., 85., 90., 95., 100., 112.5, 125., 137.5, 150., 175., 200., 225., 250., 275., 500., 625., 750., 875., 1000.]):
      self.burn = burn
      self.chains = chains
      self.temps = array(temps)
      self.steps = array([i+1. for i in xrange(len(temps))])
      self.weights = None
      ps = self.temps[:-1] / self.temps[1:]
      self.pdown =  ps[:-1]/ ps[:-1] + ps[1:]
      self.pdown = append(zeros(1), self.pdown, axis=0)
      self.pdown = append(self.pdown, zeros(1), axis=0)
      self.binary = binary
   
   def __call__(self, rbm, size=1, initial = None, useAlternate=False):
      sample = []
      uniform = BernoulliTernary(rbm)
      if self.weights is None:
         wsample = uniform.batch(10000)
         self.weights = []
         base = array([exp(-rbm.energy(x, useAlternate)) for x in wsample]) 
         for temp in self.temps:
            self.weights.append(((base ** (1. / temp)) / 10000.).sum())
      
      
      for n in xrange(self.chains):
         # initialize the chain
         x = random.random_sample(rbm.vsize + rbm.hsize)
         idx = 1
         for l in xrange(self.burn + size / self.chains):
            print "ST: chain", n, l
            if self.binary:
               yv = random.binomial(1, x[:rbm.vsize], rbm.vsize)
            else:
               yv = x[:rbm.vsize]
            yh = dot(yv, rbm.w) + rbm.bh
            yh = 1. / (1. + exp(-yh / self.temps[idx]))
            yh = random.binomial(1, yh, rbm.hsize)
            yv = dot(rbm.w, yh) + rbm.bv
            yv = 1. / (1. + exp(-yv / self.temps[idx]))
            if self.binary:
               yv = random.binomial(1, yv, rbm.vsize)
            y = append(yv, yh, axis=0)
            
            
            ratio = exp((rbm.energy(TernaryString.fromArray(x), useAlternate)-rbm.energy(TernaryString.fromArray(y),useAlternate)) / self.temps[idx]) 
            if random.random_sample() < ratio:
               x = y
            ex = -rbm.energy(TernaryString.fromArray(x))
            
            if random.random_sample() < self.pdown[idx]:
               idx2 = idx - 1
               p1 = 1 - self.pdown[idx2]
               p2 = self.pdown[idx]
            else:
               idx2 = idx + 1
               p1 = self.pdown[idx2]
               p2 = 1. - self.pdown[idx]
            e1 = exp(ex / self.temps[idx]) * p1 * self.weights[idx]
            e2 = exp(ex / self.temps[idx2]) * p2 * self.weights[idx2]
            if random.random_sample() < e2 / e1:
               idx = idx2 
            if l >= self.burn:
               sample.append(x)
      return sample


class BMSimulatedTempering(BoltzmannSampler):
   def __init__(self, burn, chains = 1, binary=True, temps = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0,  15., 20., 25., 30., 35., 40., 45., 50., 60., 70., 80., 90., 100., 125., 150., 200., 250., 500., 750., 1000.]):
      self.burn = burn
      self.chains = chains
      self.temps = array(temps)
      self.steps = array([i+1. for i in xrange(len(temps))])
      self.weights = None
      ps = self.temps[:-1] / self.temps[1:]
      self.pdown =  ps[:-1]/ ps[:-1] + ps[1:]
      self.pdown = append(zeros(1), self.pdown, axis=0)
      self.pdown = append(self.pdown, zeros(1), axis=0)
   
   def __call__(self, rbm, size=1, initial = None, useAlternate=False):
      sample = []
      uniform = BernoulliTernary(rbm)
      if self.weights is None:
         wsample = uniform.batch(10000)
         self.weights = []
         base = array([exp(-rbm.energy(x, useAlternate)) for x in wsample]) 
         for temp in self.temps:
            self.weights.append(((base ** (1. / temp)) / 10000.).sum())
      
      
      for n in xrange(self.chains):
         # initialize the chain
         x = random.random_sample(rbm.vsize + rbm.hsize)
         idx = 1
         for l in xrange(self.burn + size / self.chains):
            print "ST: chain", n, l
            if self.binary:
               yv = random.binomial(1, x[:rbm.vsize], rbm.vsize)
            else:
               yv = x[:rbm.vsize]
            yh = dot(yv, rbm.w) + dot(rbm.lh, x[rbm.vsize:]) + rbm.bh
            yh = 1. / (1. + exp(-yh / self.temps[idx]))
            yh = random.binomial(1, yh, rbm.hsize)
            yv = dot(rbm.w, yh) + dot(rbm.lv, yv) +rbm.bv
            yv = 1. / (1. + exp(-yv / self.temps[idx]))
            if self.binary:
               yv = random.binomial(1, yv, rbm.vsize)
            y = append(yv, yh, axis=0)
            
            
            ratio = exp((rbm.energy(TernaryString.fromArray(x), useAlternate)-rbm.energy(TernaryString.fromArray(y),useAlternate)) / self.temps[idx]) 
            if random.random_sample() < ratio:
               x = y
            ex = -rbm.energy(TernaryString.fromArray(x))
            
            if random.random_sample() < self.pdown[idx]:
               idx2 = idx - 1
               p1 = 1 - self.pdown[idx2]
               p2 = self.pdown[idx]
            else:
               idx2 = idx + 1
               p1 = self.pdown[idx2]
               p2 = 1. - self.pdown[idx]
            e1 = exp(ex / self.temps[idx]) * p1 * self.weights[idx]
            e2 = exp(ex / self.temps[idx2]) * p2 * self.weights[idx2]
            if random.random_sample() < e2 / e1:
               idx = idx2 
            if l >= self.burn:
               if self.binary:
                  sample.append(TernaryString.fromArray(x))
               else:
                  sample.append(x)
      return sample



class RBMParallelTempering(BoltzmannSampler):
   def __init__(self, burn, chains = 1, binary=True, temps = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0,  15., 20., 25., 30., 35., 40., 45., 50., 60., 70., 80., 90., 100., 125., 150., 200., 250., 500., 750., 1000.]):
      self.burn = burn
      self.chains = chains
      self.temps = array(temps)  
      self.binary = binary

   def __call__(self, rbm, size=1, initial = None, useAlternate=False):
      sample = []
      for n in xrange(self.chains):
         # initialize the chain
         x = random.random_sample((len(self.temps), rbm.vsize + rbm.hsize))
         idx = 1
         for l in xrange(self.burn + size / self.chains):
            print "PT: chain", n, l
            y = x.copy()
            ex = zeros(len(self.temps))
            for i, t in enumerate(self.temps):
               if self.binary:
                  yv = random.binomial(1, x[i,:rbm.vsize], rbm.vsize)
               else:
                  yv = x[i,:rbm.vsize]
               yh = dot(yv, rbm.w) + rbm.bh
               yh = 1. / (1. + exp(-yh / t))
               yh = random.binomial(1, yh, rbm.hsize)
               yv = dot(rbm.w, yh) +rbm.bv
               yv = 1. / (1. + exp(-yv / t))
               if self.binary:
                  yv = random.binomial(1, yv, rbm.vsize)
               y[i] = append(yv, yh, axis=0)
            
            
               ratio = exp((rbm.energy(TernaryString.fromArray(x[i]), useAlternate)-rbm.energy(TernaryString.fromArray(y[i]),useAlternate)) / self.temps[idx]) 
               if random.random_sample() < ratio:
                  x[i] = y[i]
               ex[i] = rbm.energy(TernaryString.fromArray(x[i]))
            
            
            #shift temps
            for j in xrange(len(self.temps) - 1):
               j1 = j
               j2 = j+1
               e1 = ex[j1]
               e2 = ex[j2]
               ratio = exp((e1 - e2) * (1./self.temps[j1] - 1./self.temps[j2]))
               if random.random_sample() < ratio:
                  z = x[j1]
                  x[j1] = x[j2]
                  x[j2] = z
            
            if l >= self.burn:
               if self.binary:
                  sample.append(TernaryString.fromArray(x[0]))
               else:
                  sample.append(x[0])
      return sample   
 
class BMParallelTempering(BoltzmannSampler):
   def __init__(self, burn, chains = 1, binary=True, temps = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0,  15., 20., 25., 30., 35., 40., 45., 50., 60., 70., 80., 90., 100., 125., 150., 200., 250., 500., 750., 1000.]):
      self.burn = burn
      self.chains = chains
      self.temps = array(temps)
      self.steps = array([i+1. for i in xrange(len(temps))])
      self.binary = binary   

   def __call__(self, rbm, size=1, initial = None, useAlternate=False):
      sample = []
      for n in xrange(self.chains):
         # initialize the chain
         x = random.random_sample((len(self.temps), rbm.vsize + rbm.hsize))
         idx = 1
         for l in xrange(self.burn + size / self.chains):
            print "PT: chain", n, l
            y = x.copy()
            ex = zeros(len(self.temps))
            for i, t in enumerate(self.temps):
               if self.binary:
                  yv = random.binomial(1, x[i,:rbm.vsize], rbm.vsize)
               else:
                  yv = x[i,:rbm.vsize]
               yh = dot(yv, rbm.w) + dot(rbm.lh, x[i,rbm.vsize:]) + rbm.bh
               yh = 1. / (1. + exp(-yh / t))
               yh = random.binomial(1, yh, rbm.hsize)
               yv = dot(rbm.w, yh) + dot(rbm.lv, yv) +rbm.bv
               yv = 1. / (1. + exp(-yv / t))
               if self.binary:
                  yv = random.binomial(1, yv, rbm.vsize)
               y[i] = append(yv, yh, axis=0)
            
            
               ratio = exp((rbm.energy(TernaryString.fromArray(x[i]), useAlternate)-rbm.energy(TernaryString.fromArray(y[i]),useAlternate)) / self.temps[idx]) 
               if random.random_sample() < ratio:
                  x[i] = y[i]
               ex[i] = rbm.energy(TernaryString.fromArray(x[i]))
            
            
            #shift temps
            for j in xrange(len(self.temps) - 1):
               j1 = j
               j2 = j+1
               e1 = ex[j1]
               e2 = ex[j2]
               ratio = exp((e1 - e2) * (1./self.temps[j1] - 1./self.temps[j2]))
               if random.random_sample() < ratio:
                  z = x[j1]
                  x[j1] = x[j2]
                  x[j2] = z

            
            if l >= self.burn:
               if self.binary:
                  sample.append(TernaryString.fromArray(x[0]))
               else:
                  sample.append(x[0])
      return sample    
 
"""
This is not a sampler; it only computes marginals.

class RBMLoopyBeliefPropagation(BoltzmannSampler):
   def __init__(self, binary = True):
      self.binary = binary
      
   def __call__(self, rbm, size=1, initial = None, useAlternate=False):
      sample = []
      w = exp(rbm.w)
      wt = w.transpose()
      bv = outer(exp(rbm.bv), ones(rbm.hsize))
      bh = outer(exp(rbm.bh), ones(rbm.vsize))
      wbv = w * bv
      wbh = wt * bh
      
      # initialize messages
      mvh0 = ones((rbm.vsize, rbm.hsize))
      mvh1 = ones((rbm.vsize, rbm.hsize))
      mhv0 = ones((rbm.hsize, rbm.vsize))
      mhv1 = ones((rbm.hsize, rbm.vsize))
      for j in xrange(500):
         print "BP: ", j
         zv = outer(mhv0.prod(axis=0), ones(rbm.hsize)) / mhv0.transpose()
         pv = outer(mhv1.prod(axis=0), ones(rbm.hsize)) / mhv1.transpose()
         nmvh0 = zv + bv * pv
         nmvh1 = zv + wbv * pv
         zh = outer(mvh0.prod(axis=0), ones(rbm.vsize)) / mvh0.transpose()
         ph = outer(mvh1.prod(axis=0), ones(rbm.vsize)) / mvh1.transpose()
         nmhv0 = zh + bh * ph
         nmhv1 = zh + wbh * ph
         svh = nmvh0 + nmvh1
         shv = nmhv0 + nmhv1
         mvh0 = nmvh0 / svh
         mvh1 = nmvh1 / svh
         mhv0 = nmhv0 / shv
         mhv1 = nmhv1 / shv
      
      # compute belief
      bpv0 = mhv0.prod(axis=0)
      bpv1 = exp(rbm.bv) * mhv1.prod(axis=0)
      bpv = bpv1 / (bpv0 + bpv1)
         
      bph0 = mvh0.prod(axis=0)
      bph1 = exp(rbm.bh) * mvh1.prod(axis=0)
      bph = bph1 / (bph0 + bph1)
         
               
                        
      for i in xrange(size):   
         x = random.binomial(1, append(bpv, bph, axis=0), rbm.vsize + rbm.hsize)
         if self.binary:
            sample.append(TernaryString.fromArray(x))
         else:
            sample.append(x)
      return sample
      
"""      
