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
import gc

class BoltzmannMachine(Distribution):
   center = 0.5
   scale = 0.5
   
   def __init__(self, vsize, hsize, lr=.001):
      self.vsize = vsize
      self.hsize = hsize
      self.dim = vsize + hsize
      self.rate = lr
      self.w = 0.01 * random.randn(vsize, hsize)
      self.bv = zeros(vsize)
      self.bh = zeros(hsize)
      self.lv = 0.01 * random.randn(vsize, vsize)
      self.lh = 0.01 * random.randn(hsize, hsize)
      self.wc = zeros((vsize,hsize))
      self.bvc = zeros(vsize)
      self.bhc = zeros(hsize)
      self.lvc = zeros((vsize, vsize))
      self.lhc = zeros((hsize, hsize))
      for i in xrange(hsize): 
         for j in xrange(i+1):
            self.lh[i,j] = 0.0
      for i in xrange(vsize): 
         for j in xrange(i+1):
            self.lv[i,j] = 0.0
      

   def __call__(self, x):
      return self.energy(x)   
   
   def complete(self, vs, dummy=False):
      hs = self.meanFieldUp(len(vs), vs)
      xs = [append(vs[i], h, axis=0) for i,h in enumerate(hs)]
      return [TernaryString.fromArray(random.binomial(1, x, shape(x))) for x in xs] 
   
   def energy(self, x, useAlternate=False):
      """ compute the energy function """
      if useAlternate: return self.energy2(x)
      v = x[:self.vsize].toArray(self.vsize)
      h = x[self.vsize:self.vsize+self.hsize].toArray(self.hsize)
      ret = -(dot(v, dot(self.w, h)) + dot(h, dot(self.lh, h)) + dot(v, dot(self.lv, v)) + dot(v, self.bv) + dot(h, self.bh))
      return ret
   
   def logistic(self, x):
      x = minimum(maximum(x, -10.), 10.)
      return 1. / (1. + exp(-x))   
   
   def dot3(self, w, nx):
      nxpr = tile(nx, (shape(nw)[1], 1, 1))
      nxpr = swapaxes(nxpr, 0, 1)
      return (nw * nxpr).sum(axis=2)
                     
   def meanFieldUp(self, n, vs):
      wt = self.w.transpose()
      hs = random.random_sample((n,self.hsize))
      for j in xrange(n):
         for i in xrange(25):
            hs[j] = self.logistic(dot(wt, vs[j]) + dot(self.lh, hs[j]) + self.bh)
      gc.collect()
      return hs   
   
   def meanFieldDown(self, n, hs):
      vs = random.random_sample((n,self.vsize))
      for j in xrange(n):
         for i in xrange(25):
            vs[j] = self.logistic(dot(self.w, hs[j]) + dot(self.lv, vs[j]) + self.bv)
      gc.collect()
      return vs  
   
   def updateChains(self, nchains, vchains, hchains, nsteps):
      wt = self.w.transpose()
      for i in xrange(nchains):
      
         for j in xrange(nsteps):
            nvp = self.bv.copy()
            nvp += dot(self.w,hchains[i])
            nvp += dot(self.lv,vchains[i]) 
            nvp = self.logistic(nvp)
            vchains[i] = random.binomial(1, nvp, self.vsize)
            nhp = self.bh.copy()
            nhp += dot(wt, vchains[i])
            nhp += dot(self.lh, hchains[i])
            nhp = self.logistic(nhp)
            hchains[i] = random.binomial(1, nhp, self.hsize)  
      gc.collect()
      return vchains, hchains
   
   def train(self, data, epochs, nchains=100, nsteps=1):
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
            sn = sqrt(n)
            vsn = vs / sn
            hsn = hs / sn
            
            # compute correlation matrices
            ws = tensordot(vsn, hsn, axes=(0,0))
            lvs = tensordot(vsn, vsn, axes=(0,0))
            lhs = tensordot(hsn, hsn, axes=(0,0))
            bvs = vs.sum(axis=0) / n
            bhs = hs.sum(axis=0) / n
            
            # update the gibbs chains
            vchains, hchains = self.updateChains(nchains, vchains, hchains, nsteps) 
                  
     
            # compute sample correlations
            snchains = sqrt(nchains)
            vchainsn = vchains / snchains
            hchainsn = hchains / snchains
            ws2 = tensordot(vchainsn, hchainsn, axes=(0,0))
            lvs2 = tensordot(vchainsn, vchainsn, axes=(0,0))
            lhs2 = tensordot(hchainsn, hchainsn, axes=(0,0))
            bvs2 = vchains.sum(axis=0) / nchains
            bhs2 = hchains.sum(axis=0) / nchains
            
            # compute gradient
            dw = ws - ws2
            dv = lvs - lvs2
            dh = lhs - lhs2
            self.wc += lr * dw
            self.lvc += lr * dv
            self.lhc += lr * dh
            self.bvc += lr * (bvs -  bvs2)
            self.bhc += lr * (bhs - bhs2)
            for x in xrange(self.hsize):
               for y in xrange(x+1):
                  self.lhc[x,y] = 0.0
            for x in xrange(self.vsize): 
               for y in xrange(x+1):
                  self.lvc[x,y] = 0.0
            self.w += self.wc
            self.lv += self.lvc
            self.lh += self.lhc
            self.bv += self.bvc
            self.bh += self.bhc
            self.wc *= mo
            self.lv *= mo
            self.lh *= mo
            self.bv *= mo
            self.bh *= mo
            
            err += (abs(dw).sum() / self.vsize / self.hsize + abs(dv).sum() / self.vsize / self.vsize + abs(dh).sum() /self.hsize / self.hsize) / 3.
            
            gc.collect()
         print "Epoch ", i, ": ", err / len(data)
            