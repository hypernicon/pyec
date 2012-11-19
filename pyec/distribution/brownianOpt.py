"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from numpy import *
from scipy.integrate import quad, quadrature
from scipy.interpolate import interp1d, SmoothBivariateSpline, InterpolatedUnivariateSpline
from scipy.special import erf
from basic import PopulationDistribution
from pyec.config import Config, ConfigBuilder

import cPickle
import os.path

interpolatedBest = None
interpolatedMin = None
interpolatedMinSpl = None

def loadInterpolatedBest(cubic=False):
   """
      Good for range 0 to 5.0
      
      Equivalent to 
      
      def best_interpolate(y):
         opt = BrownianOpt(None)
         bbm = BoundedBrownianMotion(0,0,1,y)
         return opt.bestInt(bbm, bbm.x0, bbm.x1, bbm.expected_error(.5), 1e-10)
   """
   global interpolatedBest
   if interpolatedBest is not None:
      return interpolatedBest
   f = open(os.path.join(os.path.dirname(__file__), "brownian_opt_interpolates.dat"))
   xs = cPickle.load(f)
   ys = cPickle.load(f)
   f.close()
   if cubic:
      spl = InterpolatedUnivariateSpline(xs,ys) 
      interpolatedBest = lambda x: spl(x)[0] #interp1d(xs,ys,kind='cubic')
   else:
      interpolatedBest = interp1d(xs,ys)
   return interpolatedBest

def loadInterpolatedMin(degree=5):
   """
      Good for range 0 to 1, 0 to 5.0
      
      Equivalent to 
      
      def min_interpolate(x,y):
         return BoundedBrownianMotion(0,0,1,y).expected_error(x)
   """
   global interpolatedMin, interpolatedMinSpl
   if interpolatedMin is not None:
      return interpolatedMin
   
   f = open(os.path.join(os.path.dirname(__file__), "brownian-opt-min-interpolates.dat"))
   xys = cPickle.load(f)
   zs = cPickle.load(f)
   f.close()
   interpolatedMinSpl = SmoothBivariateSpline([x for x,y in xys],[y for x,y in xys],zs,kx=degree,ky=degree)
   interpolatedMin = lambda x,y: interpolatedMinSpl(x,y)[0,0]
   return interpolatedMin

   


class BoundedBrownianMotion(object):
   def __init__(self, x0,f0,x1,f1):
      self.x0 = x0
      self.x1 = x1
      self.f0 = f0
      self.f1 = f1
      self.x = x0
      self.f = f0
      self.min = min([f0,f1])
      self.minx = self.min == f0 and x0 or x1
      loadInterpolatedMin()


   def mu(self, z):
      return (self.f0 * self.x1 + (self.f1 - self.f0)*z - self.f1 * self.x0) / (self.x1 - self.x0)

   def sigma(self, z):
      sig = (self.x1 - z) * (z - self.x0) / (self.x1 - self.x0)
      if sig < 0:
         sig = 0
      return sqrt(sig)
      
   def var_min(self):
      z = ((self.f1 - self.f0)**2)/(self.x1-self.x0)
      y = self.x1 + self.x0
      a = -4*(1+z)
      b = 4*y*(1+z)
      c = -4*self.x0*self.x1*z - y*y
      rad = b*b - 4*a*c
      if rad < 0.0: rad = 0.0
      return (-b + sqrt(rad)) / (2*a)

   def g(self,x,y, x0, f0):
      ret =  min([f0,y]) - sqrt(2*pi*abs(x-x0))/4. * exp(((y-f0)**2)/(2. * abs(x-x0)))*(1. - erf(-2*(min([f0,y]) - .5 * (f0 + y))/sqrt(2*abs(x-x0))))
      if ret != ret:
         return min([f0,y])
      return ret

   def expected_minimum(self,x, useInterpolation=False):
      bbm = self
      x0 = self.x0
      x1 = self.x1
      f1 = self.f1
      f0 = self.f0
      
      if useInterpolation:
         scaleFactor = sqrt(x1-x0)
         if f0 < f1:
            scalePos = (x-x0)/(x1-x0)
         else:
            scalePos = (x1-x)/(x1-x0)
         return interpolatedMin(scalePos, abs(f1-f0)/scaleFactor)*scaleFactor + min([f0,f1])
      
      mux = float(bbm.mu(x))
      sigx = float(bbm.sigma(x))
      fmin = min([f0,f1])
      integrand = lambda y: min([self.g(x,y,x0,f0),self.g(x,y,x1,f1)])*exp(-((y-mux)**2)/(2*(sigx**2)))/sqrt(2*pi*(sigx**2))
      integral = quad(integrand,mux-10*sigx,mux+10*sigx,limit=100, full_output=1)[0]
      if integral > fmin:
         print x0, f0
         print x1, f1
         print mux, sigx
         print fmin, integral
         print integrand(mux - sigx), integrand(mux), integrand(mux + sigx)
         step = 6*sigx/100000.
         pts = array([integrand(z)*step for z in arange(mux-10*sigx, mux+10*sigx, step)])
         print pts.sum()
         raise Exception, "Overestimated expected minimum!"
      return integral
   
   def expected_error(self, x):
      return min([self.f0,self.f1]) - self.expected_minimum(x)




class BrownianOptConfigurator(ConfigBuilder):
   def __init__(self, *args):
      super(BrownianOptConfigurator, self).__init__(BrownianOpt)
      self.cfg.useInterpolation = True
      self.cfg.useVarMin = False
      
class BrownianOpt(PopulationDistribution):
   
   def __init__(self, cfg):
      super(BrownianOpt, self).__init__(cfg)
      self.map = []
      self.cache = {}
      center, scale = self.config.center, self.config.scale
      if self.config.bounded and hasattr(self.config.in_bounds, 'extent'):
         center, scale = self.config.in_bounds.extent()
      self.fscale = sqrt(2*scale)
      if self.config.useInterpolation:
         loadInterpolatedBest(True)
      
   @classmethod
   def configurator(cls):
      return BrownianOptConfigurator(cls)

   def bestInt(self, bbm, low, high, currentf, tol=1e-10):
      #print low, high, .5*(low + high), currentf
      if high - low < tol:
         return (low + high) / 2.
      lower = bbm.expected_error(low + (high - low) * .25)
      upper = bbm.expected_error(low + (high - low) * .75)
      if lower < currentf < upper:
         return self.bestInt(bbm, low, (low + high) * .5, lower)
      elif upper < currentf < lower:
         return self.bestInt(bbm, (high + low) * .5, high, upper)
      else:
         return self.bestInt(bbm, low + (high - low) * .25, low + (high - low) * .75, currentf) 

   def best(self, bbm):
      sx0 = str(bbm.x0)
      sx1 = str(bbm.x1)
      # Do an internal recursive search for the expected best
      if self.cache.has_key(sx0):
         if self.cache[sx0].has_key(sx1):
            return self.cache[sx0][sx1]
            
      if self.config.useVarMin:
         ret = bbm.x1 - (bbm.var_min() - bbm.x0)
      elif self.config.useInterpolation:
         x0 = bbm.x0
         x1 = bbm.x1
         f0 = bbm.f0
         f1 = bbm.f1
         if f1 >= f0:
            ret = x0 + (x1 - x0) * interpolatedBest(abs(f1 - f0)/sqrt(x1-x0))
         else:
            ret = x1 - (x1 - x0) * interpolatedBest(abs(f1 - f0)/sqrt(x1-x0))
      else:
         m = .5*(bbm.x0 + bbm.x1)
         mid = bbm.expected_error(m)
        
         ret = self.bestInt(bbm, bbm.x0, bbm.x1, mid, tol=(bbm.x1-bbm.x0)*1e-5)
      
      gain = -bbm.expected_minimum(ret, self.config.useInterpolation)
      
      if not self.cache.has_key(sx0):
         self.cache[sx0] = {sx1:(ret,gain)}
      else:
         self.cache[sx0][sx1] = (ret,gain)
      return ret,gain

   def batch(self, popSize):
      center, scale = self.config.center, self.config.scale
      if self.config.bounded and hasattr(self.config.in_bounds, 'extent'):
         center, scale = self.config.in_bounds.extent()
      ret = self.innerBatch(popSize)
      return center - scale + 2*scale*ret

   def innerBatch(self, popSize):
      if len(self.map) == 0:
         return array([[0.0]])
      elif len(self.map) == 1:
         return array([[1.0]])
    
      # iterator of the intervals of the map
      maxf = -1e300
      maxx = self.map[0][0]
      scores = []
      for i in xrange(len(self.map) - 1):
         # choose the best point in each range, and score it
         x0,f0 = self.map[i]
         x1,f1 = self.map[i+1]
         if abs(x0 - x1) < 1e-20:
            continue 
         bbm = BoundedBrownianMotion(float(x0),-float(f0),float(x1),-float(f1))
         x,gain = self.best(bbm)
         f = gain # max([f0,f1]) + gain
         #print "best: ", x, x0, x1, f
         scores.append((x,f,x1-x0))
         if f > maxf:
            maxf = f
            maxx = x
      
      # implement a simple tournament
      ps = []
      i = 0
      s = 0
      for x,f,w in sorted(scores, key=lambda k: -k[1]):
         p = w * (.85 ** (i))
         i += 1
         s += p
         ps.append((p,x))
      prob = random.random_sample() * s
      cump = 0
      for p,x in ps:
         cump += p
         if cump > prob:
            self.var = x
            return array([[x]])
      
      # return the best             
      self.var = maxx
      return array([[maxx]])
      
   def update(self, generation, population):
      center, scale = self.config.center, self.config.scale
      if self.config.bounded and hasattr(self.config.in_bounds, 'extent'):
         center, scale = self.config.in_bounds.extent()
      pop = [((x-center+scale) / (2*scale), s/self.fscale) for x,s in population]
      self.innerUpdate(generation, pop)
         
   def innerUpdate(self, generation, population):
      for x,f in population:
         self.map.append((x,f))
      self.map = sorted(self.map, key=lambda x: x[0])
