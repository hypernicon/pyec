"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from numpy import *
from pyec.distribution.basic import PopulationDistribution
from pyec.config import Config
from pyec.history import LocalMinimumHistory
from pyec.space import Euclidean

   

class DifferentialEvolution(PopulationDistribution):
   """
      Implements Differential Evolution (DE) as described by:
      
      Storn, Ranier and Price, Kenneth. Differential Evolution -- A simple and efficient adaptive scheme for global optimization over continuous spaces. 1995. 
      
      See <http://en.wikipedia.org/wiki/Differential_evolution> for algorithm details.    
      
      See <http://www.hvass-labs.org/people/magnus/publications/pedersen10good-de.pdf> for a good discussion of parameter settings for DE.

      Config parameters:
      
      * CR -- crossover probability (default .2) 
      * F -- the learning rate (default .5)
      
      Other defaults:
      
      * history -- :class:`LocalMinimumHistory`
      * space -- :class:`Euclidean`
      * populationSize -- 100
      * initial -- ``None``
      
      :param cfg: The configuration object for Differential Evolution.
      :type cfg: :class:`Config`

   """
   config = Config(CR=.2,
                   F=.5,
                   history= LocalMinimumHistory,
                   populationSize=100,
                   space=Euclidean(),
                   initial=None)

   def compatible(self, history):
      return hasattr(history, 'localBest')

   def batch(self, popSize):
      idx = 0
      ys = []
      xs = self.history.localBest()
      for x,s in xs:
         y = ones(self.config.space.dim)
         if True: 
            i1 = idx
            while i1 == idx:
               i1 = random.randint(0,len(xs))
            i2 = i1
            while i1 == i2 or i2 == idx:
               i2 = random.randint(0,len(xs))
            i3 = i2
            while i1 == i3 or i2 == i3 or i3 == idx:
               i3 = random.randint(0,len(xs))
         
            a, s1 = xs[i1]
            b, s2 = xs[i2]
            c, s3 = xs[i3]
         
            d = random.randint(0, len(x))
            y = copy(x)
            idx2 = 0
            for yi in y:
               r = random.random_sample()
               if idx2 == d or r < self.config.CR:
                  y[idx2] = a[idx2] + self.config.F * (b[idx2] - c[idx2]) 
               idx2 += 1 
         ys.append(y)
         idx += 1
      return ys
