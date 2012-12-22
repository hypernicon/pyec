"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import numpy as np
from pyec.distribution.basic import Distribution
from pyec.distribution.ec.mutators import *
from pyec.distribution.ec.selectors import *
from pyec.config import Config as _
from pyec.space import Euclidean, EndogeneousProduct

import logging
log = logging.getLogger(__file__)

"""A (1+1)-ES; Note the use of EndogeneousProduct space"""
ES1_1 = (
   EvolutionStrategySelection[_(mu=1,selection="plus")] <<
   EndogeneousGaussian[_(sd=0.05)]
)[_(populationSize=2,
    space=EndogeneousProduct(Euclidean(dim=5), Euclidean(dim=5)))]

"""A (10,100)-ES"""
ES10_100 = (
   EvolutionStrategySelection[_(mu=10)] <<
   EndogeneousGaussian[_(sd=0.05)]
)[_(populationSize=100,
    space=EndogeneousProduct(Euclidean(dim=5), Euclidean(dim=5)))]

"""A (10/3,100)-ES with intermediate crossover"""
IntermediateCrossES10Slash3_100 = (
   EvolutionStrategySelection[_(mu=10)] <<
   ((EvolutionStrategySelection[_(mu=10)] >> 1) << 
    ((EvolutionStrategySelection[_(mu=10)] >> 2) <<
     Crossover[_(crosser=IntermediateCrosser,order=3)])) <<
   EndogeneousGaussian[_(sd=0.05)]
)[_(populationSize=100,
    space=EndogeneousProduct(Euclidean(dim=5), Euclidean(dim=5)))]

"""A (10/3,100)-ES with dominany crossover"""
DominantCrossES10Slash3_100 = (
   EvolutionStrategySelection[_(mu=10)] <<
   ((EvolutionStrategySelection[_(mu=10)] >> 1) << 
    ((EvolutionStrategySelection[_(mu=10)] >> 2) <<
     Crossover[_(crosser=DominantCrosser,order=3)])) <<
   EndogeneousGaussian[_(sd=0.05)]
)[_(populationSize=100,
    space=EndogeneousProduct(Euclidean(dim=5),
                             Euclidean(dim=5,center=0.05,scale=0.0)))]


class Cmaes1996Initial(Distribution):
   """Initialize the 1996 version of CMA-ES.
   
   """
   def buildRotation(self, dim, sd):
      ret = []
      for i in xrange(dim):
         for j in xrange(dim):
            if i == j:
               ret.append(sd)
            elif j > i:
               ret.append(0.0)
      return ret
   
   def sample(self):
      return [self.config.space.spaces[0].random(),
              self.buildRotation(self.config.space.spaces[0].dim,self.config.sd),
              np.zeros(self.config.space.spaces[2].dim),
              np.ones(self.config.space.spaces[3].dim),
              np.zeros(self.config.space.spaces[4].dim)]
      
   def batch(self, popSize):
      return [self.sample() for i in xrange(popSize)]
   
   
def makeCmaesConfig(populationSize, mu, dim, sd):
   """Convenience function to configure the 1996 version of CMA-ES
   
   """
   config = _(populationSize=populationSize,
              space=EndogeneousProduct(Euclidean(dim=dim),
                                       Euclidean(dim=dim * (dim+1) / 2),
                                       Euclidean(dim=dim),
                                       Euclidean(dim=dim),
                                       Euclidean(dim=dim)),
              sd=sd,
              mu=mu)
   config.initial = Cmaes1996Initial(**config.__properties__)
   return config

"""Old CMA-ES"""
Cmaes1996 = (
   EvolutionStrategySelection <<
   CorrelatedEndogeneousGaussian
)[makeCmaesConfig(100,10,5,0.05)]

