"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from numpy import *
from pyec.config import Config
from pyec.distribution.bayes.net import BayesNet
from pyec.distribution.ec.mutators import Crosser, Mutation

class Merger(Crosser):
   def __call__(self, nets, prob):
      net = self.config.space.copy(nets[0])
      if random.random_sample() < prob:
         for net2 in nets[1:]:
            net.merge(net2, net.structureGenerator.config.data)
      return net

class UniformBayesCrosser(Crosser):
   def __call__(self, nets, prob):
      net = self.config.space.copy(nets[0])
      if random.random_sample() < prob:
         for net2 in nets[1:]:
            net.cross(net2, net.structureGenerator.config.data)
      return net
         
class StructureMutator(Mutation):
   config = Config(varDecay=1.0,
                   varExp=0.25)
   
   def __init__(self, **kwargs):
      super(StructureMutator, self).__init__(**kwargs)
      self.decay = 1.0

   def mutate(self, net):
      net = self.config.space.copy(net)
      net.decay = self.decay
      return net.structureSearch(net.structureGenerator.config.data)

   def update(self, history, fitness):
      super(StructureMutator, self).update(history, fitness)
      self.decay = exp(-self.history.updates *
                       self.config.varDecay) ** self.config.varExp

class AreaSensitiveStructureMutator(Mutation):
   config = Config(sd=1.0)
   
   def mutate(self, x):
      net, area = x
      net = self.config.space.copy(net)
      area = area.area
      net.decay = self.config.sd * log(1 + -log2(area))
      return net.structureSearch(net.structureGenerator.config.data)