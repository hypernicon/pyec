"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from basic import PopulationDistribution


class Convolution(PopulationDistribution):
   depth = 0

   def __init__(self, subs, initial = None, passScores=False):
      super(Convolution, self).__init__(subs[0].config)
      self.subs = subs
      self.population = None
      self.initial = initial
      self.simple = []
      self.generation = 1
      self.passScores = passScores
      self.scoreDict = {}

   def score(self, population):
      return [(x,self.scoreDict.get(str(x), None)) for x in population]

   def batch(self, popSize):
      if self.population is None and self.initial is not None:
         return  self.initial.batch(popSize)
      if self.population is None:
         self.population = []
      scoredPopulation = self.population
      population = self.simple
      for i, sub in enumerate(self.subs):
         sub.update(self.generation, scoredPopulation)
         population = sub.batch(popSize)
         if self.passScores:
            scoredPopulation = self.score(population)
         else:
            scoredPopulation = [(x,None) for x in population]
      return population

   def update(self, generation, population):
      self.generation = generation
      self.population = population
      self.simple = [x for x,s in self.population]
      if self.passScores:
         self.scoreDict = dict([(str(x), s) for x,s in population])
