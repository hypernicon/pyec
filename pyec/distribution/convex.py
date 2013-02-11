"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import numpy as np

from pyec.distribution.basic import PopulationDistribution, HistoryMapper
from pyec.config import Config
from pyec.history import MultipleHistory


class Convex(PopulationDistribution, HistoryMapper):
    """Build a convex optimizer from two or more optimizers. Uses the
    ``weight`` property of the suboptimizers to determine weights 
   
    :param subs: The optimizer instances to combine
    :type subs: A list of :class:`PopulationDistribution` objects
   
    """
    config = Config()

    def __init__(self, subs, **kwargs):
        if not len(subs):
             raise ValueError("Convex combination requires at least one optimizer")
          
        kwargs['history'] = self.makeHistory(subs)
        super(Convex, self).__init__(**kwargs)
        self.subs = subs
        self.history = None
        self.useScores = np.array([s.needsScores for s in self.subs]).any()
        
        # force any self-convolutions to self checkpoint
        # since they are now inside of another optimizer
        for sub in subs:
            if hasattr(sub, 'checkpoint') and not sub.checkpoint:
                sub.checkpoint = True
    
    def compatible(self, history):
        """A convolution is compatible with a CheckpointedHistory
        only if it contains a history that is compatible for each
        suboptimizer.
      
        """
        return (isinstance(history, MultipleHistory) and
          np.array([self.mapHistory(s, history) is not None 
                    for s in self.subs]).all()) 

    def makeHistory(self, subs):
        """Build a :class:`MultipleHistory` from the suboptimizers"""
        def generator(config):
            return MultipleHistory(config,
                                   *[opt.config.history for opt in subs])
        return generator
    
    def needsScores(self):
        return self.useScores
        
    def batch(self, popSize):
        pops = [sub() for sub in self.subs]
        pop = []
        for i in xrange(popSize):
            while len(pops) > 0:
                idx = np.random.randint(0, len(pops))
                if i < len(pops[idx]):
                    pop.append(pops[idx][i])
                    break
                else:
                    pops.remove(pop)
        
        return pop
            
    def update(self, history, fitness):
        super(Convex, self).update(history, fitness)
        for sub in self.subs:
            sub.update(self.mapHistory(sub), fitness)
            
    def chooseSub(self):
        # Cannot assume that the weight of each sub stays constant
        total = 0.0
        for sub in self.subs:
            total += sub.weight
            
        p = total * np.random.random_sample()
        total = 0.0
        for sub in self.subs:
            total += sub.weight
            if p < total:
               return sub
               
        raise ValueError("Failed to sample subs; problem with weights?")
        