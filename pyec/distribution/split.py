"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import numpy as np

from pyec.distribution.basic import PopulationDistribution
from pyec.config import Config
from pyec.history import MultipleHistory


class Splitter(PopulationDistribution):
    """Build an optimizer from two or more optimizers. Uses the
    ``weight`` property of the suboptimizers to deterministically
    apportion the population among the suboptimizers.
   
    :param subs: The optimizer instances to combine
    :type subs: A list of :class:`PopulationDistribution` objects
   
    """
    config = Config()

    def __init__(self, subs, **kwargs):
        if not len(subs):
             raise ValueError("Splitter requires at least one optimizer")
          
        kwargs['history'] = self.makeHistory(subs)
        super(Splitter, self).__init__(**kwargs)
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
    
    def mapHistory(self, sub, history=None):
        """Find a compatible subhistory for this suboptimizer.
      
        :param sub: One of the suboptimizers for this convolution
        :type sub: :class:`PopulationDistribution`
        :returns: A compatible :class:`History` for the suboptimizer
      
        """
        if history is None:
            history = self.history
        for h in history.histories:
            if sub.compatible(h):
                return h
        c = sub.__class__.__name__  
        raise ValueError("No compatible history found for {0}".format(c ))
    
    def needsScores(self):
        return self.useScores
        
    def batch(self, popSize):
        # Cannot assume that the weight of each sub stays constant
        total = 0.0
        for sub in self.subs:
            total += sub.weight
        
        pop = []
        
        for sub in self.subs:
           portion = int((sub.weight / total) * popSize)
           pop.extend(sub.batch(portion))
        
        idx = 0
        while len(pop) < popSize:   
           pop.append(self.subs[idx].sample())
           idx += 1
           
        return pop
            
    def update(self, history, fitness):
        super(Splitter, self).update(history, fitness)
        for sub in self.subs:
            sub.update(self.mapHistory(sub), fitness)
        