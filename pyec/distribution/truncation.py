"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import numpy as np

from pyec.distribution.basic import PopulationDistribution
from pyec.config import Config
from pyec.history import DelayedHistory

class TrajectoryTruncation(PopulationDistribution):
    """Build a optimizer with a truncated trajectory.
    
    :param sub: The subordinate optimizer
    :type sub: :class:`PopulationDistribution`
    :param delay: The number of steps to truncate
    :type delay: ``int`` 
    
    """
    config = Config()
    
    def __init__(self, sub, delay, **kwargs):
        kwargs['history'] = self.makeHistory(sub)
        super(TrajectoryTruncation, self).__init__(**kwargs)
        self.opt = sub
        self.delay = delay

    def makeHistory(self, sub):
        """Build a :class:`DelayedHistory` suitable for the subordinate
        optimizer
        
        :param sub: The subordinate optimizer
        :type sub: :class:`PopulationDistribution`
        :returns: A suitable :class:`DelayedHistory`object
        
        """
        def generator(config):
            return DelayedHistory(config,
                                  sub.config.history(sub.config),
                                  self.delay)
            
        return generator
        
    def update(self, history, fitness):
        super(TrajectoryTruncation, self).update(history, fitness)
        self.opt.update(history.history, fitness)
        return self        
        
    def batch(self, popSize):
        return self.opt.batch(popSize)

    def needsScores(self):
        return self.opt.needsScores()
    
    def compatible(self, history):
        return (isinstance(history, DelayedHistory) and
                self.opt.compatible(history.history))
                
