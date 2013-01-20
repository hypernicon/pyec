"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import numpy as np
import pprint
import gc
import sys

from pyec.distribution.basic import PopulationDistribution, HistoryMapper
from pyec.config import Config
from pyec.history import CheckpointedHistory, CheckpointedMultipleHistory


class Convolution(PopulationDistribution, HistoryMapper):
   """Build a convolved optimizer.
   
   :param subs: The optimizer instances to convolve
   :type subs: A list of :class:`PopulationDistribution` objects
   
   """
   config = Config()
   
   def __init__(self, subs, **kwargs):
      if not len(subs):
          raise ValueError("Convolution requires at least one optimizer")
          
      kwargs["history"] = self.makeHistory(subs)
      super(Convolution, self).__init__(**kwargs)
      self.subs = subs
      self.history = None
      
      # force any self-convolutions to self checkpoint
      # since they are now inside of a convolution
      for sub in subs:
          if hasattr(sub, 'checkpoint') and not sub.checkpoint:
              sub.checkpoint = True
      
   def makeHistory(self, subs):
      def generator(config):
          hs = [opt.config.history for opt in subs]
          return CheckpointedMultipleHistory(config, *hs)
      return generator

   def compatible(self, history):
      """A convolution is compatible with a CheckpointedHistory
      only if it contains a history that is compatible for each
      suboptimizer.
      
      """
      return (isinstance(history, CheckpointedMultipleHistory) and
          np.array([self.mapHistory(s, history) is not None 
                    for s in self.subs]).all()) 

   def batch(self, popSize):
       pop = None
       self.history.checkpoint()
       for sub in self.subs:
           if pop is not None:
               fitness = sub.needsScores() and self.fitness or None
               self.history.update(pop, fitness, sub.config.space, sub)
           sub.update(self.mapHistory(sub), self.fitness)
           pop = sub()
             
       self.history.rollback()
       return pop
   
   def update(self, history, fitness):
      super(Convolution, self).update(history, fitness)
      for sub in self.subs:
          sub.update(self.mapHistory(sub), fitness)  
      
   def needsScores(self):
      return self.subs[0].needsScores()
      
      
class SelfConvolution(PopulationDistribution):
    """The convolution of an optimizer with itself.
    
    :param opt: The optimizer to convolve
    :type opt: :class:`PopulationDistribution`
    :param times: The number of times to self-convolve
    :type times: ``int``
    
    """
    config = Config()
    
    def __init__(self, opt, times, **kwargs):
        kwargs['history'] = self.makeHistory(opt)
        super(SelfConvolution, self).__init__(**kwargs)
        self.opt = opt
        self.times = times
        if times < 1:
           err = "Self convolution needs a positive integer; got {0}"
           err.format(times)
           raise ValueError(err)
        
        self.checkpoint = False
        
        # force any sub-selfconvolutions to checkpoint
        if hasattr(opt, 'checkpoint') and not opt.checkpoint:
            opt.checkpoint = True
    
    def makeHistory(self, opt):
      def generator(config):
          return CheckpointedHistory(config, opt.config.history)
      return generator    
                
    def compatible(self, history):
        """SelfConvolution is compatible with :class:`CheckpointedHistory`
        objects that are compatible with the 
        optimizer being self-convolved.
        
        """
        return (isinstance(history, CheckpointedHistory) and 
                self.opt.compatible(history.history))
        
    def needsScores(self):
        """SelfConvolution needs scores only if its optimizer does.
        
        """
        return self.opt.needsScores()
        
    def update(self, history, fitness):
        super(SelfConvolution, self).update(history, fitness)
        self.opt.update(history.history, fitness)
    
    def __call__(self):
        return self.batch(self.config.populationSize)    
                
    def batch(self, popSize):
        if self.checkpoint:
           self.history.checkpoint()
        from pyec.distribution.bayes.net import BayesNet
        
        times = self.times
        pop = None
        if self.history.empty():
            if self.config.initial is None:
                pop = [self.config.space.random() for i in xrange(popSize)]
            elif hasattr(self.config.initial, 'batch'):
                pop = self.config.initial.batch(popSize)
            else:
                pop = [self.config.initial() for i in xrange(popSize)]
            times -= 1
            
        fitness = self.opt.needsScores() and self.fitness or None
        for i in xrange(times):
            #self.config.stats.start("self-convolve.loop")
            #self.config.stats.start("history.update")
            if pop is not None:
                self.history.update(pop, fitness,
                                    self.opt.config.space, self.opt)
            #self.config.stats.stop("history.update")
            #self.config.stats.start("opt.update")
            self.opt.update(self.history.history, self.fitness)
            #self.config.stats.stop("opt.update")
            #self.config.stats.start("opt.call")
            pop = self.opt()
            #self.config.stats.stop("opt.call")
            gc.collect()
            #self.config.stats.stop("self-convolve.loop")
           
        if self.checkpoint:
            self.history.rollback()
        
        return pop
        