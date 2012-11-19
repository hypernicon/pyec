"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from numpy import *
import copy
   



class Config(object):
   """
      A configuration object for an optimization algorithm. Each optimizer is created with a configuration object that is used to parameterize the optimizer instance. A configuration object may have arbitrary properties and methods. Default versions of methods used by several optimizers are provided.
   """
   def __get__(self, key):
      if not self.__dict__.has_key(key):
         return None
      return self.__dict__[key]

   def __set__(self, key, val):
      self.__dict__[key] = val

   def in_bounds(self, solution):
      """
         Check whether a solution falls inside the optimization constraints.
         
         Default implementation returns ``True``, implementing unconstrained optimization.
         
         :param solution: The python object being checked for constraints.
         :type solution: varied
         :returns: bool -- whether the solution is within the constraints.
      """
      return True

   def encode(self, solution):
      """
         Given a solution object, encode it as a unicode string, e.g. for storage in a database.
         
         Default implementation simply calls ``unicode``.
      
         :param solution: The name to use.
         :type solution: varied
         :returns:  unicode -- the encoded solution.
      """
      return unicode(solution)
      
   def convert(self, solution):
      """
         Given a string representation of a solution, convert it to a python object for computation.
         
         In the language of genetic algorithms, this method is used to decode a genome into a phenome.
         
         Default implementation simply parrots back the argument.
      
         :param solution: The name to use.
         :type solution: ``unicode`` or ``str``.
         :returns:  varied -- the converted solution.
      """
      return solution

      
class ConfigBuilder(object):
   """
      A builder to generate configuration objects with certain parameters.
      
      A builder creates a specific configuration object for a specific optimization method. Its ``__init__`` method is used to set the optimization class and the default parameters. The ``cfg`` property can then be modified to replace the defaults, and then ``configure`` can be called to generate an optimizer with the desired configuration. When an optimizer is instantiated, a copy of the default configuration is used, so the the builder can be reused.
      
      Several default training parameters are placed into the :class:`Config` object; view the source for details. 
      
      :param algcls: A class object or other generator that produces a :class:`PopulationDistribution` instance when called.
      :type algcls: class
      
      
      
   """
   
   def __init__(self, algcls):
      """
         Initialize the ``ConfigBuilder`` object.
         
         
         
      """
      self.cfg = Config()
      self.cfg.stopAt = 1e300
      self.cfg.scale = 0.5
      self.cfg.center = 0.5
      self.cfg.space_scale = 10.
      self.cfg.recording = False
      self.cfg.bounded = True
      self.cfg.segment = 'test'
      self.cfg.activeField = 'point'
      self.cfg.binaryPartition = False
      self.cfg.layered = False
      self.cfg.varInit = None
      self.cfg.sort = True
      self.dimension = 5
      self.cfg.fitness = ""
      self.algcls = algcls
   
   def postConfigure(self,cfg):
      """
         Called by `configure` to install any final properties into the
         :class:`Config` object after it has been copied. Properties that 
         are changed by `postConfigure` are not shared among different 
         optimizer instances. This method should be used to install 
         any objects that contain state that is specific to an optimizer.
         
         :param cfg: The copied :class:`Config` object.
         :type cfg: :class:`Config`
      """
      pass
   
   def configure(self, generations, populationSize, dimension=1, function=None):
      """
         Creates an optimizer instance and applies the built configuration
         to the optimizer.
         
         :param generations: The number of generations (samples) to take during optimization
         :type generations: int
         :param populationSize: The size of each population per generation (number of proposed solutions per sample)
         :type populationSize: int
         :param dimension: The dimension of the object, for vector optimization (binary or real); default `1`.
         :type dimension: int
         :param function: A function to be maximized; the fitness function or objective. This function will be placed in the :class:`Config` object so that the optimizer can access it as necessary.
         :type function: any callable object
         :returns: The configured optimizer instance, usually a :class:`PopulationDistribution` instance. 
      """
      cfg = copy.copy(self.cfg)
      cfg.generations = generations
      cfg.populationSize = populationSize
      cfg.dim = dimension
      self.dimension = dimension
      if function: cfg.function = function
      self.postConfigure(cfg)
      return self.algcls(cfg)
