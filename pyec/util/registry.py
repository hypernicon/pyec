"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from pyec.util.importlib import import_module
from pyec.config import ConfigBuilder

import traceback

class RegistryCodeNotFound(Exception):
   pass
   
class BadRegistryEntry(Exception):
   pass

class Registry(object):
   registry = {}
   def __init__(self, parentCls):
      self.parentCls = parentCls
   
   def update(self, code, path):
      self.registry.update({code:path})

   def load(self, code, *extra):
      if self.registry.has_key(code):
         try:
            path = self.registry[code]
            pkg,cls = path.rsplit('.',1)
            mod = import_module(pkg)
            cls = getattr(mod, cls)
            return cls(*extra)
         except:
            traceback.print_exc()
            raise BadRegistryEntry, "Could not load " + str(path) + " associated to code " + str(code)
      elif self.registry.has_key('__dynamic__'):
         try:
            path = self.registry['__dynamic__']
            pkg,cls = path.rsplit('.',1)
            mod = import_module(pkg)
            cls = getattr(mod, cls)
            return cls(code, *extra)
         except:
            raise BadRegistryEntry, "Could not load " + str(path) + " associated to code " + str(code) 
      else:
         raise RegistryCodeNotFound, "Could not find code " + str(code)

BENCHMARKS = Registry(object)
BENCHMARKS.registry = {
   'zero':'pyec.util.benchmark.zero',
   'triangle':'pyec.util.benchmark.triangle',
   'sphere':'pyec.util.benchmark.sphere',
   'ellipsoid':'pyec.util.benchmark.ellipsoid',
   'rosenbrock':'pyec.util.benchmark.rosenbrock',
   'rastrigin':'pyec.util.benchmark.rastrigin',
   'miscaledRastrigin':'pyec.util.benchmark.miscaledRastrigin',
   'schwefel':'pyec.util.benchmark.schwefel',
   'salomon':'pyec.util.benchmark.salomon',
   'whitley':'pyec.util.benchmark.whitley',
   'ackley':'pyec.util.benchmark.ackley',
   'ackley2':'pyec.util.benchmark.ackley2',
   'langerman':'pyec.util.benchmark.langerman',
   'shekelsFoxholes':'pyec.util.benchmark.shekelsFoxholes',
   'shekel2':'pyec.util.benchmark.shekel2',
   'rana':'pyec.util.benchmark.rana',
   'griewank':'pyec.util.benchmark.griewank',
   'weierstrass':'pyec.util.benchmark.weierstrass',
   'brownian':'pyec.util.benchmark.brownian',
}     


__all__ = ['BENCHMARKS']
