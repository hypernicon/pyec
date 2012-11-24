"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import pyec.optimize
from numpy import *

seterr(all="ignore")

def test_de_dim1():
   x,f = pyec.optimize.differential_evolution("sphere",dimension=1)
   print "de dim 1: ", x, f
   assert sqrt((x**2).sum()) < 1e-1
   assert abs(f) < 1e-5
   
   x,f = pyec.optimize.de("sphere",dimension=1)
   assert sqrt((x**2).sum()) < 1e-1
   assert abs(f) < 1e-5
   
   
def test_cmaes_dim1():
   x,f = pyec.optimize.cmaes("sphere",dimension=1)
   print "cmaes dim 1: ", x, f
   assert sqrt((x**2).sum()) < 1e-1
   assert abs(f) < 1e-5
   

def test_nm_dim1():
   x,f = pyec.optimize.nelder_mead("sphere",dimension=1, generations=10000)
   print "nm dim 1: ", x, f
   assert sqrt((x**2).sum()) < 1e-1
   assert abs(f) < 1e-5
   
   x,f = pyec.optimize.nm("sphere",dimension=1, generations=10000)
   assert sqrt((x**2).sum()) < 1e-1
   assert abs(f) < 1e-5
   
def test_gss_dim1():
   x,f = pyec.optimize.generating_set_search("sphere",dimension=1, generations=10000)
   print "gss dim 1: ", x, f
   assert sqrt((x**2).sum()) < 1e-1
   assert abs(f) < 1e-5
   
   x,f = pyec.optimize.gss("sphere",dimension=1, generations=10000)
   assert sqrt((x**2).sum()) < 1e-1
   assert abs(f) < 1e-5
   
def test_pso_dim1():
   x,f = pyec.optimize.particle_swarm_optimization("sphere",dimension=1, generations=1000)
   print "pso dim 1: ", x, f
   assert sqrt((x**2).sum()) < 1e-1
   assert abs(f) < 1e-5
   
   x,f = pyec.optimize.pso("sphere",dimension=1, generations=1000)
   assert sqrt((x**2).sum()) < 1e-1
   assert abs(f) < 1e-5
   
def test_evoanneal_dim1():
   x,f = pyec.optimize.evolutionary_annealing("sphere",dimension=1)
   print "evoanneal dim 1: ", x, f
   assert sqrt((x**2).sum()) < 1e-1
   assert abs(f) < 1e-5
   
   x,f = pyec.optimize.evoanneal("sphere",dimension=1)
   assert sqrt((x**2).sum()) < 1e-1
   assert abs(f) < 1e-5   
   
def test_sa_dim1():
   x,f = pyec.optimize.simulated_annealing("sphere",dimension=1,schedule="log",learning_rate=10., generation=25000)
   print "sa dim 1: ", x, f
   assert sqrt((x**2).sum()) < 1
   assert abs(f) < 1
   
   x,f = pyec.optimize.sa("sphere",dimension=1,schedule="log",learning_rate=10., generation=25000)
   print "sa dim 1: ", x, f
   assert sqrt((x**2).sum()) < 1
   assert abs(f) < 1

def test_de_dim5():
   x,f = pyec.optimize.differential_evolution("sphere",dimension=5,CR=.5,F=.5,generations=1000)
   print "de dim 5: ", x, f
   assert sqrt((x**2).sum()) < 1e-1
   assert abs(f) < 1e-5
   
   x,f = pyec.optimize.de("sphere",dimension=5,CR=.5,F=.5,generations=1000)
   assert sqrt((x**2).sum()) < 1e-1
   assert abs(f) < 1e-5
   
   
def test_cmaes_dim5():
   x,f = pyec.optimize.cmaes("sphere",dimension=5)
   print "cmaes dim 5: ", x, f
   assert sqrt((x**2).sum()) < 1e-1
   assert abs(f) < 1e-5
   
def test_nm_dim5():
   x,f = pyec.optimize.nelder_mead("sphere",dimension=5)
   print "nm dim 5: ", x, f
   assert sqrt((x**2).sum()) < 1e-1
   assert abs(f) < 1e-2
   
   x,f = pyec.optimize.nm("sphere",dimension=5)
   assert sqrt((x**2).sum()) < 1e-1
   assert abs(f) < 1e-2
   
def test_gss_dim5():
   x,f = pyec.optimize.generating_set_search("sphere",dimension=5)
   print "gss dim 5: ", x, f
   assert sqrt((x**2).sum()) < 1e-1
   assert abs(f) < 1e-2
   
   x,f = pyec.optimize.gss("sphere",dimension=5)
   assert sqrt((x**2).sum()) < 1e-1
   assert abs(f) < 1e-2

def test_sa_dim5():
   x,f = pyec.optimize.simulated_annealing("sphere",dimension=5,schedule="linear",generations=25000, learning_rate=10.)
   print "sa dim 5: ", x, f
   assert sqrt((x**2).sum()) < 1e-1
   assert abs(f) < 1e-2
   
   x,f = pyec.optimize.sa("sphere",dimension=5,schedule="linear",generations=25000,learning_rate=10.)
   assert sqrt((x**2).sum()) < 1e-1
   assert abs(f) < 1e-2
   
def test_pso_dim5():
   x,f = pyec.optimize.particle_swarm_optimization("sphere",dimension=5,generations=10000, population=50)
   print "pso dim 5: ", x, f
   assert sqrt((x**2).sum()) < 5e-1
   assert abs(f) < 1e-1
   
   x,f = pyec.optimize.pso("sphere",dimension=5,generations=1000, population=50)
   assert sqrt((x**2).sum()) < 5e-1
   assert abs(f) < 1e-1
   
def test_evoanneal_dim5():
   x,f = pyec.optimize.evolutionary_annealing("sphere",dimension=5, learning_rate=1000., generations=100)
   print "evoanneal dim 5: ", x, f
   assert sqrt((x**2).sum()) < 1e-1
   assert abs(f) < 1e-5
   
   x,f = pyec.optimize.evoanneal("sphere",dimension=5, learning_rate=1000., generations=100)
   assert sqrt((x**2).sum()) < 1e-1
   assert abs(f) < 1e-5   

def test_de_dim10():
   x,f = pyec.optimize.differential_evolution("sphere",dimension=10,CR=.5,F=.5,generations=2500)
   print "de dim 10: ", x, f
   assert sqrt((x**2).sum()) < 1e-1
   assert abs(f) < 1e-5
   
   x,f = pyec.optimize.de("sphere",dimension=10,CR=.5,F=.5,generations=2500)
   assert sqrt((x**2).sum()) < 1e-1
   assert abs(f) < 1e-5
   
   
def test_cmaes_dim10():
   x,f = pyec.optimize.cmaes("sphere",dimension=10)
   assert sqrt((x**2).sum()) < 1e-1
   assert abs(f) < 1e-5
   print "cmaes dim 10: ", x, f

def test_nm_dim10():
   x,f = pyec.optimize.nelder_mead("sphere",dimension=10, generations=10000)
   assert sqrt((x**2).sum()) < 1e-1
   assert abs(f) < 1e-5
   print "nm dim 10: ", x, f
   
   x,f = pyec.optimize.nm("sphere",dimension=10, generations=10000)
   assert sqrt((x**2).sum()) < 1e-1
   assert abs(f) < 1e-5
   
def test_gss_dim10():
   x,f = pyec.optimize.generating_set_search("sphere",dimension=10, generations=10000)
   assert sqrt((x**2).sum()) < 1e-1
   assert abs(f) < 1e-5
   print "gss dim 10: ", x, f
   
   x,f = pyec.optimize.gss("sphere",dimension=10, generations=10000)
   assert sqrt((x**2).sum()) < 1e-1
   assert abs(f) < 1e-5

def test_sa_dim10():
   x,f = pyec.optimize.simulated_annealing("sphere",dimension=10, schedule="linear", learning_rate=100., generations=25000)
   print "sa dim 10: ", x, f
   assert sqrt((x**2).sum()) < 5e-1
   assert abs(f) < 5e-1
   
   x,f = pyec.optimize.sa("sphere",dimension=10, schedule="linear", learning_rate=100., generations=25000)
   assert sqrt((x**2).sum()) < 5e-1
   assert abs(f) < 5e-1
   
def test_pso_dim10():
   x,f = pyec.optimize.particle_swarm_optimization("sphere",dimension=10, generations=10000, population=50)
   print "pso dim 10: ", x, f
   assert sqrt((x**2).sum()) < 1.0
   assert abs(f) < 5e-1
   
   x,f = pyec.optimize.pso("sphere",dimension=10, generations=10000, population=50)
   assert sqrt((x**2).sum()) < 1.0
   assert abs(f) < 5e-1

def test_evoanneal_dim10():
   x,f = pyec.optimize.evolutionary_annealing("sphere",dimension=10, learning_rate=10000., generations=500)
   print "evoanneal dim 10: ", x, f
   assert sqrt((x**2).sum()) < 1e-1
   assert abs(f) < 1e-5
   
   x,f = pyec.optimize.evoanneal("sphere",dimension=10, learning_rate=10000., generations=500)
   assert sqrt((x**2).sum()) < 1e-1
   assert abs(f) < 1e-5     
