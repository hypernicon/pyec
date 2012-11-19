from numpy import *
from pyec.optimize import evoanneal, Hyperrectangle
from pyec.util.benchmark import shekel2


s = shekel2()
vals = []
for i in xrange(10):
    ret =  evoanneal(s, constraint=Hyperrectangle(center=5,scale=10), learning_rate=.05, population=25, generations=200, minimize=False)
    print i, ": ", ret
    vals.append(ret[1])

print average(vals)

