from pyec.distribution.ec.evoanneal import Neuroannealing
from pyec.distribution.nn.genotypes import HYPERBOLIC
from pyec.distribution.nn.space import LayeredRnnSpace

use_gpu = True
try:
    if not use_gpu:
        raise Exception("Skip GPU")
    import theano
    from pyec.util.net_gpu_benchmarks import concentric_spirals
except:
    use_gpu = False
    from pyec.util.net_benchmarks import concentric_spirals

import numpy as np
import sys

if len(sys.argv) <= 1:
    import subprocess
    import tempfile
    
    with tempfile.NamedTemporaryFile(delete=False) as f:
        fname = f.name
        pass
    
    for i in xrange(200):
        cmd = ['python',__file__,str(i),fname]
        subprocess.check_call(cmd)
    
    with open(fname) as f:
        results = [int(x) for x in f.readlines()]
        
    steps = 50 * np.array([x for x in results if x >= 0])
    incomplete = len([x for x in results if x < 0])    
    
    avg = np.average(steps)
    sd = np.sqrt(np.average((steps - avg)**2))
    print "AVERAGE EVALS: ", avg
    print "SD: ", sd
    print "success prob: ", ((200. - incomplete) / 200.0)
    
else:
    i = int(sys.argv[1])
    fname = sys.argv[2]
    
    class FitnessThreshold(Exception):
        pass

    class StepObserver(object):
        def __init__(self):
            self.steps = []
    
        def report(self, opt, pop):
            if opt.history.maximal()[1] >= 1.0:
                self.steps.append(opt.history.updates)
                print "Complete at ", self.steps[-1]
                raise FitnessThreshold(self.steps[-1])

    observer = StepObserver()
    try:
        p = ((Neuroannealing)
             (space=LayeredRnnSpace([2], [1], bias=False, activator=HYPERBOLIC,
                                    scale=1.0, gpu=use_gpu),
              netVerifyLinks=True,
              printEvery=1,
              observer=observer,
              populationSize=50,
              minimize=False,
              learningRate=1.0,
              sd=.25)
             << 1001)[None, concentric_spirals]()
    except FitnessThreshold:
        print i, "Complete at ", observer.steps[-1]
        with open(fname, "ab") as f:
            f.write(str(observer.steps[-1]) + "\r\n")
    else:
        print i, "did not complete"
        with open(fname, "ab") as f:
            f.write("{0}\r\n".format(opt.history.maximal()))
