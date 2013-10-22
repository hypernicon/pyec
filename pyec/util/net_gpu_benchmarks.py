"""Use theano to run net benchmarks on the GPU.

"""
import numpy as np

from .net_benchmark_util import load_concentric

concentric, concentricCorr = load_concentric()

def concentric_spirals(net):
    total = 0.0
    try:
        size = float(len(concentric))
        for pt,corr in zip(concentric, concentricCorr):
            outputs = net([pt.astype(np.float32)], times=25)
            netCorr = outputs[0][0] >= 0
            total += (netCorr == corr) / size
        print "TOTAL: ", total
        return total
    except:
        #import traceback
        #traceback.print_exc()
        return -1.0
