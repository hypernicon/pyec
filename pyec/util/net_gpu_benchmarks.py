"""Use theano to run net benchmarks on the GPU.

"""
import numpy as np

from .net_benchmark_util import load_concentric

concentric, concentricCorr = load_concentric()
concentric = np.array([pt.astype(np.float32) for pt in concentric],
                      dtype=np.float32)

def concentric_spirals(net):
    total = 0.0
    size = float(len(concentric))
    try:
        outputs = net([concentric], times=1024)
        ret = np.average((outputs[0] >= 0.0) == concentricCorr)
        return ret
    
        """
        for pt,corr in zip(concentric, concentricCorr):
            outputs = net([pt.astype(np.float32)], times=1024)
            netCorr = outputs[0][0] >= 0
            total += (netCorr == corr) / size
        return total
        """
    except:
        #import traceback
        #traceback.print_exc()
        return -1.0
