"""Use theano to run net benchmarks on the GPU.

"""
import theano
import theano.tensor as T
from datetime import datetime
import numpy as np

from .net_benchmark_util import load_concentric

concentric, concentricCorr = load_concentric()
concentric = np.array([pt.astype(np.float32) for pt in concentric],
                      dtype=np.float32)

try:
    concentricOutput = T.fvector()
    concentricCorrGpu = theano.shared((2.0 * concentricCorr - 1.0).astype(np.float32))
    concentricCheck = T.mean(T.abs_((T.sgn(concentricOutput) - concentricCorrGpu))) / 2
    concentricAverage = theano.function([concentricOutput],concentricCheck)
except:
    import traceback
    traceback.print_exc()
    raise

def concentric_spirals(net):
    total = 0.0
    size = float(len(concentric))
    try:
        start = datetime.now()
        outputs = net([concentric], times=1024)
        print (datetime.now() - start).total_seconds(), "sec"
        start = datetime.now()
        ret = concentricAverage(outputs[0].reshape(len(concentric)))
        print (datetime.now() - start).total_seconds(), "sec result: ", ret
        return ret
    
        #start = datetime.now()
        #for pt,corr in zip(concentric, concentricCorr):
        #    outputs = net([pt.astype(np.float32)], times=1024)
        #    netCorr = outputs[0][0] >= 0
        #    total += (netCorr == corr) / size
        #print (datetime.now() - start).total_seconds(), "sec TOTAL: ", total
        #return total
        
    except:
        import traceback
        traceback.print_exc()
        return -1.0
