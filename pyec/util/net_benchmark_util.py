import os.path
import numpy as np

def load_concentric():
    ptFile = open(os.path.join(os.path.dirname(__file__),"concentric.points"))
    base = ptFile.read().split(" ")
    numPoints = int(base[0])
    dim = int(base[1])
    idx = 2
    concentric = np.zeros((numPoints, dim), dtype=float)
    concentricCorr = np.zeros(numPoints, dtype=bool)
    for i in xrange(numPoints):
        point = []
        for j in xrange(dim):
            concentric[i,j] = float(base[idx])
            idx += 1
        concentricCorr[i] = int(base[idx])
        idx += 1
    ptFile.close()
    return concentric, concentricCorr
