"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from pyec.config import Config
from pyec.history import *
from pyec.space import *
from pyec.distribution.convex import Convex
from pyec.distribution.convolution import Convolution, SelfConvolution
from pyec.distribution.gss import GeneratingSetSearch as DE
from pyec.distribution.gss import GeneratingSetSearchHistory
from pyec.distribution.truncation import TrajectoryTruncation

space = Euclidean(dim=5)

def run(opt):
    """A small test run to verify behavior"""
    p = None
    f = lambda x: (x**2).sum()
    s = opt.config.space
    t = opt.config.history(opt.config)
    
    for i in xrange(10):
        p = opt[t.update(p,f,s), f]()


def test_cls_getitem():
    DE7 = DE[Config(populationSize=7)]
    assert DE7.config.populationSize == 7
    de7 = DE7(space=space)
    assert de7.config.populationSize == 1
    de5 = DE7(populationSize=5, space=space)
    assert de5.config.populationSize == 1
    run(de7)
    run(de5)
    assert isinstance(de7.history, GeneratingSetSearchHistory)
    assert isinstance(de5.history, GeneratingSetSearchHistory)

def test_cls_convolve():
    DEDE = DE << DE
    assert issubclass(DEDE, Convolution)
    dede = DEDE(populationSize=11, space=space)
    assert dede.config.populationSize == 1
    assert len(dede.subs) == 2
    de = dede.subs[0]
    assert de.config.populationSize == 1
    assert isinstance(de, DE)
    assert isinstance(dede, DEDE)
    run(dede)
    assert isinstance(dede.history, CheckpointedMultipleHistory)
    assert isinstance(de.history, GeneratingSetSearchHistory)
    print "evals: ", dede.history.evals, de.history.evals
    assert dede.history.evals == de.history.evals

def test_cls_self_convolve():
    DEx10 = DE << 10
    assert issubclass(DEx10, SelfConvolution)
    dex10 = DEx10(populationSize=13, space=space)
    assert dex10.config.populationSize == 1
    assert dex10.times == 10
    de = dex10.opt
    assert de.config.populationSize == 1
    assert isinstance(de, DE)
    assert isinstance(dex10, DEx10)
    run(dex10)
    assert isinstance(dex10.history, CheckpointedHistory)
    assert isinstance(de.history, GeneratingSetSearchHistory)
    assert dex10.history.evals == de.history.evals

def test_cls_scalar_multiply():
    DE2 = 2 * DE
    DE3_4 = DE2 * 1.7
    assert issubclass(DE2, DE)
    assert issubclass(DE3_4, DE)
    assert issubclass(DE3_4, DE2)
    assert DE.weight == 1.0
    assert DE2.weight == 1.0 * 2
    assert DE3_4.weight == 1.0 * 2 * 1.7
    de = DE(populationSize=5, space=space)
    de2 = DE2(populationSize=7, space=space)
    de3_4 = DE3_4(populationSize=11, space=space)
    assert de.weight == 1.0
    assert de2.weight == 1.0 * 2
    assert de3_4.weight == 1.0 * 2 * 1.7
    assert de.config.populationSize == 1
    assert de2.config.populationSize == 1
    assert de3_4.config.populationSize == 1
    run(de)
    run(de2)
    run(de3_4)
    assert isinstance(de2.history, GeneratingSetSearchHistory)

def test_cls_convex():
    DEC = .1 * DE + .6 * DE
    assert issubclass(DEC, Convex)
    de = DEC(populationSize=13, space=space)
    assert len(de.subs) == 2
    assert de.subs[0].weight == .1
    assert de.subs[1].weight == .6
    assert isinstance(de.subs[0], .1*DE)
    assert isinstance(de.subs[1], .6*DE)
    run(de)
    assert isinstance(de.history, MultipleHistory)
    assert isinstance(de.subs[0].history, GeneratingSetSearchHistory)
    assert isinstance(de.subs[1].history, GeneratingSetSearchHistory)
    assert de.subs[0].history.evals == de.history.evals

def test_cls_truncate():
    DET = DE >> 5
    assert issubclass(DET, TrajectoryTruncation)
    det = DET(populationSize=17, space=space)
    assert isinstance(det.opt, DE)
    assert det.delay == 5
    run(det)
    assert isinstance(det.history, DelayedHistory)
    assert isinstance(det.opt.history, GeneratingSetSearchHistory)
    assert det.history.delay == 5
    assert det.history.evals == det.opt.history.evals

def test_cls_truncate_convolve():
    DET = DE >> DE
    assert issubclass(DET, Convolution)
    det = DET(populationSize=19, space=space)
    assert len(det.subs) == 2
    assert isinstance(det.subs[0], DE)
    assert isinstance(det.subs[1], TrajectoryTruncation)
    assert det.subs[1].delay == 1
    assert isinstance(det.subs[1].opt, DE)
    run(det)
    assert isinstance(det.history, CheckpointedMultipleHistory)
    assert isinstance(det.subs[0].history, GeneratingSetSearchHistory)
    assert isinstance(det.subs[1].history, DelayedHistory)
    assert isinstance(det.subs[1].opt.history, GeneratingSetSearchHistory)
    assert det.history.evals == det.subs[0].history.evals
    assert det.subs[1].history.evals == det.history.evals - space.dim - 1

def test_obj_convolve():
    DE2 = DE[Config(populationSize=11, space=space)]
    dede = DE2() << DE2()
    assert isinstance(dede, Convolution)
    assert dede.config.populationSize == 1
    assert len(dede.subs) == 2
    de = dede.subs[0]
    assert de.config.populationSize == 1
    assert isinstance(de, DE2)
    run(dede)
    assert isinstance(dede.history, CheckpointedMultipleHistory)
    assert isinstance(de.history, GeneratingSetSearchHistory)
    assert dede.history.evals == de.history.evals


def test_obj_self_convolve():
    dex10 = DE(populationSize=13, space=space) << 10
    assert isinstance(dex10, SelfConvolution)
    assert dex10.config.populationSize == 1
    assert dex10.times == 10
    de = dex10.opt
    assert de.config.populationSize == 1
    assert isinstance(de, DE)
    run(dex10)
    assert isinstance(dex10.history, CheckpointedHistory)
    assert isinstance(de.history, GeneratingSetSearchHistory)
    assert dex10.history.evals == de.history.evals

def test_obj_convex():
    DE2 = DE[Config(populationSize=13, space=space)]
    de = .1 * DE2() + .6 * DE2()
    assert isinstance(de, Convex)
    assert len(de.subs) == 2
    assert de.subs[0].weight == .1
    assert de.subs[1].weight == .6
    assert isinstance(de.subs[0], DE2)
    assert isinstance(de.subs[1], DE2)
    run(de)
    assert isinstance(de.history, MultipleHistory)
    assert isinstance(de.subs[0].history, GeneratingSetSearchHistory)
    assert isinstance(de.subs[1].history, GeneratingSetSearchHistory)
    assert de.subs[0].history.evals == de.history.evals
    
def test_obj_scalar_multiply():
    DE2 = DE[Config(populationSize=5, space=space)]
    de = DE2()
    de2 = 2 * DE2(populationSize=7)
    de3_4 = de2 * 1.7
    assert isinstance(de3_4, DE2)
    assert de2 is not de
    assert de3_4 is not de2
    assert de.weight == 1.0
    assert de2.weight == 1.0 * 2
    assert de3_4.weight == 1.0 * 2 * 1.7
    assert de.config.populationSize == 1
    assert de2.config.populationSize == 1
    assert de3_4.config.populationSize == 1
    run(de)
    run(de2)
    run(de3_4)
    assert isinstance(de2.history, GeneratingSetSearchHistory)

def test_obj_truncate():
    det = DE(populationSize=17, space=space) >> 5
    assert isinstance(det, TrajectoryTruncation)
    assert isinstance(det.opt, DE)
    assert det.delay == 5
    run(det)
    assert isinstance(det.history, DelayedHistory)
    assert isinstance(det.opt.history, GeneratingSetSearchHistory)
    assert det.history.delay == 5
    assert det.history.evals == det.opt.history.evals

def test_obj_truncate_convolve():
    DE2 = DE[Config(populationSize=19, space=space)]
    det = DE2() >> DE2()
    assert isinstance(det, Convolution)
    assert isinstance(det.subs[0], DE)
    assert isinstance(det.subs[1], TrajectoryTruncation)
    assert det.subs[1].delay == 1
    assert isinstance(det.subs[1].opt, DE)
    run(det)
    assert isinstance(det.history, CheckpointedMultipleHistory)
    assert isinstance(det.subs[0].history, GeneratingSetSearchHistory)
    assert isinstance(det.subs[1].history, DelayedHistory)
    assert isinstance(det.subs[1].opt.history, GeneratingSetSearchHistory)
    assert det.history.evals == det.subs[0].history.evals
    assert det.subs[1].history.evals == det.history.evals - space.dim - 1