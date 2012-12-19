"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
from pyec.config import Config
from pyec.space import Euclidean, Hyperrectangle, Binary, BinaryRectangle
from pyec.util.partitions import *
from pyec.util.RunStats import RunStats
from pyec.util.TernaryString import TernaryString
import unittest

class TestPartitions(unittest.TestCase):
    def test_segment(self):
        space = Euclidean(dim=2)
        segment = Segment(Config(space=space))
        self.assertTrue(segment.config.space is space)
        self.assertEqual(segment.points, [])
        self.assertTrue(isinstance(segment.partitionTree, Partition))
        self.assertTrue(isinstance(segment.scoreTree, ScoreTree))
        segment.clearSegment()

    def test_partition_euclidean(self):
        stats = RunStats()
        segment = Segment(Config(
            space=Euclidean(dim=2),
            separator=VectorSeparationAlgorithm,
            stats=stats,
            taylorDepth=10,
            learningRate=1.0,
            taylorCenter=1.0,
            minimize=False,
            pressure=0.025
        ))
        points = [
            Point(segment=segment, point=np.array([0.0,0.0]), score=2.0),
            Point(segment=segment, point=np.array([1.0,0.5]), score=1.0),
            Point(segment=segment, point=np.array([2.0,1.0]), score=3.0),
            Point(segment=segment, point=np.array([0.25,1.0]), score=1.2),
        ]
        Point.bulkSave(segment, points, stats)
        
        # check the structure of the partition tree
        tree = segment.partitionTree
        current = tree.root
        self.assertEqual(current.area, 1.0)
        self.assertTrue(current.bounds is segment.config.space)
        self.assertTrue(current.parent is None)
        self.assertTrue(current.point is None)
        self.assertTrue(tree.segment is segment)
        self.assertTrue(current.upper is None)
        self.assertTrue(current.lower is None)
        self.assertEqual(len(current.children), 2)
        
        left, right = current.children
        self.assertTrue(left.parent is current)
        self.assertTrue(left.point is None)
        self.assertTrue(isinstance(left.bounds, Hyperrectangle))
        self.assertTrue((left.bounds.lower == np.array([.5, -np.inf])).all())
        self.assertTrue((left.bounds.upper == np.inf).all())
        self.assertTrue(np.abs(left.area - .3085) < .001)
        self.assertEqual(len(left.children), 2)
        self.assertEqual(left.index, 0)
        
        self.assertTrue(right.parent is current)
        self.assertTrue(right.point is None)
        self.assertTrue(isinstance(right.bounds, Hyperrectangle))
        self.assertTrue((right.bounds.lower == -np.inf).all())
        self.assertTrue((right.bounds.upper == np.array([.5, np.inf])).all())
        self.assertTrue(np.abs(right.area - .6915) < .001)
        self.assertEqual(len(right.children), 2)
        self.assertEqual(right.index, 0)
        
        lleft, lright = left.children
        self.assertTrue(lleft.parent is left)
        self.assertTrue(lleft.point is points[2])
        self.assertTrue(lleft.point.partition_node is lleft)
        self.assertTrue(isinstance(lleft.bounds, Hyperrectangle))
        self.assertTrue((lleft.bounds.lower == np.array([1.5, -np.inf])).all())
        self.assertTrue((lleft.bounds.upper == np.inf).all())
        self.assertTrue(np.abs(lleft.area - .0668) < .001)
        self.assertEqual(len(lleft.children), 0)
        self.assertEqual(lleft.index, 0)
        
        self.assertTrue(lright.parent is left)
        self.assertTrue(lright.point is points[1])
        self.assertTrue(lright.point.partition_node is lright)
        self.assertTrue(isinstance(lright.bounds, Hyperrectangle))
        self.assertTrue((lright.bounds.lower == np.array([.5, -np.inf])).all())
        self.assertTrue((lright.bounds.upper == np.array([1.5,np.inf])).all())
        self.assertTrue(np.abs(lright.area - .2417) < .001)
        self.assertEqual(len(lright.children), 0)
        self.assertEqual(lright.index, 0)
        
        rleft, rright = right.children
        self.assertTrue(rleft.parent is right)
        self.assertTrue(rleft.point is points[3])
        self.assertTrue(rleft.point.partition_node is rleft)
        self.assertTrue(isinstance(rleft.bounds, Hyperrectangle))
        self.assertTrue((rleft.bounds.lower == np.array([-np.inf,0.5])).all())
        self.assertTrue((rleft.bounds.upper == np.array([0.5,np.inf])).all())
        self.assertTrue(np.abs(rleft.area - .2133) < .001)
        self.assertEqual(len(rleft.children), 0)
        self.assertEqual(rleft.index, 1)
        
        self.assertTrue(rright.parent is right)
        self.assertTrue(rright.point is points[0])
        self.assertTrue(rright.point.partition_node is rright)
        self.assertTrue(isinstance(rright.bounds, Hyperrectangle))
        self.assertTrue((rright.bounds.lower == -np.inf).all())
        self.assertTrue((rright.bounds.upper == 0.5).all())
        self.assertTrue(np.abs(rright.area - .4782) < .001)
        self.assertEqual(len(rright.children), 0)
        self.assertEqual(rright.index, 1)
        
        #check the structure of the area tree
        tree = segment.partitionTree.areaTree
        current = tree.root
        self.assertTrue(current.parent is None)
        self.assertTrue(current.val is None)
        self.assertTrue(np.abs(current.low - 0.0) < 0.001)
        self.assertTrue(np.abs(current.high - 1.0) < 0.001)
        self.assertFalse(current.left)
        self.assertEqual(len(current.children), 2)
        
        left, right = current.children
        self.assertTrue(left.parent is current)
        self.assertTrue(np.abs(left.val.area - 0.4781) < 0.001)
        self.assertTrue(np.abs(left.low - 0.3085) < 0.001)
        self.assertTrue(np.abs(left.high - 1.0) < 0.001)
        self.assertTrue(left.left)
        self.assertEqual(len(left.children), 0)
        
        self.assertTrue(right.parent is current)
        self.assertTrue(right.val is None)
        self.assertTrue(np.abs(right.low - 0.0) < 0.001)
        self.assertTrue(np.abs(right.high - 0.3085) < 0.001)
        self.assertFalse(right.left)
        self.assertEqual(len(right.children), 2)
        
        lleft, lright = right.children
        self.assertTrue(lleft.parent is right)
        self.assertTrue(lleft.val is None)
        self.assertTrue(np.abs(lleft.low - 0.0668) < 0.001)
        self.assertTrue(np.abs(lleft.high - 0.3085) < 0.001)
        self.assertTrue(lleft.left)
        self.assertEqual(len(lleft.children), 2)
        
        self.assertTrue(lright.parent is right)
        self.assertTrue(np.abs(lright.val.area - 0.0668) < 0.001)
        self.assertTrue(np.abs(lright.low - 0.0) < 0.001)
        self.assertTrue(np.abs(lright.high - 0.0668) < 0.001)
        self.assertFalse(lright.left)
        self.assertEqual(len(lright.children), 0)
        
        rleft, rright = lleft.children
        self.assertTrue(rleft.parent is lleft)
        self.assertTrue(np.abs(rleft.val.area - 0.2417) < 0.001)
        self.assertTrue(np.abs(rleft.low - 0.2133) < 0.001)
        self.assertTrue(np.abs(rleft.high - 0.3085) < 0.001)
        self.assertTrue(rleft.left)
        self.assertEqual(len(rleft.children), 0)
        
        self.assertTrue(rright.parent is lleft)
        self.assertTrue(np.abs(rright.val.area - 0.2133) < 0.001)
        self.assertTrue(np.abs(rright.low - 0.0668) < 0.001)
        self.assertTrue(np.abs(rright.high - 0.2133) < 0.001)
        self.assertFalse(rright.left)
        self.assertEqual(len(rright.children), 0)
        
        #check the structure of the score tree
        tree = segment.scoreTree
        current = tree.root
        self.assertTrue(current.parent is None)
        self.assertTrue(current.segment is segment)
        self.assertTrue(current.point is None)
        self.assertTrue(np.abs(current.area - 1.0) < 0.001)
        self.assertEqual(current.min_score, 1.0)
        self.assertEqual(current.max_score, 3.0)
        self.assertEqual(current.child_count, 7)
        self.assertEqual(current.height, 2)
        self.assertEqual(current.balance, 0)
        self.assertFalse(current.left)
        self.assertEqual(len(current.children), 2)
        
        left, right = current.children
        self.assertTrue(left.parent is current)
        self.assertTrue(left.segment is segment)
        self.assertTrue(left.point is None)
        self.assertTrue(np.abs(left.area - .5449) < .001)
        self.assertEqual(left.min_score, 2.0)
        self.assertEqual(left.max_score, 3.0)
        self.assertEqual(left.child_count, 3)
        self.assertEqual(left.height, 1)
        self.assertEqual(left.balance, 0)
        self.assertTrue(left.left)
        self.assertEqual(len(left.children), 2)
        
        self.assertTrue(right.parent is current)
        self.assertTrue(right.segment is segment)
        self.assertTrue(right.point is None)
        self.assertTrue(np.abs(right.area - .4551) < .001)
        self.assertEqual(right.min_score, 1.0)
        self.assertEqual(right.max_score, 1.2)
        self.assertEqual(right.child_count, 3)
        self.assertEqual(right.height, 1)
        self.assertEqual(right.balance, 0)
        self.assertFalse(right.left)
        self.assertEqual(len(right.children), 2)
        
        lleft, lright = left.children
        self.assertTrue(lleft.parent is left)
        self.assertTrue(lleft.segment is segment)
        self.assertTrue(lleft.point is points[2])
        self.assertTrue(lleft.point.score_node is lleft)
        self.assertTrue(np.abs(lleft.area - .0668) < .001)
        self.assertEqual(lleft.min_score, 3.0)
        self.assertEqual(lleft.max_score, 3.0)
        self.assertEqual(lleft.child_count, 1)
        self.assertEqual(lleft.height, 0)
        self.assertEqual(lleft.balance, 0)
        self.assertTrue(lleft.left)
        self.assertEqual(len(lleft.children), 0)
        
        self.assertTrue(lright.parent is left)
        self.assertTrue(lright.segment is segment)
        self.assertTrue(lright.point is points[0])
        self.assertTrue(lright.point.score_node is lright)
        self.assertTrue(np.abs(lright.area - .4781) < .001)
        self.assertEqual(lright.min_score, 2.0)
        self.assertEqual(lright.max_score, 2.0)
        self.assertEqual(lright.child_count, 1)
        self.assertEqual(lright.height, 0)
        self.assertEqual(lright.balance, 0)
        self.assertFalse(lright.left)
        self.assertEqual(len(lright.children), 0)
        
        rleft, rright = right.children
        self.assertTrue(rleft.parent is right)
        self.assertTrue(rleft.segment is segment)
        self.assertTrue(rleft.point is points[3])
        self.assertTrue(rleft.point.score_node is rleft)
        self.assertTrue(np.abs(rleft.area - .2133) < .001)
        self.assertEqual(rleft.min_score, 1.2)
        self.assertEqual(rleft.max_score, 1.2)
        self.assertEqual(rleft.child_count, 1)
        self.assertEqual(rleft.height, 0)
        self.assertEqual(rleft.balance, 0)
        self.assertTrue(rleft.left)
        self.assertEqual(len(rleft.children), 0)
        
        self.assertTrue(rright.parent is right)
        self.assertTrue(rright.segment is segment)
        self.assertTrue(rright.point is points[1])
        self.assertTrue(rright.point.score_node is rright)
        self.assertTrue(np.abs(rright.area - .2417) < .001)
        self.assertEqual(rright.min_score, 1.0)
        self.assertEqual(rright.max_score, 1.0)
        self.assertEqual(rright.child_count, 1)
        self.assertEqual(rright.height, 0)
        self.assertEqual(rright.balance, 0)
        self.assertFalse(rright.left)
        self.assertEqual(len(rright.children), 0)
        
        # check sampleTournament
        probs = zeros(4)
        for i in xrange(25000):
            point, node = Point.sampleTournament(segment, 1.0, segment.config)
            probs += np.array([int(point is p)/25000. for p in points])
        ranks = np.array([1., 2., 0., 3.])
        scores = (1.0 - segment.config.pressure) ** ranks
        areas = np.array([p.partition_node.area for p in points])
        correct = scores * areas / ((scores * areas).sum())
        self.assertTrue(np.abs(correct - probs).max() < 0.015)
        
        probs = zeros(4)
        for i in xrange(25000):
            point, node = Point.sampleTournament(segment, 10.0, segment.config)
            probs += np.array([int(point is p)/25000. for p in points])
        scores = (1.0 - segment.config.pressure) ** (0.10 * ranks)
        correct = scores * areas / ((scores * areas).sum())
        self.assertTrue(np.abs(correct - probs).max() < 0.015)
        
        # check sampleProportional
        segment.scoreTree.resetTaylor(segment, 1.0, segment.config)
        probs = zeros(4)
        for i in xrange(25000):
            point, node = Point.sampleProportional(segment, 1./1.05, segment.config)
            probs += np.array([int(point is p)/25000. for p in points])
        scores = np.exp(np.array([segment.config.learningRate *
                                  p.score*1.05 for p in points]))
        scores /= scores.sum()
        correct = scores * areas / ((scores * areas).sum())
        self.assertTrue(np.abs(correct - probs).max() < 0.01)
        
        bottom = .5 * floor(2*(3.25)*segment.config.learningRate)
        segment.scoreTree.resetTaylor(segment, 1./bottom, segment.config)
        segment.config.taylorCenter = 1./bottom
        probs = zeros(4)
        for i in xrange(25000):
            point, node = Point.sampleProportional(segment, 1./3.25, segment.config)
            probs += np.array([int(point is p)/25000. for p in points])
        scores = np.exp(np.array([segment.config.learningRate *
                                  p.score*3.25 for p in points]))
        correct = scores * areas
        correct /= correct.sum()
        self.assertTrue(np.abs(correct - probs).max() < 0.01)

    def test_partition_longest(self):
        stats = RunStats()
        segment = Segment(Config(
            space=Euclidean(dim=2),
            separator=LongestSideVectorSeparationAlgorithm,
            stats=stats,
            taylorDepth=10,
            learningRate=1.0,
            taylorCenter=1.0,
            minimize=False,
            pressure=0.025
        ))
        points = [
            Point(segment=segment, point=np.array([0.0,0.0]), score=2.0),
            Point(segment=segment, point=np.array([1.0,0.5]), score=1.0),
            Point(segment=segment, point=np.array([2.0,1.0]), score=3.0),
            Point(segment=segment, point=np.array([0.25,1.0]), score=1.2),
        ]
        Point.bulkSave(segment, points, stats)
        
        # check the structure of the partition tree
        tree = segment.partitionTree
        current = tree.root
        self.assertEqual(current.area, 1.0)
        self.assertTrue(current.bounds is segment.config.space)
        self.assertTrue(current.parent is None)
        self.assertTrue(current.point is None)
        self.assertTrue(tree.segment is segment)
        self.assertTrue(current.upper is None)
        self.assertTrue(current.lower is None)
        self.assertEqual(len(current.children), 2)
        
        left, right = current.children
        self.assertTrue(left.parent is current)
        self.assertTrue(left.point is None)
        self.assertTrue(isinstance(left.bounds, Hyperrectangle))
        self.assertTrue((left.bounds.lower == np.array([.5, -np.inf])).all())
        self.assertTrue((left.bounds.upper == np.inf).all())
        self.assertTrue(np.abs(left.area - .3085) < .001)
        self.assertEqual(len(left.children), 2)
        self.assertEqual(left.index, 0)
        
        self.assertTrue(right.parent is current)
        self.assertTrue(right.point is None)
        self.assertTrue(isinstance(right.bounds, Hyperrectangle))
        self.assertTrue((right.bounds.lower == -np.inf).all())
        self.assertTrue((right.bounds.upper == np.array([.5, np.inf])).all())
        self.assertTrue(np.abs(right.area - .6915) < .001)
        self.assertEqual(len(right.children), 2)
        self.assertEqual(right.index, 0)
        
        lleft, lright = left.children
        self.assertTrue(lleft.parent is left)
        self.assertTrue(lleft.point is points[2])
        self.assertTrue(lleft.point.partition_node is lleft)
        self.assertTrue(isinstance(lleft.bounds, Hyperrectangle))
        self.assertTrue((lleft.bounds.lower == np.array([.5, .75])).all())
        self.assertTrue((lleft.bounds.upper == np.inf).all())
        self.assertTrue(np.abs(lleft.area - .0699) < .001)
        self.assertEqual(len(lleft.children), 0)
        self.assertEqual(lleft.index, 1)
        
        self.assertTrue(lright.parent is left)
        self.assertTrue(lright.point is points[1])
        self.assertTrue(lright.point.partition_node is lright)
        self.assertTrue(isinstance(lright.bounds, Hyperrectangle))
        self.assertTrue((lright.bounds.lower == np.array([.5, -np.inf])).all())
        self.assertTrue((lright.bounds.upper == np.array([np.inf,.75])).all())
        self.assertTrue(np.abs(lright.area - .2386) < .001)
        self.assertEqual(len(lright.children), 0)
        self.assertEqual(lright.index, 1)
        
        rleft, rright = right.children
        self.assertTrue(rleft.parent is right)
        self.assertTrue(rleft.point is points[3])
        self.assertTrue(rleft.point.partition_node is rleft)
        self.assertTrue(isinstance(rleft.bounds, Hyperrectangle))
        self.assertTrue((rleft.bounds.lower == np.array([-np.inf,0.5])).all())
        self.assertTrue((rleft.bounds.upper == np.array([0.5,np.inf])).all())
        self.assertTrue(np.abs(rleft.area - .2133) < .001)
        self.assertEqual(len(rleft.children), 0)
        self.assertEqual(rleft.index, 1)
        
        self.assertTrue(rright.parent is right)
        self.assertTrue(rright.point is points[0])
        self.assertTrue(rright.point.partition_node is rright)
        self.assertTrue(isinstance(rright.bounds, Hyperrectangle))
        self.assertTrue((rright.bounds.lower == -np.inf).all())
        self.assertTrue((rright.bounds.upper == 0.5).all())
        self.assertTrue(np.abs(rright.area - .4781) < .001)
        self.assertEqual(len(rright.children), 0)
        self.assertEqual(rright.index, 1)
        
    def test_partition_binary(self):
        stats = RunStats()
        segment = Segment(Config(
            space=Binary(dim=100),
            separator=BinarySeparationAlgorithm,
            stats=stats,
            taylorDepth=10,
            learningRate=1.0,
            taylorCenter=1.0,
            minimize=False,
            pressure=0.025
        ))
        points = [
            Point(segment=segment, point=TernaryString(0L,-1L,100), score=2.0),
            Point(segment=segment, point=TernaryString(1L,-1L,100), score=1.0),
            Point(segment=segment, point=TernaryString(2L,-1L,100), score=3.0),
            Point(segment=segment, point=TernaryString(3L,-1L,100), score=1.2),
        ]
        Point.bulkSave(segment, points, stats)
        
        # check the structure of the partition tree
        tree = segment.partitionTree
        tree.printTree(segment)
        current = tree.root
        self.assertEqual(current.area, 1.0)
        self.assertTrue(current.bounds is segment.config.space)
        self.assertTrue(current.parent is None)
        self.assertTrue(current.point is None)
        self.assertTrue(tree.segment is segment)
        self.assertTrue(current.upper is None)
        self.assertTrue(current.lower is None)
        self.assertEqual(len(current.children), 2)
        
        left, right = current.children
        self.assertTrue(left.parent is current)
        self.assertTrue(left.point is None)
        self.assertTrue(isinstance(left.bounds, BinaryRectangle))
        self.assertTrue(left.bounds.spec == TernaryString(1L,1L,100))
        self.assertTrue(np.abs(left.area - .5) < .001)
        self.assertEqual(len(left.children), 2)
        self.assertEqual(left.index, 0)
        
        self.assertTrue(right.parent is current)
        self.assertTrue(right.point is None)
        self.assertTrue(isinstance(right.bounds, BinaryRectangle))
        self.assertTrue(right.bounds.spec == TernaryString(0L,1L,100))
        self.assertTrue(np.abs(right.area - .5) < .001)
        self.assertEqual(len(right.children), 2)
        self.assertEqual(right.index, 0)
        
        lleft, lright = left.children
        self.assertTrue(lleft.parent is left)
        self.assertTrue(lleft.point is points[3])
        self.assertTrue(lleft.point.partition_node is lleft)
        self.assertTrue(isinstance(lleft.bounds, BinaryRectangle))
        print lleft.bounds.spec.base
        print lleft.bounds.spec.known
        print lleft.bounds.spec.length
        self.assertTrue(lleft.bounds.spec == TernaryString(3L,3L,100))
        self.assertTrue(np.abs(lleft.area - .25) < .001)
        self.assertEqual(len(lleft.children), 0)
        self.assertEqual(lleft.index, 1)
        
        self.assertTrue(lright.parent is left)
        self.assertTrue(lright.point is points[1])
        self.assertTrue(lright.point.partition_node is lright)
        self.assertTrue(isinstance(lright.bounds, BinaryRectangle))
        self.assertTrue(lright.bounds.spec == TernaryString(1L,3L,100))
        self.assertTrue(np.abs(lright.area - .25) < .001)
        self.assertEqual(len(lright.children), 0)
        self.assertEqual(lright.index, 1)
        
        rleft, rright = right.children
        self.assertTrue(rleft.parent is right)
        self.assertTrue(rleft.point is points[2])
        self.assertTrue(rleft.point.partition_node is rleft)
        self.assertTrue(isinstance(rleft.bounds, BinaryRectangle))
        self.assertTrue(rleft.bounds.spec == TernaryString(2L,3L,100))
        self.assertTrue(np.abs(rleft.area - .25) < .001)
        self.assertEqual(len(rleft.children), 0)
        self.assertEqual(rleft.index, 1)
        
        self.assertTrue(rright.parent is right)
        self.assertTrue(rright.point is points[0])
        self.assertTrue(rright.point.partition_node is rright)
        self.assertTrue(isinstance(rright.bounds, BinaryRectangle))
        self.assertTrue(rright.bounds.spec == TernaryString(0L,3L,100))
        self.assertTrue(np.abs(rright.area - .25) < .001)
        self.assertEqual(len(rright.children), 0)
        self.assertEqual(rright.index, 1)