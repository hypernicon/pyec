"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from numpy import *
from scipy.special import erf
from pyec.space import Hyperrectangle, BinaryRectangle
from pyec.util.TernaryString import TernaryString

import traceback

import logging
logg = logging.getLogger(__file__)


class SeparationException(Exception):
   pass


class AreaException(Exception):
   pass


class Segment(object):
   segment = None

   def __init__(self, config):
      self.config = config
      
      self.points = []
      self.partitionTree = Partition(self,config)
      self.scoreTree = ScoreTree(self,config)
      
   def clearSegment(self):
      del self.points
      del self.partitionTree
      del self.scoreTree
      self.points = []
      self.partitionTree = Partition(self,self.config)
      self.scoreTree = ScoreTree(self,self.config)


class Point(object):
   
   def __init__(self, segment, point=None, statistics = None,
                score=0.0, alive=True, count=1):
      self.segment = segment
      self.point = point
      self.statistics = statistics
      self.score = score
      self.alive = alive
      self.score_node = None
      self.count = count
      self.partition_node = None
    
   @property
   def id(self):
      return self   
          
   def save(self):
      pass
      
   def separable(self, config):
      """
         Get a representation of the point suitable for insertion in the Partition tree.
         
         Called by PartitionTree.separate()
         
         DEPRECATED, will be removed.
      """
      if self.point is not None: 
         return self.point
      elif self.bayes is not None:
         rep = self.bayes.edgeBinary()
         return rep
      elif self.binary:
         if self.segment.config.binaryPartition:
            return self.binary
         else:
            return self.binary.toArray(self.segment.config.dim)
      elif self.other:
         if config.behavioralPartition:
            return array(self.statistics)
         topology, angles = str(other).split("::")
         rnn = cpp.createFromTopology(topology)
         rnn = cpp.setBinaryAnglesAtDepth(rnn, RnnAngles.parse(angles).toCpp(),16)
         #rnn = cpp.convert(str(self.other))
         weights = cpp.getAngles(rnn)
         cpp.discard(rnn)
         del rnn
         return weights
      else:
         return None

   @classmethod
   def bulkSave(cls, segment, points, stats):
      segment.points.extend(points)
      
      for gp in points:
         try:
            stats.start("separate")
            sep = segment.config.separator(segment.config)
            sep.separate(segment.partitionTree, gp)
            stats.stop("separate")
            stats.start("insert")
            segment.scoreTree.insert(gp, segment.config, stats)
            stats.stop("insert")
         except SeparationException:
            logg.debug("Exception when separating or inserting points")
            gp.alive = False

   @classmethod
   def sampleTournament(cls, segment, temp, config):
      """
         Sample evolutionary annealing in tournament mode by walking the (balanced) score tree
      
         segment - the segment to sample 
         temp - the temperature at which to sample
         config - a evo.config.Config object containing parameters "pressure" and "learningRate"
         
         if config.shiftToDb is True, this will attempt to call sampleTournamentInDb.
      """
      current = segment.scoreTree.root
      
      while True:
         children = sorted(current.children, key=lambda c: not c.left)
      
         if len(children) == 0:
            break
         
         try:
            if children[0].max_score == children[1].min_score:
               p = .5
            else:
               p = 1. / (1. + ((1. - config.pressure) ** (config.learningRate / temp * (2 ** children[0].height))))
         except:
            p = 1.0
         
         if p < 1.0:
            p = (p / (1. - p))
            p *= children[0].area
            div = p + children[1].area
            if div > 0.0:
               p /= div
            else:
               p = 0.5
         
         rnd = random.random_sample()
         if (config.minimize and rnd > p or
             not config.minimize and rnd < p):
            current = children[0]
         else:
            current = children[1]
         
      return current.point, current.point.partition_node

   @classmethod
   def sampleProportional(cls, segment, temp, config):
      """
         Sample evolutionary annealing in proportional mode by walking the (balanced) score tree.
      
         segment - the segment to sample 
         temp - the temperature at which to sample
         config - a evo.config.Config object containing parameters "pressure" and "learningRate"
         
         if config.shiftToDb is True, this will attempt to use a stored procedure for postgres.
      """
      # choose the proper center
      # this function has better approximates to the right of the center
      center = config.taylorCenter
      offset = 0
      depth = config.taylorDepth
      
      current = segment.scoreTree.root
      
      children = sorted(current.children, key=lambda c: not c.left)
      ns = array([i+0. for i in xrange(depth)])
      diffs = (1./temp - 1./center) ** ns
      while len(children) > 0:
         # build choice
         p1 = (children[0].taylor[offset:offset+depth] * diffs).sum()
         p2 = (children[1].taylor[offset:offset+depth] * diffs).sum()
         if p1 > 0.0 or p2 > 0.0:
            p1 = p1 / (p1 + p2)
         else:
            p1 = .5
         idx = 1
         if random.random_sample() < p1:
            idx = 0
         current = children[idx]
         children = sorted(current.children, key=lambda c: not c.left)
      
      return current.point, current

   @classmethod
   def sampleRegionalTournament(cls, segment, temp, config):
      """
         Sample regions from partition tree using a tournament at each node
      """
      current = segment.partitionTree.root
      currentArea = current.area
      children = current.children
      q = .5 + .5 * (config.pressure ** (1. / (temp*config.learningRate)))
      while len(children) > 0:
         try:
            if children[1].best_score == children[0].best_score:
               p = children[0].area / (children[0].area + children[1].area)
            else:
               if children[0].best_score > children[1].best_score:
                  p0 = q * children[0].area
                  p1 = (1. - q) * children[1].area
               else: #  children[1].best_score > children[0].best_score:
                  p0 = (1. - q) * children[0].area
                  p1 = q * children[1].area
               p = p0 / (p0 + p1)
         except:
            p = .5
         if random.random_sample() < p:
            current = children[0]
            currentArea = children[0].area
         else:
            current = children[1]
            currentArea = children[1].area
         children = current.children
      
      return current.point, current 


class SeparationAlgorithm(object):
   """A algorithm that enables traversal and separation of the partition tree.
   This class abstracts the space-dependent portions of the partitioning
   logic.
   
   :param config: Configuration parameters
   :type config: :class:`Config`
   
   """
   def __init__(self, config):
      if not self.compatible(config.space):
         err = "{0} is not compatible with space {1}"
         raise ValueError(err.format(self.__class__.__name__, config.space))
      self.config = config
      
   def compatible(self, space):
      """Check whether this partitioning method is compatible with a given space
      
      :param space: The space to check
      :type space: class:`Space`
      
      """
      return True
   
   def separate(self, tree, point):
      """Insert the point into the partition tree,
         separating the current partition that contains it.
         
         :param tree: The partition tree
         :type tree: :class:`Partition`
         :param point: The point to insert
         :type point: ``config.space.type``
         :param config: A :class:`Config` object
         :type config: :class:`Config`
         :param stats: - an :class:`RunStats` object
         :type stats: :class:`RunStats`
         
      """
      self.config.stats.start("separate.traverse")
      lr = self.config.learningRate
      node, path = tree.traverse(tree, point)
      self.config.stats.stop("separate.traverse")
      if node.point is None:
         node.point = point
         point.partition_node = node
         tree.areaTree.insert(node)
         return
      
      self.config.stats.start("separate.main")
      self.config.stats.start("separate.prepare")
      
      other = node.point
      n1, n2 = self.split(point, node)
         
      # update the best score
      if path is not None:
         if self.config.minimize and point.score < other.score:
            for n in path:
               n.best_score = min([point.score,n.best_score])
         elif point.score > other.score:
            for n in path:
               n.best_score = max([point.score,n.best_score])
      
      self.config.stats.stop("separate.compute")
      self.config.stats.stop("separate.main")
      
      self.config.stats.start("separate.areaTree")
      
      try:
         tree.areaTree.insert(n1)
         tree.areaTree.insert(n2)
         tree.areaTree.remove(node)
      except:
         traceback.print_exc()
         raise
      
      self.config.stats.stop("separate.areaTree")
      self.config.stats.start("separate.propagate")
      
      # correct area in score tree
      if other.score_node is not None:
         sn = other.score_node
         diff = sn.area - other.partition_node.area
         sn.segment.scoreTree.propagateAreaOnly(sn, self.config, diff)
      
      #print "end separate"
      self.config.stats.stop("separate.propagate")
      
   def split(self, point, node):
      """Do the actual work to partition a region.
      
      :param point: The point that resulted in the split
      :type point: ``config.space.type``
      :param node: A node from the partition tree corresponding to a region
                   of the space
      :type node: :class:`PartitionTreeNode`
      :throws: :class:`SeparationException` when separation of the node fails
      
      """
      raise NotImplementedError("Abstract Separation Algorithm; use a subclass")

class VectorSeparationAlgorithm(SeparationAlgorithm):
   def compatible(self, space):
      return space.type == ndarray
   
   def cast(self, point):
      return point.point
   
   def rectify(self, pointrep, otherrep):
      try:
         if len(pointrep) == len(otherrep):
            return pointrep, otherrep
         
         pointrep = list(pointrep)
         otherrep = list(otherrep)
      
         while len(pointrep) < len(otherrep):
            pointrep.append(0)
         while len(pointrep) > len(otherrep):
            otherrep.append(0)
            
      except:
         pass
      
      return pointrep, otherrep
   
   def chooseIndex(self, pointrep, otherrep, lower, upper):
      newDiff = 0
      newIndex = 0
      for i in xrange(len(pointrep)):
         if lower[i] == upper[i]:
            continue
         diff = abs(pointrep[i] - otherrep[i])
         if diff > newDiff:
            newDiff = diff
            newIndex = i
      
      return newIndex, newDiff
   
   def makeParts(self, point, pointrep, other, otherrep, newIndex, lower, upper):
      if pointrep[newIndex] > otherrep[newIndex]:
         upPoint = point
         downPoint = other
         midPoint = (otherrep[newIndex] 
                     + (pointrep[newIndex] - otherrep[newIndex])/2.)
      else:
         upPoint = other
         downPoint = point
         midPoint = (pointrep[newIndex] 
                     + (otherrep[newIndex] - pointrep[newIndex])/2.)
      
      downUpper = upper.copy()
      downLower = lower.copy()
      downUpper[newIndex] = midPoint
      down = Hyperrectangle(downLower, downUpper)
      down.parent = other.partition_node.bounds
      down.owner = self.config.space
      
      upUpper = upper.copy()
      upLower = lower.copy()
      upLower[newIndex] = midPoint
      up = Hyperrectangle(upLower, upUpper)
      up.parent = other.partition_node.bounds
      up.owner = self.config.space
      
      return down, up, downPoint, upPoint
   
   def split(self, point, node):
      """A vector-based separation algorithm that generates
      Hyperrectangular partitions parallel to the axes.
      
      """
      other = node.point
      node.point = None
      
      newIndex = 0
      newDiff = 0
      midPoint = 0
      upPoint = other
      downPoint = point
      self.config.stats.stop("separate.prepare")
      self.config.stats.start("separate.separable")
      pointrep = self.cast(point)
      otherrep = self.cast(other)
      pointrep, otherrep = self.rectify(pointrep, otherrep)
      lower, upper = node.bounds.extent()
      newIndex, newDiff = self.chooseIndex(pointrep, otherrep, lower, upper)
      self.config.stats.stop("separate.separable")
      if newDiff == 0.0:
         node.point = other
         raise SeparationException("No difference in points")
      
      self.config.stats.start("separate.compute")
      down, up, downPoint, upPoint = self.makeParts(point, pointrep,
                                                    other, otherrep,
                                                    newIndex,
                                                    lower, upper)
      
      downArea = float(down.area(index=newIndex))      
      upArea = float(up.area(index=newIndex))
      
      if upArea != upArea or downArea != downArea:
         raise SeparationException("NaN area!")
      
      n1 = PartitionTreeNode(
         segment=node.segment,
         point=upPoint,
         area = upArea,
         bounds = up,
         index = newIndex,
         upper=None,
         lower=None,
         best_score = upPoint.score,
         score_sum = upPoint.score,
         parent = node,
         layer = 0,
         layerrep = ''
      )
    
      n2 = PartitionTreeNode(
         segment=node.segment,
         point=downPoint,
         area = downArea,
         bounds = down,
         index = newIndex,
         upper=None,
         lower=None,
         best_score = downPoint.score,
         score_sum = downPoint.score,
         parent = node,
         layer = 0,
         layerrep = ''
      )

      node.children = [n1,n2]
      upPoint.partition_node = n1
      downPoint.partition_node = n2
      
      return n1, n2
 

class LongestSideVectorSeparationAlgorithm(VectorSeparationAlgorithm):
   """Like :class:`VectorSeparationAlgorithm`, except that it always
   partitions the longest side. This can be helpful to guarantee that the
   longest side length of any partition hyperrectangle is reduced
   evenly.
   
   """
   def chooseIndex(self, pointrep, otherrep, lower, upper):
      newDiff = 0.0
      infinity = False
      for i in xrange(len(pointrep)):
         if upper[i] == inf and lower[i] == -inf:
            return i, inf
         elif upper[i] == inf:
            if infinity:
               if -lower[i] > newDiff:
                  newDiff = -lower[i]
                  newIndex = i
            else:
               infinity = True
               newDiff = -lower[i]
               newIndex = i
         elif lower[i] == -inf:
            if infinity:
               if upper[i] > newDiff:
                  newDiff = upper[i]
                  newIndex = i
            else:
               infinity = True
               newDiff = upper[i]
               newIndex = i
         elif not infinity:
            diff = upper[i] - lower[i]
            if diff > newDiff:
               newDiff = diff
               newIndex = i
            
      if infinity:
         return newIndex, inf
      
      return newIndex, newDiff


class BinarySeparationAlgorithm(VectorSeparationAlgorithm):
   def compatible(self, space):
      return space.type == TernaryString
   
   def rectify(self, pointrep, otherrep):
      if pointrep.length < otherrep.length:
         pointrep.known &= (1L << (pointrep.length)) - 1L
         pointrep.length = otherrep.length
      
      if otherrep.length < pointrep.length:
         otherrep.known &= (1L << (otherrep.length)) - 1L
         otherrep.length = pointrep.length
         
      return pointrep, otherrep

   def chooseIndex(self, pointrep, otherrep, lower, upper):
      if (pointrep.known & pointrep.base) == (otherrep.known & otherrep.base):
         raise SeparationException("No difference between points")
      
      diff = pointrep ^ otherrep
      
      # randomly wrap the bits, and then take the first difference
      wrap = random.randint(0,diff.length)
      maskAll = (1L << diff.length) - 1L
      maskLow = 1L << wrap
      wrapHigh = (diff.base & maskAll) // maskLow
      wrapKnownH = (diff.known & maskAll) // maskLow
      maskLow -= 1L
      wrapLow = diff.base & maskLow
      wrapKnownL = diff.known & maskLow
      rebuilt = TernaryString(wrapHigh | (wrapLow << wrap),
                              wrapKnownH | (wrapKnownL << wrap),
                              diff.length)
      mask = 1L
      for i in xrange(rebuilt.length):
         if rebuilt[i]:
            return ((i + (diff.length - wrap)) % diff.length), 1.0
      
      raise SeparationException("No difference in points")
   
   def makeParts(self, point, pointrep, other, otherrep, newIndex, lower, upper):
      if pointrep[newIndex]:
         upPoint = point
         downPoint = other
      else:
         upPoint = other
         downPoint = point
      
      mask = 1L << newIndex
      
      downSpec = TernaryString((lower.base & lower.known) & ~mask,
                               lower.known | mask,
                               lower.length)
      down = BinaryRectangle(downSpec)
      down.parent = other.partition_node.bounds
      
      upSpec = TernaryString((upper.base & upper.known) | mask,
                             upper.known | mask,
                             upper.length)
      up = BinaryRectangle(upSpec)
      up.parent = other.partition_node.bounds
      
      print downSpec.base, downSpec.known, downSpec.length
      print upSpec.base, upSpec.known, upSpec.length
      return down, up, downPoint, upPoint


class BayesSeparationAlgorithm(BinarySeparationAlgorithm):
   """Separation algorithm for Bayesian networks based on structure."""
   def compatible(self, space):
      from pyec.distribution.bayes.net import BayesNet
      return space.type == BayesNet
   
   def cast(self, point):
      return point.point.edgeBinary()


class PartitionTreeNode(object):
   idgen = 1
   
   def __init__(self, segment, index, parent, upper, lower, area,
                bounds, point=None, layer=None, layerrep=None,
                score_sum=0.0, best_score=None):
      self.segment = segment,
      self.index = index
      self.parent = parent
      self.children = []
      self.upper = upper
      self.lower = lower
      self.parent = parent
      self.area = area
      self.layer = layer
      self.layerrep = layerrep
      self.score_sum = score_sum
      self.best_score = best_score
      self.point = point
      self.traversals = 0
      self.bounds = bounds
      self.id = self.__class__.idgen
      self.__class__.idgen += 1


class Partition(object):
   areaTree = None
   
   def __init__(self, segment, config):
      self.config = config
      self.segment = segment
               
      self.root = PartitionTreeNode(
         segment = self.segment,
         index = 0,
         parent = None,
         upper = None,
         lower = None,
         area = config.space.area(),
         bounds = config.space,
         point = None
      )
      
      self.areaTree = AreaTree()
      
   def save(self):
      pass

   def traverse(self, tree, point):
      """
         Traverse the partition tree to find the partition within which "point" is located.
         
         :param tree: The partition tree
         :type tree: :class:`Partition`
         :param point: The point for searching.
         :type point: ``config.space.type``
         :returns: A tuple with (partitionNode, path); with the
                   :class:`PartitionTreeNode` for the matching partition, and
                   the list of tree nodes traversed

      """
      # get the top parent
      current = tree.root
      region = tree.root.bounds
      path = [current]
      
      last = None
      children = current.children
      while len(children) > 0:
         if current == last:
            print "partition.traverse looped, exception"
            print current.bounds.extent()
            print point.point
            raise Exception("Loop in partition.traverse!")
         last = current
         
         for child in children:
            if child.bounds.in_bounds(point.point, index=child.index):
               current = child
               path.append(current)
               children = current.children
               break
            
      node = current
      
      return node, path

   def propagateScoreSum(self, node, config):
      """
         Propagate the score up the tree
         
         node - the node to start at (bottom-up traversal)
         config - an evo.config.Config
         
      """
      current = node
      while current is not None:
         current.score_sum = sum([child.score_sum for child in current.children])
         current = current.parent
      
   def computeDepth(self, point):
      """
         compute the depth of the leaf that points to the db_point with id pointId
         
      """
      depth = 0
      current = point.partition_node
      while current is not None:
         current = current.parent
         if current is not None:
            depth += 1
      
      return depth

   def traverseLayered(self, pointrep, segment, config, stats):
      """
         Traverse the tree to the split point, track the criteria as you go.
         
         Start at top layer, traverse to boundary
         When boundary is identified, then check whether to proceed
         into next layer; if not, break at the boundary
      """
      raise NotImplementedError("Need to abstract this to a LayeredSpace")
      current = self.root
      children = current.children
      currentrep = current.layerrep
      
      for layer, layerrep in enumerate(pointrep):
         #print "NEW LAYER traverse: ", current[0], layer, layerrep
         layerspec = config.layerspec(layer, pointrep, config)
         lower = layerspec.lower()
         upper = layerspec.upper()
         last = None
         
         while len(children) > 0 and children.layer == layer:
            if current == last:
               #if not (pointrep < pointrep).any():
               #   print "partition.traverse looped, exception"
               #   print lower
               #   print upper
               #   print pointrep
               raise Exception, "Loop in partition.traverseLayers!"
            last = current
            
            for child in children:
            
               if (layerrep[child.index] <= child.upper) and \
                  (layerrep[child.index] >= child.lower):
                  current = child
                  currentrep = current.layerrep
                  children = current.children
                  lower[current.index] = current.lower
                  upper[current.index] = current.upper
                  
                  break
         
               
         # should we check next layer or return here?
         ret = False
         # we are at a layer boundary; do the layerreps match?
         #print "deciding whether to return, ", layer
         #print currentrep
         #print pointrep
         if currentrep.split(";")[layer] != layerspec.serializeLayer(layerrep,layer):
            #print "return", layer
            ret = True

         
         if ret:
            node = current
            return node, lower, upper, layer
      
                        
      node = current
      return node, lower, upper, len(pointrep) - 1

   def separateLayered(self, point, config, stats):
      """
         Separate points according to a sequence of criteria
         Each element in the sequence is a separate portion of the tree
         If two points reach the boundary in the tree between 
         criteria, then the node at the boundary is split.
      """
      raise NotImplementedError("Need to abstract this to LayeredSeparationAlgorithm")
      pointrep = config.layerize(getattr(point, config.activeField))
      #print [unicode(layer) for layer in pointrep]
      stats.start("separate.traverse")
      node, lower, upper, layer = self.traverseLayered(pointrep, point.segment, config, stats)
      layerspec = config.layerspec(layer, pointrep, config)
      
      #print "traversed: ", layer, pointrep[layer]
      
      stats.stop("separate.traverse")
      
      isLeaf = len(node.children) == 0
      if isLeaf and node.parent is None and node.point is None:
         node.point = point
         node.layer = layer
         node.area = layerspec.area()
         node.layerrep = layerspec.serialize(pointrep)
         point.partition_node = node
         return
         
               
      stats.start("separate.main")
      stats.start("separate.prepare")
      
      if isLeaf:
         other = node.point
         otherrep = config.layerize(getattr(other, config.activeField))
         node.point = None
      else:
         otherrep = layerspec.deserialize(node.layerrep)
      
      #print "point: ", [unicode(layer2) for layer2 in pointrep]
      #print "other: ", [unicode(layer2) for layer2 in otherrep]
      
      # node.save()
      
      newIndex = 0
      newDiff = 0
      midPoint = 0
      upPointIsOther = True
      stats.stop("separate.prepare")
      stats.start("separate.separable")
      
      pointrep1 = pointrep[layer]
      otherrep1 = otherrep[layer]
      #print "separating: "
      #print pointrep
      #print otherrep
      try:
       for i in xrange(len(pointrep1)):
         if lower[i] == upper[i]:
            continue
         diff = abs(pointrep1[i] - otherrep1[i])
         #print i, " - ", pointrep1[i], otherrep1[i]
         #print i, " - ", diff
         if diff > newDiff:
            newDiff = diff
            newIndex = i
            if pointrep1[i] > otherrep1[i]:
               upPointIsOther = False
               midPoint = (pointrep1[i] + otherrep1[i])/2.
            else:
               upPointIsOther = True
               midPoint = (otherrep1[i] + pointrep1[i])/2.
      except:
         #print "EXCEPTION in separateLayered"
         #print "node: ", node.id
         #print "layer: ", layer
         #print "point: ", pointrep
         #print "rep: ", point.other
         #print "weights: ", cpp.getAngles(cpp.convert(point.other))
         #print "other: ", otherrep
         #print "rep: ", node.point.other 
         #print "weights: ", cpp.getAngles(cpp.convert(node.point.other))
        
         #self.printTree(point.segment)
         raise
      stats.stop("separate.separable")
      stats.start("separate.compute")
      
      #print "got diff"
      
      if newDiff == 0.0:
        if isLeaf:
           node.point = other
        #print "layer: ", layer
        #print "unseparable: ", point.id, other.id
        #print layerspec.serialize(pointrep)
        #print layerspec.serialize(otherrep)
        # node.save()
        #print point.other
        #print other.other
        raise SeparationException, "No difference in points"
      
      
      if isLeaf:
         #print "partition at index ", newIndex, " with diff ", newDiff
      
         # compute proportion that goes to the lower node
         proportion = layerspec.proportion(lower[newIndex], midPoint, upper[newIndex],ub=upper,lb=lower,idx=newIndex)
      
         #print "got proportion: ", lower[newIndex], midPoint, upper[newIndex], proportion
      
         
         upArea = float(node.area * (1.0 - proportion))
         downArea = float(node.area * proportion)
         if upArea < 0.0 or downArea < 0.0:
            print "negative! ", upArea, downArea, node.area, proportion, lower[newIndex], midPoint, upper[newIndex], layer
            print newIndex
            print pointrep1
            print otherrep1
            print upper
            print lower
            
      else:
         if upPointIsOther:
            upArea = node.area 
            downArea = layerspec.area()
            #print layer, " new area ", downArea, " old area ", upArea
         else:
            upArea = layerspec.area()
            downArea = node.area
            #print layer, " new area ", upArea, " old area ", downArea
         
      
      stats.stop("separate.compute")
      stats.start("separate.sql")
      
      node.point = None
      
      if upPointIsOther:
         down = point
         downScore = point.score * downArea
         downBest = point.score
         downLayer = layerspec.serialize(pointrep)
         if isLeaf:
            up = other
            upScore = other.score * upArea
            upBest = other.score
            upLayer = node.layerrep
         else:
            up = None
            upScore = node.score_sum
            upBest = node.best_score
            upLayer = layerspec.serialize(otherrep)
      else:
         up = point
         upScore = point.score * upArea
         upBest = point.score
         upLayer = layerspec.serialize(pointrep)
         if isLeaf:
            down = other
            downScore = other.score * downArea
            downBest = other.score
            downLayer = node.layerrep
         else:
            down = None
            downScore = node.score_sum
            downBest = node.best_score
            downLayer = layerspec.serialize(otherrep)
      
      
      #print upPointIsOther, upLayer, downLayer
      
      if not isLeaf:
        # get the current child ids
        children = node.children
      
      n1 = PartitionTreeNode(
         upper=float(upper[newIndex]), 
         lower=float(midPoint),
         segment=node.segment,
         point=up,
         area = upArea,
         index = newIndex,
         best_score = upBest,
         score_sum = upScore,
         parent = node,
         layer = layer,
         layerrep = upLayer
      )
    
      n2 = PartitionTreeNode(
         upper=float(midPoint), 
         lower=float(lower[newIndex]),
         segment=node.segment,
         point=down,
         area = downArea,
         index = newIndex,
         best_score = downBest,
         score_sum = downScore,
         parent = node,
         layer = layer,
         layerrep = downLayer
      )

      node.children = [n1,n2]
      if up is not None:
         up.partition_node = n1
      if down is not None:
         down.partition_node = n2

         
      if not isLeaf:
         # update the children
         if upPointIsOther:
            parent = n1
         else:
            parent = n2
            
         for child in children:
            child.parent = parent
         parent.children = children
         
         
      # propagate the new area and score sum
      parent = node
      while parent is not None:
         parent.area = sum([child.area for child in node.children])
         parent.score_sum = sum([child.score_sum for child in node.children])
         parent.best_score = sum([child.best_score for child in node.children])
         parent = node.parent
      
      
      stats.stop("separate.sql")
      stats.stop("separate.main")
      stats.start("separate.propagate")
      
      # correct area in score tree
      if isLeaf:
         sn = other.score_node
         if sn is not Node:
            if not upPointIsOther:
               diff = sn.area - downArea
            else:
               diff = sn.area - upArea
            sn.segment.scoreTree.propagateAreaOnly(sn, config, diff)
      
      #print "end separate"
      stats.stop("separate.propagate")   
      
      #print "after"
      #self.printTree(point.segment)
      
      
   def printTree(self, segment, node=None, indent=""):
      if node is None:
         node = self.root
      print indent, node.id, ": ", node.layer, node.index, node.lower, node.upper, node.area
      children = node.children
      for child in children:
         self.printTree(segment, child, indent + "\t")      

   def largestArea(self):
      try:
         return self.areaTree.largest()
      except AreaException:
         return 1.0

      
class ScoreTreeNode(object):
   idgen = 1

   def __init__(self, segment, parent=None, point=None, area=1.0, min_score=-1e300, max_score=1e300, child_count=1, taylor=None, height=0, balance=0, left=False):
      self.segment = segment
      self.parent = parent
      self.point = point
      self.area = area
      self.min_score=min_score
      self.max_score=max_score
      self.child_count=child_count
      self.taylor=taylor
      self.height=height
      self.balance=balance
      self.left=left
      self.children = []
      self.id = self.__class__.idgen
      self.__class__.idgen += 1
      
class ScoreTree(object):
   
   def __init__(self, segment, config):
      self.config = config
      self.segment = segment
      self.root = ScoreTreeNode(
         segment = self.segment,
         parent = None,
         point = None
      )
      
   def save(self):
      pass

   def printTree(self, segment, node=None, indent=''):
      """
         Recursively print the score tree.
         
         segment - the segment for the tree to print
         node - the node to start at, or None for the root
         indent - a prefix for printing; tabs added for each new level
      """
      if node is None: 
         node = self.root
      
      print indent, node.id, ',', node.balance, ",", node.area, ",", node.height,
      print ",", node.child_count, ",", node.min_score, " - ", node.max_score
      
      children = node.children
      if len(children) > 0:
         for child in sorted(children, key=lambda c: not c.left):
            self.printTree(segment, child, indent + '\t')

   def rotateLeft(self, node, config):
      """
         Perform a left rotation at a node in the tree. For balancing.
         
         node - the node to rotate left.
         config - a config object with property "shiftToDb"
         
         Returns -1, the change in tree height. This return will only be correct
         if used in the context of an AVL tree balancing algorithm - otherwise
         the height change would need to be computed differently (it is possible).
      """
      center = node
      children = sorted(node.children, key=lambda c: not c.left)
      left = children[0]
      right = children[1]
      
      right.left = center.left
      center.left = True
      left.left = True
      
      children = sorted(right.children, key=lambda c: not c.left)
      rightleft = children[0]
      rightright = children[1]   
      
      rightleft.parent = center
      rightleft.left = False
      right.parent = center.parent
      center.parent = right

      center.area = left.area + rightleft.area
      right.area = center.area + rightright.area
      
      center.taylor = left.taylor + rightleft.taylor
      right.taylor = center.taylor + rightright.taylor
      
      center.min_score = rightleft.min_score
      center.max_score = left.max_score
      right.min_score = rightright.min_score
      right.max_score = center.max_score
      
      
      center.height = max([left.height, rightleft.height]) + 1
      right.height = max([center.height, rightright.height]) + 1

      center.child_count = left.child_count + rightleft.child_count + 1
      right.child_count = center.child_count + rightright.child_count + 1

      center.balance = center.balance + 1
      if right.balance < 0:
         center.balance = center.balance - right.balance
      
      right.balance = right.balance + 1
      if center.balance > 0:
         right.balance = right.balance + center.balance
      
        
      center.children = [left,rightleft]
      right.children = [center,rightright]
      if right.parent is None:
         right.segment.scoreTree.root = right
      else:
         newChildren = [right]
         for child in right.parent.children:
            if child != center:
               newChildren.append(child)
         right.parent.children = newChildren 
         
      return -1

   def rotateRight(self, node, config):
      """
         Perform a right rotation at a node in the tree. For balancing.
         
         node - the node to rotate right.
         config - a config object with property "shiftToDb"
         
         Returns -1, the change in tree height. This return will only be correct
         if used in the context of an AVL tree balancing algorithm - otherwise
         the height change would need to be computed differently (it is possible).
      """
      center = node
      children = sorted(node.children, key=lambda c: not c.left)
      
      left = children[0]
      right = children[1]
      
      left.left = center.left
      center.left = False
      right.left = False
      
      children = sorted(left.children, key=lambda c: not c.left)
      leftleft = children[0]
      leftright = children[1]

      leftright.parent = center
      leftright.left = True
      left.parent = center.parent
      center.parent = left

      center.area = right.area + leftright.area
      left.area = center.area + leftleft.area
      
      center.taylor = right.taylor + leftright.taylor
      left.taylor = center.taylor + leftleft.taylor
   
      center.min_score = right.min_score
      center.max_score = leftright.max_score
      left.min_score = center.min_score
      left.max_score = leftleft.max_score
      

      center.height = max([right.height, leftright.height]) + 1
      left.height = max([center.height, leftleft.height]) + 1

      center.child_count = leftright.child_count + right.child_count + 1
      left.child_count = center.child_count + leftleft.child_count + 1

      center.balance = center.balance - 1
      if left.balance > 0:
         center.balance = center.balance - left.balance

      left.balance = left.balance - 1
      if center.balance < 0:
         left.balance = left.balance + center.balance

      center.children = [leftright,right]      
      left.children = [leftleft,center]
      if left.parent is None:
         left.segment.scoreTree.root = left
      else:
         newChildren = [left]
         for child in left.parent.children:
            if child != center:
               newChildren.append(child)
         left.parent.children = newChildren 
      
      return -1
      
      
   def traverse(self, point, config):
      """
         Traverse the score tree.
         
         point - the point to compare
         config - an evo.config.Config instance with property "shiftToDb"
         
         Returns node in the tree that must be split to insert the new point into the tree.
      """
      current = self.root
      while True:
         children = sorted(current.children, key=lambda c: not c.left)
         if len(children) == 0:
            break
         if children[1].max_score < point.score:
            current = children[0]
         else:
            current = children[1]
      return current

   def removeLeaf(self, node, config):
      if node.parent is None:
         if node.point is not None:
            node.point.score_node = None
         node.point = None
         return
          
      parent = node.parent
      other = [child for child in node.parent.children if child != node][0]
      
      left = True
      for i,child in enumerate(parent.parent.children):
         if child == parent:
            left = i == 0 
            break
      
      parent.area = other.area
      parent.point = other.point
      parent.min_score = other.min_score
      parent.max_score = other.max_score
      parent.taylor = other.taylor
      parent.balance = 0
      parent.children = []
      del node
      del other
      parent.height -= 1
      if parent.parent is not None:
         #print "before, ", parent.parent_id
         #self.printTree(parent.segment)
      
      
         superparent = parent.parent
         if left:
            superparent.balance -= 1
         else:
            superparent.balance += 1
         if superparent.balance == 0:
            superparent.height -= 1
         
         self.propagateArea(superparent, config, -1)
         
         #print "after, ", superparent.id
         #self.printTree(parent.segment)
      

   def propagateAreaOnly(self, node, config, diff):
      """
         Propagate the change in area given by "diff" 
      """
      parent = node
      while parent is not None:
         parent.area -= float(diff)
         parent = parent.parent
      

   def propagateArea(self, node, config, inserted=1):
      """
         Propagate the area and other features up the score tree, balancing as needed using an AVL algorithm.
         
         For balancing, positive numbers are heavy on the left, negative numbers are heavy on the right.
         
         node - the node to start at (bottom-up traversal)
         config - an evo.config.Config object with property "shiftToDb"
         inserted - how many new levels have been added to the tree, 1 or 0. If 0, no balancing is needed.
      """
      
      if inserted > 0:
         heightChange = inserted
      elif inserted < 0:
         if node.balance == 0:
            heightChange = -1
         else:
            heightChange = 0
      else:
         heightChange = 0
      current = node
      
      children = sorted(current.children, key=lambda c: not c.left)
      while True:
      
         left = children[0]
         right = children[1]
         
         if left.min_score < right.max_score:
            self.printTree(node.segment)
            raise Exception, "Violated ordered assumption!"
            
         
         # is the current the right or the left child?
         onLeft = current.left
         
         
         rotated = False
         if inserted != 0:
            if current.balance > 1:
               if left.balance != 0:
                  heightChange -= 1
               if left.balance >= 0:
                  rotated = True
                  self.rotateRight(current, config)
               else:# left.balance < 0:
                  rotated = True
                  self.rotateLeft(left, config)
                  self.rotateRight(current, config)
            elif current.balance < -1:
               if right.balance != 0:
                  heightChange -= 1
               if right.balance <= 0:
                  rotated = True
                  self.rotateLeft(current, config)
               else:# right.balance > 0:
                  rotated = True
                  self.rotateRight(right, config)
                  self.rotateLeft(current, config)
               
         
         if not rotated:
            current.area = left.area + right.area
            current.taylor = left.taylor + right.taylor
            current.child_count = left.child_count + right.child_count + 1
            current.min_score = right.min_score
            current.max_score = left.max_score
         
         next = None
         if current.parent is not None:
            next = current.parent
            children = sorted(next.children, key=lambda c: not c.left)
         
         if inserted != 0 and next is not None:
            oldbal = next.balance
            if onLeft:
               newbal = next.balance + heightChange
            else:
               newbal = next.balance - heightChange
            next.balance = newbal
            if inserted > 0:
               if newbal == 0:
                  heightChange = 0
            else:
               if newbal != 0:
                  heightChange = 0
            next.height += heightChange
         
         if next is None:
            break
         current = next
         
      #if inserted != 0:
      #   print "after, ", node.id, inserted
      #   self.printTree(node.segment)
      
      
   def insert(self, point, config, stats):
      """
         Insert a new point into the score tree, then propagate area, taylor scores, and rebalance.
         Assumes the point has already been inserted into the partition tree
         
         point - the point to insert
         config - an evo.config.Config object with properties "shiftToDb", "taylorDepth", "taylorCenter", "selection"
         stats - an evo.trainer.RunStats object
      """
      
      stats.start("insert.traverse")
      node = self.traverse(point, config)
      stats.stop("insert.traverse")
      stats.start("insert.main")
      lr = config.learningRate
      
      ns = [i+0. for i in xrange(config.taylorDepth)]
      fs = ns * 1
      if config.taylorDepth > 0:
         fs[0] = 1.0
         for i in xrange(config.taylorDepth - 1):
            fs[i+1] *= fs[i]
      ns = array(ns)
      fs = array(fs)
      center = config.taylorCenter      
      
      if node.point is None:
         node.point = point
         node.area = point.partition_node.area
         node.min_score = point.score
         node.max_score = point.score
         score = (config.minimize and -1 or 1) * point.score * lr 
         taylor = nan_to_num(score ** ns) / fs
         taylor *= node.area
         taylor *= nan_to_num(exp(score) ** (1./center)) 
         node.taylor = nan_to_num(taylor)
         return
      
      
      other = node.point
      node.point = None
      
      if other.score > point.score:
         upPoint = other
         downPoint = point
         node.max_score = other.score
         node.min_score = point.score
      else:
         upPoint = point
         downPoint = other
         node.max_score = point.score
         node.min_score = other.score
      
      node.height = 1
      
      upArea = upPoint.partition_node.area
      downArea = downPoint.partition_node.area
      
      score1 = (config.minimize and -1 or 1) * upPoint.score * lr    
      score2 = (config.minimize and -1 or 1) * downPoint.score * lr    
      taylor1 = (score1 ** ns) / fs
      taylor1 *= upArea
      taylor1 *= (exp(score1) ** (1./center)) 
      taylor2 = (score2 ** ns) / fs
      taylor2 *= downArea
      taylor2 *= (exp(score2) ** (1./center))
      
      if config.taylorDepth > 0:
         if (abs(taylor1) < 1e-300).all() and (abs(taylor2) < 1e-300).all():
            node.point = other
            node.height = 0
            node.max_score = other.score
            node.min_score = other.score
            # node has not been saved to the DB, so we don't need to rollback
            #node.save()
            #transaction.rollback()
            raise SeparationException, "Zero taylor coeffs"
      
      n1 = ScoreTreeNode(
         segment=node.segment,
         point=upPoint,
         area = upArea,
         parent = node,
         max_score = upPoint.score,
         min_score = upPoint.score,
         left = upPoint.score >= downPoint.score,
         taylor = taylor1
      )
      
      n2 = ScoreTreeNode(
         segment=node.segment,
         point=downPoint,
         area = downArea,
         parent = node,
         max_score = downPoint.score,
         min_score = downPoint.score,
         left = upPoint.score < downPoint.score,
         taylor = taylor2
      )
      node.children = [n1,n2]
      upPoint.score_node = n1
      downPoint.score_node = n2
      
      stats.stop("insert.main")
      stats.start("insert.propagate")
      self.propagateArea(node, config)
      stats.stop("insert.propagate")

   def resetTaylor(self, segment, temp, config):
      """
         Recompute the taylor coefficients on the Score tree.
         
         This should be done whenever 1/temp gets more than 0.5 away
         from the inverse of the taylor center for accuracy.
         
         Requires a loop through all points, but only needs to be done
         at a logarithmically often pace.
         
         segment - the segment to recompute
         temp - the new temperature center
         config - an evo.config.Config object with "shiftToDb", "taylorCenter", "taylorDepth"
         
         This method sets config.taylorCenter to temp when complete.
      """
      logg.info("resetTaylor: segment %s, new center %s, old center %s" % (segment, temp, config.taylorCenter))
      
      ns = [i+0. for i in xrange(config.taylorDepth)]
      fs = ns * 1
      if len(fs) > 0:
         fs[0] = 1.0
         for i in xrange(config.taylorDepth - 1):
            fs[i+1] *= fs[i]
      ns = array(ns)
      fs = array(fs)
      lr = config.learningRate
      center = temp
      
      next = []
      for point in segment.points:
         if point.alive and point.score_node is not None:
            next.append(point.score_node)
      
      heights = {0: next}
      height = 0
      while len(heights) > 0:
         nodes = set(heights[height])
         for node in nodes:
            if node.point is not None:
               score = config.minimize and -node.point.score or node.point.score
               score = score * lr
               taylor = nan_to_num(score ** ns) / fs
               taylor *= node.area
               taylor *= nan_to_num(exp(score/center)) 
               node.taylor = nan_to_num(taylor)
            else:
               node.taylor = zeros(config.taylorDepth)
               for child in node.children:
                  node.taylor += child.taylor
            if node.parent is not None:
               if not heights.has_key(node.parent.height):
                  heights[node.parent.height] = [node.parent]
               else:
                  if node.parent not in heights[node.parent.height]:
                     heights[node.parent.height].append(node.parent)
         del heights[height]
         height += 1
            

class AreaTreeNode(object):
    idgen = 1
    parent = None
    children = []
    val = None
    low = 0.0
    high = 1.0
    left = False
    
    def __init__(self):
        self.id = self.__class__.idgen
        self.__class__.idgen += 1
        

class AreaTree(object):
    
    @classmethod
    def get(cls,**kwargs):
        return cls.segment
      
    root = None
    map = {}
    
    def traverse(self, area):
        if self.root is None:
           raise AreaException("Cannot traverse empty tree")
        elif self.root.low > area or self.root.high < area:
           err = "Area {0} out of bounds ({1},{2})"
           err = err.format(area, self.root.low, self.root.high)
           raise AreaException(err)
           
        next = self.root
        while len(next.children):
            if area >= next.children[0].low:
                next = next.children[0]
            else:
                next = next.children[1]
        
        return next
    
    def insert(self, partitionNode):
        if self.root is None:
            self.root = AreaTreeNode()
            self.root.val = partitionNode
            assert self.root.val is not None
            return
            
        parent = self.traverse(partitionNode.area)
        
        mid = partitionNode.area
        
        left = AreaTreeNode()
        left.left = True
        left.parent = parent
        left.high = parent.high
        left.low = mid
        
        right = AreaTreeNode()
        right.left = False
        right.parent = parent
        right.high = mid
        right.low = parent.low
        
        if parent.val.area >= partitionNode.area:
            right.val = partitionNode
            left.val = parent.val
        else:
            left.val = partitionNode
            right.val = parent.val
        
        assert left.val is not None
        assert right.val is not None
        assert left.low >= right.high
        assert left.val.area >= right.val.area    
                    
        parent.val = None
        parent.children = [left,right]
        
        self.map[left.val.id] = left
        self.map[right.val.id] = right
        
        #print "AFTER INSERT: "
        #self.printTree()
        
    def remove(self, partitionNode):        
        if not self.map.has_key(partitionNode.id):
            err = "Area tree has no node {0}".format(partitionNode.id)
            raise AreaException(err)
            
        target = self.map[partitionNode.id]
        #print "Removing ", target.id
        if target == self.root:
            raise AreaException("Cannot remove root")
        
        if target.left:
            other = target.parent.children[1]
        else:
            other = target.parent.children[0]
        
        #print "Other: ", other.id
        #print "Parent: ", target.parent.id
        
        #assert len(other.children) or other.val is not None
        
        target.parent.val = other.val
        target.parent.children = other.children
        for child in other.children:
            child.parent = target.parent
        other.val = None
        target.val = None
        if target.parent.val is not None:
            self.map[target.parent.val.id] = target.parent

        #assert len(target.parent.children) or target.parent.val is not None 
        
        del other
        del target
        del self.map[partitionNode.id]
              
        #print "AFTER REMOVE: "
        #self.printTree()
      
    def largest(self):
        if self.root is None:
            raise AreaException("Area tree has no root")
        
        next = self.root
        while len(next.children):
            #print "id: ", (next.low, next.high)
            next = next.children[0]
        
        return next.val.area
    
    def printTree(self, node=None, indent=""):
        if node is None:
            node = self.root
        print indent, node.id, node.low, node.high,
        print node.val is None and "None" or node.val.area
        children = node.children
        for child in children:
            self.printTree(child, indent + "\t") 
   