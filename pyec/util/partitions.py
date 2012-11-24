"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from numpy import *
from scipy.special import erf

import traceback

import logging
logg = logging.getLogger(__file__)

class SeparationException(Exception):
   pass

class AreaException(Exception):
   pass

class ClassProperty(property):
   def __get__(self, cls, owner):
      return self.fget.__get__(None, owner)()

class Segment(object):
   segment = None
   
   @classmethod
   def _objects(cls):
      return cls.segment   

   @classmethod
   def _blank(cls):
      pass

   objects = ClassProperty(_objects,_blank) 

   @classmethod
   def get(cls,**kwargs):
      return cls.segment

   def __init__(self, name, config):
      self.name = name
      self.config = config
      
      Point.segment = self
      Partition.segment = self
      ScoreTree.segment = self
      Segment.segment = self
      
      self.points = []
      self.partitionTree = Partition(self,config)
      self.scoreTree = ScoreTree(self,config)
      Point.segment = self
      
   def clearSegment(self):
      del self.points
      del self.partitionTree
      del self.scoreTree
      self.points = []
      self.partitionTree = Partition(self,config)
      self.scoreTree = ScoreTree(self,config)
      
      
class Point(object):
   segment = None

   @classmethod
   def _objects(cls):
      return cls 

   @classmethod
   def _blank(cls):
      pass

   objects = ClassProperty(_objects,_blank)

   def __init__(self, segment, point=None, bayes=None, binary=None, other=None,statistics = None, score=0.0, alive=True,count=1):
      self.segment = segment
      self.point = point
      self.bayes = bayes
      self.binary = binary
      self.other = other
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
   def bulkSave(cls, points, stats):
      cls.segment.points.extend(points)
      
      for gp in points:
         try:
            stats.start("separate")
            cls.segment.partitionTree.separate(gp, cls.segment.config, stats)
            stats.stop("separate")
            stats.start("insert")
            cls.segment.scoreTree.insert(gp, cls.segment.config, stats)
            stats.stop("insert")
         except:
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
               p = 1. / (1. + ((1. - config.pressure) ** (config.learningRate * temp * (2 ** children[0].height))))
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
         
         if random.random_sample() < p:
            current = children[0]
         else:
            current = children[1]
         
      config.selectedScores.append(current.point.score)
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
      diffs = (temp - center) ** ns
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
      
      
      config.selectedScores.append(current.point.score)
      return current.point, current

   @classmethod
   def samplePartition(cls, segment, startNode, config):
      
      current = startNode
      currentArea = startNode.area
      children = current.children
      while len(children) > 0:
         try:
            p = children[0].score_sum / (children[0].score_sum + children[1].score_sum)   
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

   @classmethod
   def sampleTournamentSecondary(cls, segment, start, temp, config):
      node = start
      last = start
      while node.parent is not None and node.layer == config.layerspec.maxLayer:
         #print "at ", node.id, " with layer ", node.layer
         last = node
         node = node.parent 
      
      # second, sample tournament downward
      return cls.samplePartition(segment, node, config)

class PartitionTreeNode(object):
   idgen = 1
   
   def __init__(self, segment, index, parent, upper, lower, area, bounds, point=None, layer=None, layerrep=None, score_sum=0.0, best_score=0.0):
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
   segment = None
   areaTree = None
   
   @classmethod
   def _objects(cls):
      return cls.segment.partitionTree 

   @classmethod
   def _blank(cls):
      pass

   objects = ClassProperty(_objects,_blank)   

   def __init__(self, segment, config):
      self.config = config
      self.segment = segment
      self.center = config.center
      self.scale = config.scale
      if config.bounded and hasattr(config.in_bounds,'extent'):
         self.center,self.scale = config.in_bounds.extent()
      if not config.bounded:
         self.scale = inf
      
      lower = ones(config.dim) * (self.center-self.scale)
      upper = ones(config.dim) * (self.center+self.scale)   
               
      self.root = PartitionTreeNode(
         segment = self.segment,
         index = 0,
         parent = None,
         upper = self.center + self.scale,
         lower = self.center - self.scale,
         area = 1.0,
         bounds = (lower, upper),
         point = None
      )
      
      self.traverseCache = []
      self.traverseCacheSize = 5
      
      self.areaTree = AreaTree()
      
      
   def save(self):
      pass
      
   def gaussInt(self, z):
      # x is std normal from zero to abs(z)
      x = erf(abs(z)) / 2 / sqrt(2)
      return .5 + sign(z) * x
   
   def considerCaching(self, node, lower, upper, path):
      if len(self.traverseCache) < self.traverseCacheSize or \
         node.traversals >= self.traverseCache[-1][0].traversals:
         while len(self.traverseCache) >= self.traverseCacheSize:
            self.traverseCache = self.traverseCache[:-1]
            
         self.traverseCache.append((node, lower, upper, path))
         self.traverseCache = sorted(self.traverseCache, key=lambda x:-x[0].traversals)
    
   def checkTraverseCache(self, pointrep):
      n,l,u,p = None,None,None,None
      for node, lower, upper, path in sorted(self.traverseCache, key=lambda x:-len(x[3])):
         low = iter(lower)
         up = iter(upper)
         match = True
         for v in pointrep:
            if low.next() > v or up.next() < v:
               match = False
               break
         if match:
            return node, lower, upper, path
      return n,l,u,p
      
            
   def traverse(self, point, config):
      """
         Traverse the partition tree to find the partition within which "point" is located.
         
         point - the point for searching.
         config - an evo.config.Config object with "dim", "center", and "scale" attributes.
         
         
         Returns (partition, upper, lower)
         
         partition - the matching partition
         upper - the upper boundaries of the current partition
         lower - the lower boundaries of the current partition
      """
      pointrep = point.separable(config)
      return self.traversePoint(point.segment, pointrep, config)
    
   def traversePoint(self, segment, pointrep, config):
      #print "traversing"
      if (pointrep < pointrep).any():
         raise Exception, "NaN sent to traverse!"
      
      """
      current,lower,upper,path = self.checkTraverseCache(pointrep)
      if current is not None:
         # cache hit
         lower = lower.copy()
         upper = upper.copy()
         path = [p for p in path]
      
      else:
      """
      # cache miss
      # get the top parent
      lower = ones(config.dim) * (self.center-self.scale)
      upper = ones(config.dim) * (self.center+self.scale)
      
      current = self.root
      path = [current]
      
      # end else
      
      last = None
      children = current.children
      while len(children) > 0:
         if current == last:
            #if not (pointrep < pointrep).any():
            #   print "partition.traverse looped, exception"
            #   print lower
            #   print upper
            #   print pointrep
            raise Exception, "Loop in partition.traverse!"
         last = current
         #current.traversals += 1
         #self.considerCaching(current,lower,upper,path)
         for child in children:
            enter = False
            if child.index < 0:
               enter = len(pointrep) <= child.upper
            
            if len(pointrep) <= child.index:
               try:
                  pointrep = list(pointrep)
                  while len(pointrep) <= child.index:
                     pointrep.append(0.0)
               except:
                  pass
            
            if enter or \
               (pointrep[child.index] <= child.upper) and \
               (pointrep[child.index] >= child.lower):
               current = child
               path.append(current)
               children = current.children
               if current.index >= 0:
                  lower[current.index] = current.lower
                  upper[current.index] = current.upper
               break
      node = current
      
      #self.traverseCache = sorted(self.traverseCache, key=lambda x:-x[0].traversals)
      
      #print "returning from traverse", len(path)
      #print lower
      #print upper
      return node, lower, upper, path
   
   def separate(self, point, config, stats):
      """
         Insert the point into the partition tree, separating the current partition that contains it.
         
         point - the point to insert
         config - a config object with properties "shiftToDb", "dim", "center", "scale"
         stats - an evo.trainer.RunStats object
         
         If shiftToDb is true, this will attempt to call traverseInDb()
      """
      #print "in separate"
      if config.layered:
         return self.separateLayered(point, config, stats)
      
      
      stats.start("separate.traverse")
      lr = config.learningRate
      node, lower, upper, path = self.traverse(point, config)
      stats.stop("separate.traverse")
      if node.point is None:
         node.point = point
         point.partition_node = node
         self.areaTree.insert(node)
         return
      
      #print "traversed"   
               
      stats.start("separate.main")
      stats.start("separate.prepare")
      
      other = node.point
      node.point = None
      
      newIndex = 0
      newDiff = 0
      midPoint = 0
      upPoint = other
      downPoint = point
      stats.stop("separate.prepare")
      stats.start("separate.separable")
      pointrep = point.separable(config)
      otherrep = other.separable(config)
      try:
         pointrep = list(pointrep)
         otherrep = list(otherrep)
      
         while len(pointrep) < len(otherrep):
            pointrep.append(0)
         while len(pointrep) > len(otherrep):
            otherrep.append(0)
      except:
         pass
      #print "separating: "
      #print pointrep
      #print otherrep
      longest = getattr(config, 'partitionLongest', True)
      if longest:
         for i in xrange(len(pointrep)):
            diff = upper[i] - lower[i]
            if diff > newDiff:
               newDiff = diff
               newIndex = i
               if pointrep[i] > otherrep[i]:
                  upPoint = point
                  downPoint = other
                  midPoint = otherrep[i] \
                   + (pointrep[i] - otherrep[i])/2.
               else:
                  upPoint = other
                  downPoint = point
                  midPoint = pointrep[i] \
                   + (otherrep[i] - pointrep[i])/2.
               if pointrep.__class__.__name__ == 'BayesNet':
                  break
      else:
         for i in xrange(len(pointrep)):
            if lower[i] == upper[i]:
               continue
            diff = abs(pointrep[i] - otherrep[i])
            if diff > newDiff:
               newDiff = diff
               newIndex = i
               if pointrep[i] > otherrep[i]:
                  upPoint = point
                  downPoint = other
                  midPoint = otherrep[i] \
                   + (pointrep[i] - otherrep[i])/2.
               else:
                  upPoint = other
                  downPoint = point
                  midPoint = pointrep[i] \
                   + (otherrep[i] - pointrep[i])/2.
               if pointrep.__class__.__name__ == 'BayesNet':
                  break
      
      stats.stop("separate.separable")
      stats.start("separate.compute")
      
      #print "got diff"
      
      if newDiff == 0.0:
        node.point = other
        raise SeparationException, "No difference in points"
      
      if config.bounded:
         proportion = (midPoint - lower[newIndex]) \
          / (upper[newIndex] - lower[newIndex])
      else:
         low = self.gaussInt(lower[newIndex]/config.spaceScale) 
         num = self.gaussInt(midPoint/config.spaceScale) - low
         denom = self.gaussInt(upper[newIndex]/config.spaceScale) - low
         proportion = num / denom
      
      upArea = float(node.area * (1.0 - proportion))
      downArea = float(node.area * proportion)
      
      if upArea != upArea or downArea != downArea:
         raise SeparationException, "NaN area!"
      
      upUpper = upper.copy()
      upLower = lower.copy()
      upLower[newIndex] = midPoint
      
      downUpper = upper.copy()
      downLower = lower.copy()
      downUpper[newIndex] = midPoint
      
      n1 = PartitionTreeNode(
         upper=float(upper[newIndex]), 
         lower=float(midPoint),
         segment=node.segment,
         point=upPoint,
         area = upArea,
         bounds = (upLower, upUpper),
         index = newIndex,
         best_score = upPoint.score,
         score_sum = upPoint.score,
         parent = node,
         layer = 0,
         layerrep = ''
      )
    
      n2 = PartitionTreeNode(
         upper=float(midPoint), 
         lower=float(lower[newIndex]),
         segment=node.segment,
         point=downPoint,
         area = downArea,
         bounds = (downLower, downUpper),
         index = newIndex,
         best_score = downPoint.score,
         score_sum = downPoint.score,
         parent = node,
         layer = 0,
         layerrep = ''
      )

      node.children = [n1,n2]
      upPoint.partition_node = n1
      downPoint.partition_node = n2
         
      # update the best score
      if path is not None:
         if point.score > other.score:
            for n in path:
               n.best_score = max([point.score,n.best_score])
      
      stats.stop("separate.compute")
      stats.stop("separate.main")
      
      stats.start("separate.areaTree")
      
      try:
         self.areaTree.insert(n1)
         self.areaTree.insert(n2)
         self.areaTree.remove(node)
      except:
         traceback.print_exc()
         raise
      
      stats.stop("separate.areaTree")
      stats.start("separate.propagate")
      
      # correct area in score tree
      if point.score_node is not None:
         sn = point.score_node
         if other == downPoint:
            diff = sn.area - downArea
         else:
            diff = sn.area - upArea
         sn.segment.scoreTree.propagateAreaOnly(sn, config, diff)
      
      #print "end separate"
      stats.stop("separate.propagate")
      

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
         node = self.scoreTree
      print indent, self.id, ": ", node.layer, node.index, node.lower, node.upper, node.area
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

   def __init__(self, segment, parent=None, point=None, area=1.0, min_score=-1e300, max_score=1e300, child_count=0, taylor=None, height=0, balance=0, left=False):
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
   segment = None
   
   @classmethod
   def _objects(cls):
      return cls.segment.scoreTree   

   @classmethod
   def _blank(cls):
      pass

   objects = ClassProperty(_objects,_blank)

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
      
      print indent, node.id, ',', node.balance, ",", node.min_score, " - ", node.max_score
      
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
         score = point.score * lr 
         taylor = nan_to_num(score ** ns) / fs
         taylor *= node.area
         taylor *= nan_to_num(exp(score) ** center) 
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
      
      score1 = upPoint.score * lr    
      score2 = downPoint.score * lr    
      taylor1 = (score1 ** ns) / fs
      taylor1 *= upArea
      taylor1 *= (exp(score1) ** center) 
      taylor2 = (score2 ** ns) / fs
      taylor2 *= downArea
      taylor2 *= (exp(score2) ** center)
      
      if config.selection == "proportional":
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
         
         This should be done whenever temp gets more than 0.5 away from the taylor center for accuracy.
         
         Requires a loop through all points, but only needs to be done at a logarithmically often.
         
         segment - the segment to recompute
         temp - the new temperature center
         config - an evo.config.Config object with "shiftToDb", "taylorCenter", "taylorDepth"
         
         This method sets config.taylorCenter to temp when complete.
      """
      log.info("resetTaylor: segment %s, new center %s, old center %s" % (segment.id, temp, config.taylorCenter))
      
      ns = [i+0. for i in xrange(config.taylorDepth)]
      fs = ns * 1
      if len(fs) > 0:
         fs[0] = 1.0
         for i in xrange(config.taylorDepth - 1):
            fs[i+1] *= fs[i]
      ns = array(ns)
      fs = array(fs)
      center = config.taylorCenter 
      lr = config.learningRate
      
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
               score = node.point.score
               score = score * lr
               taylor = nan_to_num(score ** ns) / fs
               taylor *= node.area
               taylor *= nan_to_num(exp(score) ** center) 
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
    segment=None
    
    @classmethod
    def _objects(cls):
        return cls.segment   

    @classmethod
    def _blank(cls):
        pass

    objects = ClassProperty(_objects,_blank) 

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
            assert self.root.val is not Nones
            return
            
        parent = self.traverse(partitionNode.area)
        
        mid = max([parent.val.area, partitionNode.area])
        
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
        print indent, node.id, node.low, node.high, node.val is None and "None" or node.val.id
        children = node.children
        for child in children:
            self.printTree(child, indent + "\t") 
   