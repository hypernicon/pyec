"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
from scipy.special import erf

from pyec.config import Config
from pyec.distribution.ec.mutators import Crosser, Mutation
from .genotypes import *

class UniformRnnCrosser(Crosser):
    """Uniform crossover for neural networks. Layers are aligned by id, and
    any common links are subjected to uniform crossover of their matrices.
    The first network (the father) dominates in the case of lacunae.
    
    """
    def crossLink(self, w1, w2):
        """Crossover a weight matrix. Default is to uniformly choose elements
        from each.
        
        :param w1: A weight matrix
        :type w1: ``numpy.ndarray``
        :param w2: A weight matrix of the same shape
        :type w2: ``numpy.ndarray``
        :returns: A new ``numpy.ndarray`` with the same shape as ``w1``
        
        """
        mask = np.random.binomial(1, .5, np.shape(w1))
        return mask * w1 + (1-mask) * w2
    
    def __call__(self, orgs, prob):
        if np.random.random_sample() > prob:
            return orgs[0]
        
        net1, net2 = orgs
        net = self.config.space.copy(net1)
        
        # rectify the sizes
        for i in enumerate(net.layers):
            if i < len(net2.layers):
                layer = net.layers[i]
                layer2 = net2.layers[i]
                if layer.size > layer2.size:
                    toRem = np.random.random_integers(0,layer.size - layer2.size)
                    layer.size -= toRem
                    for source in layer.inLinks:
                        w = net.links[(source, layer)]
                        net.connect(source, layer, w[:-(toRem+1)])
                    for target in layer.outLinks:
                        w = net.links[(layer, target)]
                        net.connect(layer, target, w[:,:-(toRem+1)])
                elif layer.size < layer2.size:
                    diff = layer2.size - layer.size
                    toAdd = np.random.random_integers(0, diff)
                    layer.size += toAdd
                    for source in layer.inLinks:
                        w = net.links[(source, layer)]
                        w = np.append(w, np.zeros((toAdd, source.size)), axis=0)
                        net.connect(source, layer, w)
                    for target in layer.outLinks:
                        w = net.links[(layer, target)]
                        w = np.append(w, np.zeros((target.size, toAdd)), axis=1)
                        net.connect(layer, target, w)
        
        if len(net.layers) > len(net2.layers):
            diff = len(net.layers) - len(net2.layers)
            toRem = np.random.random_integers(0,diff)
            for i in xrange(toRem):
                net.removeLayer(net.layers[-1])
        elif len(net2.layers) > len(net.layers):
            diff = len(net2.layers) - len(net.layers)
            toAdd = np.random.random_integers(0,diff)
            for i in xrange(toAdd):
                layer2 = net2.layers[-diff+i]
                layer = layer2.__class__(size=layer2.size, activator=layer2.activator)
                layer.id = layer2.id
                net.addLayer(layer)
                for target2 in layer2.outLinks:
                    targetIdx = net2.layers.index(target2)
                    if targetIdx < len(net.layers):
                        target = net.layers[targetIdx]
                        w = net2.links[(layer2, target2)].copy()
                        sw = np.shape(w)
                        if sw != (target.size, layer.size):
                            if target.size < sw[0]:
                                w = w[:target.size]
                            elif target.size > sw[0]:
                                w = np.append(w, np.zeros((target.size - sw[0], layer.size)), axis=0)
                            if layer.size < sw[1]:
                                w = w[:,:layer.size]
                            elif layer.size > sw[1]:
                                w = np.append(w, np.zeros((target.size, layer.size - sw[1])), axis=1)
                        net.connect(layer, target, w)
                for source2 in layer2.inLinks:
                    sourceIdx = net2.layers.index(source2)
                    if sourceIdx < len(net.layers):
                        source = net.layers[sourceIdx]
                        w = net2.links[(source2, layer2)].copy()
                        sw = np.shape(w)
                        if sw != (layer.size, source.size):
                            if layer.size < sw[0]:
                                w = w[:layer.size]
                            elif layer.size > sw[0]:
                                w = np.append(w, np.zeros((layer.size - sw[0], source.size)), axis=0)
                            if source.size < sw[1]:
                                w = w[:,:source.size]
                            elif source.size > sw[1]:
                                w = np.append(w, np.zeros((layer.size, source.size - sw[1])), axis=1)
                        net.connect(source, layer, w)
                net.changed = True
        
        
        for i,layer in enumerate(net.layers):
            if i >= len(net1.layers) or i >= len(net2.layers):
                break
            for j,layer in enumerate(net.layers):
                if j >= len(net1.layers) or j >= len(net2.layers):
                    break
                c = (net.layers[i],net.layers[j])
                if c in net.links:
                    c2 = (net2.layers[i],net2.layers[j])
                    if c2 in net2.links:
                        w1 = net.links[c]
                        w2 = net2.links[c2]
                        s1 = np.shape(w1)
                        s2 = np.shape(w2)
                        if s1 != s2:
                            if s1[0] > s2[0]:
                                w2 = np.append(w2, np.zeros((s1[0]-s2[0], s2[1])), axis=0)
                            if s1[0] < s2[0]:
                                w2 = w2[:s1[0]]
                            if s1[1] > s2[1]:
                                w2 = np.append(w2, np.zeros((np.shape(w2)[0],s1[1]-s2[1])), axis=1)
                            if s1[1] < s2[1]:
                                w2 = w2[:,:s1[1]]
                        w = self.crossLink(w1, w2)
                        net.connect(net.layers[i], net.layers[j], w)
                        net.changed = True
        #net.checkLinks()
        return net


class UniformRnnLinkCrosser(UniformRnnCrosser):
    """Crossover for RNNs with entire matrices chosen from one RNN or the other.
    
    """
    def crossLink(self, w1, w2):
        """Just picks one of the two links; no crossover within matrices."""
        if np.random.random_sample() < .5:
            return w1.copy()
        return w2.copy()


class IntermediateRnnCrosser(UniformRnnCrosser):
    """Crossover for RNNs with matrices averaged
    
    """
    def crossLink(self, w1, w2):
        """Just picks one of the two links; no crossover within matrices."""
        return .5 * (w1 + w2)
 

class UniformOrIntermediateRnnCrosser(UniformRnnCrosser):
    def __init__(self, *args, **kwargs):
        super(UniformOrIntermediateRnnCrosser, self).__init__(*args, **kwargs)
        self._intermediate = np.random.random_sample() < .4
        
    def intermediate(self, w1, w2):
        return .5 * (w1+w2)
    
    def crossLink(self, w1, w2):
        if self._intermediate:
            return self.intermediate(w1, w2)
        else:
            return super(UniformOrIntermediateRnnCrosser, self).crossLink(w1, w2)

 
class AddNodeMutation(Mutation):
   """Add a node to a neural network.
   
   Config parameters:
      
      * node_creation_prob - The probability of adding a node
      * sd - The standard deviation for the Gaussian used to make weights
                  for any new nodes
   
   """
   config = Config(node_creation_prob=.01,
                   sd=0.1)
   
   def mutate(self, net):
      if net.changed:
         return net
      p = self.config.node_creation_prob
      if np.random.random_sample() > p:
         return net
      
      sd = self.config.sd
      hiddenLayers = net.hiddenLayers()
      if len(hiddenLayers) == 0:
         return net
      
      
      net = self.config.space.copy(net)
      layer = net.hiddenLayers()[np.random.randint(0,len(hiddenLayers))]
      layer.size += 1
      for inLayer in layer.inLinks:
          w0 = net.links[(inLayer, layer)]
          w = np.zeros((layer.size, inLayer.size))
          if inLayer == layer:
              w[:(layer.size-1),:(layer.size-1)] = w0
              w[:,layer.size-1] = sd * np.random.randn(layer.size)
          else:
              w[:(layer.size-1),:] = w0
          w[layer.size-1,:] = sd * np.random.randn(inLayer.size)
          net.connect(inLayer, layer, w)
      
      for outLayer in layer.outLinks:
          if outLayer == layer:
              continue # we already did this!
          w0 = net.links[(layer, outLayer)]
          w = np.zeros((outLayer.size, layer.size))
          w[:,:(layer.size-1)] = w0
          w[:,layer.size-1] = sd * np.random.randn(outLayer.size)
          net.connect(layer, outLayer, w)
      
      net.changed = True
      #net.checkLinks()
      return net
 

class RemoveNodeMutation(Mutation):
   """Delete a node from a neural network.
   
   Config parameters:
      
      * node_deletion_prob - The probability of deleting a node
   
   """
   config = Config(node_deletion_prob=.01)
   
   def mutate(self, net):
      if net.changed:
         return net
      p = self.config.node_deletion_prob
      if np.random.random_sample() > p:
         return net
      
      hiddenLayers = net.hiddenLayers()
      if len(hiddenLayers) == 0:
         return net
      
      net = self.config.space.copy(net)
      
      layer = net.hiddenLayers()[np.random.randint(0,len(hiddenLayers))]
      if layer.size <= 1:
         return net
      
      indexToRem = np.random.randint(0, layer.size)
      layer.size -= 1
      
      for inLayer in layer.inLinks:
          w0 = net.links[(inLayer, layer)]
          w = np.delete(w0, indexToRem, axis=0)
          if inLayer == layer:
              w = np.delete(w, indexToRem, axis=1)
          net.connect(inLayer, layer, w)
      
      for outLayer in layer.outLinks:
          if outLayer == layer:
              continue
          w0 = net.links[(layer, outLayer)]
          w = np.delete(w0, indexToRem, axis=1)
          net.connect(layer, outLayer, w)
      
      net.changed = True
      #net.checkLinks()
      return net


class AddLinkMutation(Mutation):
   """Add a link to a network.
   
   Config parameters:
   
      * link_creation_prob - The probability of adding a random link
      * sd - The std dev for the Gaussian used to create the weights
   
   """
   config = Config(link_creation_prob=0.025,
                   sd=0.1)
   
   def mutate(self, net):
      if net.changed:
         return net
      
      p = self.config.link_creation_prob
      if np.random.random_sample() > p:
         return net
      
      # randomly pick two layers
      satisfactory = lambda lay: (not isinstance(lay, RnnLayerInput) and
                                  not isinstance(lay, RnnLayerBias))
      targets = [layer for layer in net.layers if satisfactory(layer)]
      target = targets[np.random.randint(0,len(targets))]
      source = net.layers[np.random.randint(0,len(targets))]
      if not (source, target) in net.links:
         net = self.config.space.copy(net)
         source = [layer for layer in net.layers if layer.id == source.id][0]
         target = [layer for layer in net.layers if layer.id == target.id][0]
         w = self.config.sd * np.random.randn(target.size, source.size)
         net.connect(source, target, w)
         net.changed = True
         #net.checkLinks()
      
      return net


class RemoveLinkMutation(Mutation):
   """Remove a random link from a network.
   
   Config parameters:
   
      * link_deletion_prob - The probability of removing a random link
   
   """
   config = Config(link_deletion_prob=0.025)
   
   def mutate(self, net):
      if net.changed:
         return net
      
      p = self.config.link_deletion_prob
      if np.random.random_sample() > p:
         return net
      
      # randomly pick two layers
      satisfactory = lambda lay: (not isinstance(lay, RnnLayerInput) and
                                  not isinstance(lay, RnnLayerBias))
      targets = [layer for layer in net.layers if satisfactory(layer)]
      target = targets[np.random.randint(0,len(targets))]
      source = net.layers[np.random.randint(0,len(targets))]
      if (source, target) in net.links:
         net = self.config.space.copy(net)
         source = [layer for layer in net.layers if layer.id == source.id][0]
         target = [layer for layer in net.layers if layer.id == target.id][0]
         net.disconnect(source, target)
         net.changed = True
         #net.checkLinks()
      return net


class AddChainLayerMutation(Mutation):
    """Add a layer to an RNN copied from another layer,
    with a link going from the copied layer to the new layer
    whose weights are the identity matrix, and a
    link going to a random layer with random weights.
   
    Config parameters:
   
      * layer_creation_prob - The probability of adding a layer
      * sd - The std dev for the Gaussian to generate random weights
      
    """
    config = Config(layer_creation_prob=.01,
                    sd=0.1)
   
    def mutate(self, net):
        if net.changed:
            return net
      
        if np.random.random_sample() > self.config.layer_creation_prob:
            return net
      
        net = self.config.space.copy(net)
        satisfactory = lambda lay: (not isinstance(lay, RnnLayerInput) and
                                    not isinstance(lay, RnnLayerBias))
        targets = [layer for layer in net.layers if satisfactory(layer)]
        target = targets[np.random.randint(0,len(targets))]
        source = net.layers[np.random.randint(0,len(targets))]
        middle = RnnLayer(source.size,activator=self.config.space.activator)
        net.addLayer(middle)
        net.connect(source, middle, np.identity(source.size))
        #if (source, target) in net.links:
        #    net.connect(middle, target, net.links[(source, target)].copy())
        #else:
        w = self.config.sd * np.random.randn(target.size, middle.size)
        net.connect(middle, target, w)
        net.changed = True
        #net.checkLinks()
        return net


class RemoveLayerMutation(Mutation):
    """Remove a random hidden layer.
   
    Config parameters:
   
      * layer_deletion_prob - The probability of removing a layer
   
    """
    config = Config(layer_deletion_prob=0.01)
   
    def mutate(self, net):
        if net.changed:
            return net
        if np.random.random_sample() > self.config.layer_deletion_prob:
            return net
        net = self.config.space.copy(net)
        hiddenLayers = net.hiddenLayers()
        if len(hiddenLayers) == 0:
            return net
        toRemove = hiddenLayers[np.random.randint(0,len(hiddenLayers))]
        net.removeLayer(toRemove)
        net.changed = True
        #net.checkLinks()
        return net


class WeightMutation(Mutation):
    def mutateWeightMatrix(self, net, weights):
        return weights

    def mutate(self, net):
        if net.changed:
            return net
        net = self.config.space.copy(net)
        for edge, weights in net.links.iteritems():
            if not isinstance(edge[1], RnnLayerInput):
               net.connect(edge[0], edge[1], self.mutateWeightMatrix(net, weights))
        net.changed = True
        #net.checkLinks()
        return net


class GaussianWeightMutation(WeightMutation):
    config = Config(link_mutation_prob = 0.5,
                    sd=.1)

    def sd(self, net):
        return self.config.sd

    def mutateWeightMatrix(self, net, weights):
        sd = self.sd(net)
        mask = np.random.binomial(1,
                                  self.config.link_mutation_prob,
                                  np.shape(weights))
        return weights + mask * np.random.randn(*np.shape(weights))


class AreaSensitiveGaussianWeightMutation(GaussianWeightMutation):
    """Weight mutation, sensitive to area partition.
    
    Config parameters:
      * decay -- A function ``decay(n,config)`` to compute the multiplier
                 that controls the rate of decrease in standard deviation.
                 Faster decay causes faster convergence, but may miss the
                 optimum. Default is ``((1/generations))``
    
    """
    config = Config(decay=lambda n,cfg: (1./(2*np.log(n))))
    
    def gaussInt(self, z):
        # x is std normal from zero to abs(z)
        x = .5 * erf(np.abs(z)/np.sqrt(2))
        return .5 + np.sign(z) * x
    
    def sd(self, net):
        try:
            area, path = self.config.segment.partitionTree.traverse(net)
            lower, upper = area.bounds.extent()
            sd = .5 * (upper - lower)
            scale = self.config.space.scale
            sd = self.gaussInt(upper/scale) - self.gaussInt(lower/scale)
            sd *= self.config.decay(self.history.updates, self.config)
        except Exception:
            n = self.history.updates
            return self.config.sd * self.config.decay(n, self.config)


class NetAreaStripper(Mutation):
    def mutate(self, x):
        point, node = x
        net = self.config.space.copy(point.point)
        net.changed = False
        return net
