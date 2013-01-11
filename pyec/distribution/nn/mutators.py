import numpy as np
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
        layerMap = {}
        for layer in net1.layers:
            layerMap[layer.id] = [layer, None]
        for layer in net2.layers:
            if layer.id in layerMap:
                layerMap[layer.id][1] = layer
            else:
                layerMap[layer.id] = [None, layer, None]
        #delete = [id for id, layers in layerMap.iteritems() if layers[1] is None]
        #for id in delete:
        #    del layerMap[id]
        
        net = self.config.space.copy(net1)
        if len(layerMap) == 0:
            return net
        
        #print "crossing over ..."
        #TODO handle layers of same id but different sizes,
        #possibly give nodes ids within layers
        for layer in net.layers:
            if layer.id in layerMap:
               layerMap[layer.id].append(layer)
               
        for id, layers in layerMap.iteritems():
            layer1, layer2, layer3 = layers
            if layer1 is None:
                if np.random.random_sample() < 0.5:
                    layer3 = layer2.__class__(layer.size, layer.activator)
                    layer3.id = layer2.id
                    net.addLayer(layer3)
                    layerMap[id] = (layer1, layer2, layer3)
            elif layer2 is None:
                if np.random.random_sample() < 0.5:
                    net.removeLayer(layer3)
                    layerMap[id] = (layer1, layer2, None)
        ids = dict([(layer.id,layer) for layer in net.layers])
        for id, layers in layerMap.iteritems():
            layer1, layer2, layer3 = layers
            if layer3 is not None:
                if layer1 is None:
                    inLinks = [frm.id for frm in layer2.inLinks if frm.id in ids]
                    for id2 in inLinks:
                        w = net2.links[(layerMap[id2][1], layer2)]
                        net.connect(ids[id2], layer3, w)
                elif layer2 is not None:
                    # layer1 and layer2 are aligned, layer3 is the target in the new net
                    inLinks1 = dict([(frm.id,None)
                                     for frm in layer1.inLinks
                                     if frm.id in ids])
                    inLinks2 = dict([(frm.id, None)
                                     for frm in layer2.inLinks
                                     if frm.id in ids])
                    for layer in net.layers:
                        if layer.id in inLinks1 and layer.id in inLinks2:
                            w1 = net1.links[(layerMap[layer.id][0], layer1)]
                            w2 = net2.links[(layerMap[layer.id][1], layer2)]
                            w = self.crossLink(w1, w2)
                            net.connect(layer, layer3, w)
                        #elif layer.id in inLinks1: # do nothing
                        elif layer.id in inLinks2:
                            w = net2.links[(layerMap[layer.id][1],layer2)]
                            net.connect(layer, layer3, w.copy())
        
        #print net1.links
        #print net2.links
        #print net.links
        net.changed = True
        return net


class UniformRnnLinkCrosser(UniformRnnCrosser):
    """Crossover for RNNs with entire matrices chosen from one RNN or the other.
    
    """
    def crossLink(self, w1, w2):
        """Just picks one of the two links; no crossover within matrices."""
        if np.random.random_sample() < .5:
            return w1.copy()
        return w2.copy()

 
class AddNodeMutation(Mutation):
   """Add a node to a neural network.
   
   Config parameters:
      
      * node_creation_prob - The probability of adding a node
      * node_sd - The standard deviation for the Gaussian used to make weights
                  for any new nodes
   
   """
   config = Config(node_creation_prob=.01,
                   node_sd=1.0)
   
   def mutate(self, net):
      if net.changed:
         return net
      p = self.config.node_creation_prob
      if np.random.random_sample() > p:
         return net
      
      sd = self.config.node_sd
      hiddenLayers = net.hiddenLayers()
      if len(hiddenLayers) == 0:
         return net
      
      
      net = self.config.space.copy(net)
      layer = net.hiddenLayers()[np.random.randint(0,len(hiddenLayers))]
      layer.size += 1
      for inLayer in layer.inLinks:
          w0 = net.links[(inLayer, layer)]
          w = np.zeros((layer.size, inLayer.size))
          w[:(layer.size-1),:] = w0
          w[layer.size-1,:] = sd * np.random.randn(inLayer.size)
          net.connect(inLayer, layer, w)
      
      for outLayer in layer.outLinks:
          w0 = net.links[(layer, outLayer)]
          w = np.zeros((outLayer.size, layer.size))
          w[:,:(layer.size-1)] = w0
          w[:,layer.size-1] = sd * np.random.randn(outLayer.size)
          net.connect(layer, outLayer, w)
      
      net.changed = True
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
      indexToRem = np.random.randint(0, layer.size)
      if layer.size <= 1:
         return net
      
      layer.size -= 1
      
      for inLayer in layer.inLinks:
          w0 = net.links[(inLayer, layer)]
          w = np.delete(w0, indexToRem, axis=0)
          net.connect(inLayer, layer, w)
      
      for outLayer in layer.outLinks:
          w0 = net.links[(layer, outLayer)]
          w = np.delete(w0, indexToRem, axis=1)
          net.connect(layer, outLayer, w)
      
      net.changed = True
      return net


class AddLinkMutation(Mutation):
   """Add a link to a network.
   
   Config parameters:
   
      * link_creation_prob - The probability of adding a random link
      * link_sd - The std dev for the Gaussian used to create the weights
   
   """
   config = Config(link_creation_prob=0.025,
                   link_sd=10.0)
   
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
         w = self.config.link_sd * np.random.randn(target.size, source.size)
         net.connect(source, target, w)
         net.changed = True
      
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
      return net

class AddChainLayerMutation(Mutation):
    """Add a layer to an RNN copied from another layer,
    with a link going from the copied layer to the new layer
    whose weights are the identity matrix, and a
    link going to a random layer with random weights.
   
    Config parameters:
   
      * layer_creation_prob - The probability of adding a layer
      * layer_sd - The std dev for the Gaussian to generate random weights
      
    """
    config = Config(layer_creation_prob=.01,
                    layer_sd=10.0)
   
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
        w = self.config.layer_sd * np.random.randn(target.size, source.size)
        net.connect(middle, target, w)
        net.changed = True
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
        return net


class GaussianWeightMutation(WeightMutation):
    config = Config(link_mutation_prob = 0.33,
                    sd=1.0)

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
    config = Config(decay=lambda n,cfg: (1./n))
    
    def sd(self, net):
        if net.partition_node:
            area = net.partition_node
            lower, upper = area.bounds.extent()
            sd = .5 * (upper - lower)
            sd = np.minimum(sd, self.config.space.scale)
            sd *= self.config.decay(self.history.updates, self.config)
        else:
            n = self.history.updates
            return self.config.sd * self.config.decay(n, self.config)


class NetAreaStripper(Mutation):
    def mutate(self, x):
        point, node = x
        net = self.config.space.copy(point.point)
        net.partition_node = node
        net.changed = False
        return net
