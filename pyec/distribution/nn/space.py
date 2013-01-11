"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
from pyec.distribution.nn.genotypes import LayeredRnnGenotype, SIGMOID
from pyec.space import Space, LayeredSpace, LayerWrapper, Euclidean
from scipy.misc import comb


class NetLayerWrapper(LayerWrapper):
    def unwrap(self, net):
        return net.weights()


class LayeredRnnSpace(LayeredSpace):
    """A space of layered neural networks. Networks in this space may have an
    arbitrary topology but a fixed number of inputs and outputs.
    
    :param inputs: A list containing the size of each input layer
    :type inputs: A list (or tuple) of ``int``
    :param outputs: A list containing the size of each output layer
    :type outputs: A list (or tuple) of ``int``
    
    """
    def __init__(self, inputs, outputs, bias=True, activator=SIGMOID, scale=10.0):
        super(LayeredRnnSpace, self).__init__(LayeredRnnGenotype)
        self.inputs = inputs
        self.outputs = outputs
        self.bias = bias
        self.activator = activator
        self.scale = scale
   
    def random(self):
        return self.type.random(self.inputs, self.outputs,
                                self.bias, activator=self.activator)
    
    def hash(self, x):
        point = x.weights()
        parts = [((point+i)**2).sum() for i in np.arange(10)]
        return ",".join([str(pt) for pt in parts])
      
    def convert(self, x):
        return x.compile()
      
    def area(self, **kwargs):
        return 1.0
      
    def in_bounds(self, x, **kwargs):
        return super(LayeredRnnSpace, self).in_bounds(x)
      
    def extractLayers(self, x):
        hiddenLayers = x.hiddenLayers()
        sizes = [h.size for h in hiddenLayers]
        numLayers = len(sizes)
        connections = x.connectivity()
        numWeights = len(x.weights())
        kwargs = {"inputs":self.inputs,
                  "outputs": self.outputs,
                  "bias": self.bias,
                  "activator": self.activator,
                  "scale": self.scale}
        return [self,
                LayeredRnnSpaceFixedLayers(numLayers=numLayers, **kwargs),
                LayeredRnnSpaceFixedSizes(layers=sizes, **kwargs),
                LayeredRnnSpaceFixedConnectivity(layers=sizes, connections=connections, **kwargs),
                NetLayerWrapper(Euclidean(dim=numWeights,
                                          center=0,
                                          scale=self.scale))]
      
    def wrapLayer(self, region):
        return NetLayerWrapper(region)
      
    def layers(self, net):
        hiddenLayers = net.hiddenLayers()
        sizes = [h.size for h in hiddenLayers]
        connections = net.connectivity()
        weights = net.weights()
        return [None, len(hiddenLayers), sizes, connections, weights]


class LayeredRnnSpaceFixedLayers(LayeredRnnSpace):
    """A subspace of layered RNNs with a fixed number of hidden layers.
    
    :param inputs: A list containing the size of each input layer
    :type inputs: A list (or tuple) of ``int``
    :param outputs: A list containing the size of each output layer
    :type outputs: A list (or tuple) of ``int``
    :param layers: A tuple of two integers, the lower and upper bounds for the
                   number of hidden layers for networks in this space
    :type layers: A tuple with two ``int``s
    
    """
    def __init__(self, inputs, outputs, numLayers,
                 bias=True, activator=SIGMOID, scale=10.0):
        super(LayeredRnnSpaceFixedLayers, self).__init__(inputs,
                                                         outputs,
                                                         bias=bias,
                                                         activator=activator,
                                                         scale=scale)
        self.numLayers = numLayers
        self._area = None
        
    def random(self):
        hiddenSizes = 1 + np.random.poisson(self.numLayers)
        return self.type.random(self.inputs, self.outputs, 
                                self.bias, hiddenSizes, activator=self.activator)
      
    def in_bounds(self, x, **kwargs):
        return (super(LayeredRnnSpaceFixedLayers, self).in_bounds(x, **kwargs)
                and self.numLayers == len(x.hiddenLayers())) 
    
    def area(self, **kwargs):
        return 1.0
      
    def layerFactor(self):
        return 1. / (1. + self.numLayers)
      
class LayeredRnnSpaceFixedSizes(LayeredRnnSpaceFixedLayers):
    """A subspace of layered RNNs with fixed layer sizes
    
    :param inputs: A list containing the size of each input layer
    :type inputs: A list (or tuple) of ``int``
    :param outputs: A list containing the size of each output layer
    :type outputs: A list (or tuple) of ``int``
    :param layers: A list or tuple of ``int`` indicating the size of each
                   hidden layer
    :type layers: A list (or tuple) of ``int``
    
    """
    def __init__(self, inputs, outputs, layers,
                 bias=True, activator=SIGMOID, scale=10.0):
        numLayers = len(layers)
        super(LayeredRnnSpaceFixedSizes, self).__init__(inputs,
                                                        outputs,
                                                        numLayers,
                                                        bias=bias,
                                                        activator=activator,
                                                        scale=scale)
        self.layers = np.array(layers, dtype=int)
        
    def random(self):
        return self.type.random(self.inputs, self.outputs, 
                                self.bias, self.layers, activator=self.activator)
      
    def in_bounds(self, x, **kwargs):
        return (super(LayeredRnnSpaceFixedSizes, self).in_bounds(x, **kwargs)
                and np.all([l == h.size
                            for l,h in zip(self.layers, x.hiddenLayers())]))
    
    def area(self, **kwargs):
        return 1.0

    def layerFactor(self):
        return self.layers.sum()


class LayeredRnnSpaceFixedConnectivity(LayeredRnnSpaceFixedSizes):
    """A subspace of layered RNNs with fixed layer sizes and fixed
    connectivity.
    
    :param inputs: A list containing the size of each input layer
    :type inputs: A list (or tuple) of ``int``
    :param outputs: A list containing the size of each output layer
    :type outputs: A list (or tuple) of ``int``
    :param layers: A list or tuple of ``int`` indicating the size of each
                   hidden layer
    :type layers: A list (or tuple) of ``int``
    :param connections: A matrix of connections describing the connectivity
                        of a neural network. The total number of layers is
                        the number of input layers + the bias (if present) +
                        the number of output layers + the number of hidden
                        layers. The connectivity is a square matrix with
                        boolean entry. If the $(i,j)$ entry is true, then the
                        $i^{th}$ layer is connected to the $j^{th}$ layer,
                        with the orientation of the link running from $i$ to
                        $j$. In this case, the layers are indexed according
                        to the scheme above: first input layers (``len(inputs)``)
                        then the bias (if any), then the output layers
                        (``len(outputs)``), then the hidden layers
                        (``len(layers)``).
    :type connections: A ``numpy.ndarray`` with ``dtype`` ``bool`` and shape
                       ``(layers,layers)``
    
    """
    def __init__(self, inputs, outputs, layers, connections,
                 bias=True, activator=SIGMOID, scale=10.0):
        super(LayeredRnnSpaceFixedConnectivity, self).__init__(inputs,
                                                        outputs,
                                                        layers,
                                                        bias=bias,
                                                        activator=activator,
                                                        scale=scale)
        self.connections = connections
        
    def random(self):
        return self.type.random(self.inputs, self.outputs, 
                                self.bias, self.layers,
                                self.connections, activator=self.activator)
      
    def in_bounds(self, x, **kwargs):
        return (super(LayeredRnnSpaceFixedConnectivity, self).in_bounds(x,
                                                                        **kwargs)
                and np.all(x.connectivity() == self.connections))
    
    def area(self, **kwargs):
        return 1.0
    
    def layerFactor(self):
        totalLinks = len(self.connections)
        possible = len(self.layers) ** 2
        return comb(possible, totalLinks) / float(totalLinks)
