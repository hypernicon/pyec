"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import numpy as np

def IDENTITY(x):
    return x

def LOGISTIC(x):
    return 1. / (1. + np.exp(-x))

HYPERBOLIC = np.tanh

THRESHOLD = np.sign

def BIAS(x):
    return 1.0



class RnnLayer(object):
    """A representation of a layer in a neural network; that is, a group of
    artificial neurons with the same modality and activation.
    
    :param size: The number of neurons in the layer
    :type size: ``int``
    :param activator: The activator for this layer
    :type activator: A function that takes a ``numpy.ndarray`` of dtype ``float``
                     with length ``size`` and returns a similarly sized array
    
    """
    idgen = 1L
    
    def __init__(self, size, activator, **kwargs):
        self.size = size
        self.inLinks = []
        self.outLinks = []
        self.activator = activator
        self.id = self.__class__.idgen
        RnnLayer.idgen += 1
    
    def __str__(self):
        return self.__class__.__name__  + "_" + str(self.id)
    
    __repr__ = __str__
    
    def copyLinks(self, other, layerMap):
        self.inLinks = [layerMap[layer] for layer in other.inLinks]
        self.outLinks = [layerMap[layer] for layer in other.outLinks]

    def addSource(self, other):
        """Add a layer as an input to this layer.
        
        :param other: The layer to add
        :type other: :class:`RnnLayer`
        :throws: ``ValueError`` if ``other`` is already an input
        
        """
        if other in self.inLinks:
            raise ValueError("Cannot add in-link from layer twice!")
        self.inLinks.append(other)
    
    def removeSource(self, other):
        """Add a layer as an input to this layer.
        
        :param other: The layer to remove
        :type other: :class:`RnnLayer`
        
        """
        try:
            self.inLinks.remove(other)
        except ValueError:
            pass

    def addTarget(self, other):
        """Add a layer as an output of this layer.
        
        :param other: The layer to add
        :type other: :class:`RnnLayer`
        :throws: ``ValueError`` if ``other`` is already an output
        
        """
        if other in self.outLinks:
            raise ValueError("Cannot add out-link from layer twice!")
        self.outLinks.append(other)
        
    def removeTarget(self, other):
        """Remove a layer as an output of this layer.
        
        :param other: The layer to remove
        :type other: :class:`RnnLayer`
        
        """
        try:
            self.outLinks.remove(other)
        except:
            pass


class RnnLayerInput(RnnLayer):
    """An input layer for a neural network; values are assumed to be
    tied to some external source. Activator is identity
   
    """
    def __init__(self, size, **kwargs):
        super(RnnLayerInput, self).__init__(size, IDENTITY)


class RnnLayerOutput(RnnLayer):
    """An output layer for a neural network; values are assumed to be
    fed to some external source. Activator is variable, but is sigmoidal
    by default.
   
    """
    def __init__(self, size, activator=None, **kwargs):
        if activator is None:
            activator = LOGISTIC
        super(RnnLayerOutput, self).__init__(size, activator, **kwargs)


class RnnLayerBias(RnnLayer):
    """A bias node for a neural network. A layer of size 1 with an
    activation that outputs 1 and disallows addition of input nodes.
    
    """
    def __init__(self, **kwargs):
        super(RnnLayerBias, self).__init__(1, BIAS, **kwargs)
        
    def addSource(self):
        raise OperationNotSupported("Bias node does not allow input links")


class LayeredRnnGenotype(object):
    """A representation of an RNN, used for building RNNs. Does not have
    code to compute the activation values of the neural network. A computational
    network can be obtained by calling ``compile``.
      
    """
    def __init__(self):
        self.layers = []
        self.links = {}
        self._weights = None
        self.partition_node = None
        self.changed = False # used to cut off mutations later
    
    def addLayer(self, layer):
        """Add a layer to this neural network.
        
        :param layer: The layer to add.
        :type layer: :class:`RnnLayer`
        :throws: ``ValueError`` if ``layer`` is not in the network
        
        """
        if layer in self.layers:
           raise ValueError("Cannot add existing layer!")
        self.layers.append(layer)
    
    def removeLayer(self, layer):
        """Remove a layer from this neural network.
        
        :param layer: The layer to remove.
        :type layer: :class:`RnnLayer`
        :throws: ``ValueError`` if ``layer`` is not in the network
        
        """
        if layer not in self.layers:
            raise ValueError("Cannot remove layer not in network!")
        self.layers.remove(layer)
        for layer2 in self.layers:
            layer2.removeSource(layer)
            layer2.removeTarget(layer)
            if (layer, layer2) in self.links:
               del self.links[(layer, layer2)]
            if (layer2, layer) in self.links:
               del self.links[(layer2, layer)]
        self._weights = None
    
    def checkLinks(self):
        """Verify the shape of the weight matrices for the network"""
        for layer in self.layers:
            assert layer.size >= 1
        for edge, w in self.links.iteritems():
            assert edge[0] in self.layers
            assert edge[1] in self.layers
            assert np.shape(w) == (edge[1].size, edge[0].size)
    
    def connect(self, fromLayer, toLayer, weights):
        """Connect two layers with a link. If the two layers are already
        connected, then this method just changes the connection weights.
        
        :param fromLayer: The source of the link
        :type fromLayer: :class:`RnnLayer`
        :param toLayer: The target of the link
        :type toLayer: :class:`RnnLayer`
        :param weights: A ``numpy.ndarray`` with dimensions M x N, where N is the
                        size of ``toLayer`` and M is the size of ``fromLayer``
        
        """
        if fromLayer not in self.layers or toLayer not in self.layers:
            raise ValueError("Tried to connect layers not in this network")
        if np.shape(weights) != (toLayer.size, fromLayer.size):
            raise ValueError("Tried to connect layers of shape "
                             "{0} using weights shaped as {1}".format(np.shape(weights),
                                                                      (toLayer.size, fromLayer.size)))
        edge = (fromLayer, toLayer)
        if edge in self.links:
            self.links[edge] = weights
        else:
            toLayer.addSource(fromLayer)
            fromLayer.addTarget(toLayer)
            self.links[edge] = weights
        self._weights = None
            
    def disconnect(self, fromLayer, toLayer):
        """Disconnect two layers.
        
        :param fromLayer: The source of the link
        :type fromLayer: :class:`RnnLayer`
        :param toLayer: The target of the link
        :type toLayer: :class:`RnnLayer`
       
        """
        if (fromLayer, toLayer) in self.links:
            del self.links[(fromLayer, toLayer)]
            fromLayer.removeTarget(toLayer)
            toLayer.removeSource(fromLayer)
        self._weights = None
    
    def connectivity(self):
        """Return a list of connections.
        
           :returns: A list of tuples. Each tuple is ``(frm, to)`` where ``frm``
                     and ``to`` are the integer ids of the connected layers
        """
        connections = []
        for edge in self.links.keys():
            connections.append((self.layers.index(edge[0]), self.layers.index(edge[1])))
        return sorted(connections)
      
    def weights(self):
        """Return the weights as a 1d array. Internal, sort the links so
        that consistent outcomes are guaranteed.
        
        :returns: A 1d ``numpy.ndarray`` containing the weights.
        
        """
        if self._weights is not None:
            return self._weights
        weights = np.zeros(0, dtype=float)
        size = len(self.layers)
        key = lambda x: (self.layers.index(x[0][0]) +
                         size * self.layers.index(x[0][1]))
        for edge, w in sorted(self.links.items(), key=key):
            weights = np.append(weights, w.ravel(), axis=0)
        self._weights = weights
        return weights
      
    def compile(self):
        """Return a neural network object that has a state and can be
        run. Such an object must be able to set the inputs, read the outputs,
        and incrementally activate the network a specified number of times.
        
        :returns: A :class:`RnnEvaluator` that can be used to run the network
        
        """
        #try:
        #    from .cnet import RnnEvaluator
        #except:
        #    print "Fallback on pure python"
        from .net import RnnEvaluator
        slices = {}
        weightStack = []
        current = 0
        for layer in self.layers:
            next = current + layer.size
            slices[layer] = slice(current, next)
            current = next
            while len(layer.inLinks) > len(weightStack):
                weightStack.append([])
        numNeurons = current
            
        inputs = []
        outputs = []
        activationStack = []
        for layer in self.layers:
            idxs = slices[layer]
            if isinstance(layer, RnnLayerInput):
               inputs.append(idxs)
            if isinstance(layer, RnnLayerOutput):
               outputs.append(idxs)
            activationStack.append((idxs,layer.activator))
            for i, frmLayer in enumerate(layer.inLinks):
                w = self.links[(frmLayer, layer)]
                sh = (layer.size, frmLayer.size)
                if w.shape != sh:
                    raise ValueError("Weight matrix has wrong size; should be "
                                     "{0} but was {1}".format(w.shape, sh))
                weightStack[i].append((w,slices[frmLayer],idxs))
        
        return RnnEvaluator(numNeurons, inputs, outputs,
                            weightStack, activationStack)

    def hiddenLayers(self):
        hidden = lambda x: (not isinstance(x, RnnLayerInput) and
                            not isinstance(x, RnnLayerOutput) and
                            not isinstance(x, RnnLayerBias))
        return [layer for layer in self.layers if hidden(layer)]

    def __copy__(self):
        genotype = self.__class__()
        genotype.layers = [layer.__class__(size=layer.size,
                                           activator=layer.activator)
                           for layer in self.layers]
        layerMap = dict(zip(self.layers, genotype.layers))
        for layer, glayer in zip(self.layers, genotype.layers):
            glayer.id = layer.id
            glayer.copyLinks(layer, layerMap)
        for edge, w in self.links.iteritems():
            genotype.links[(layerMap[edge[0]],layerMap[edge[1]])] = w.copy()
        genotype._weights = self._weights.copy()
        return genotype

    @classmethod
    def random(cls, inputs, outputs, bias=True,
               hiddenSizes=None, connections=None, activator=LOGISTIC):
        genotype = cls()
        inputLayers = []
        idx = 0L
        for input in inputs:
            layer = RnnLayerInput(input)
            genotype.addLayer(layer)
            inputLayers.append(layer)
            genotype.connect(layer, layer, np.identity(input))
            layer.id = idx
            idx += 1
        if bias:
            bias = RnnLayerBias()
            genotype.addLayer(bias)
            bias.id = idx
            idx += 1
        outputLayers = []
        for output in outputs:
            layer = RnnLayerOutput(output, activator)
            genotype.addLayer(layer)
            outputLayers.append(layer)
            layer.id = idx
            idx += 1
        hiddenLayers = []
        if hiddenSizes is not None:
            for hidden in hiddenSizes:
                layer = RnnLayer(hidden, activator)
                genotype.addLayer(layer)
                hiddenLayers.append(layer)
        if connections is not None:
            for i,row in enumerate(connections):
                for j,col in enumerate(connections):
                    if col != 0:
                        frm = genotype.layers[i]
                        to = genotype.layers[j]
                        w = 0.1 * np.random.randn(to.size, frm.size)
                        genotype.connect(frm, to, w)
                        
        else:
            # connect inputs to every hidden, every hidden to output
            if bias:
                for layer in hiddenLayers:
                    genotype.connect(bias,layer,10.*np.random.randn(layer.size))
                for layer in outputLayers:
                    genotype.connect(bias,layer,10.*np.random.randn(layer.size))
            if len(hiddenLayers) > 0:
                for layer in hiddenLayers:
                    for input in inputLayers:
                        w = 10. * np.random.randn(layer.size, input.size)
                        genotype.connect(input, layer, w)
                for output in outputLayers:
                    for hidden in hiddenLayers:
                        w = 10. * np.random.randn(output.size, hidden.size)
                        genotype.connect(hidden, output, w)
            else:
                for layer in outputLayers:
                    for input in inputLayers:
                        w = 10. * np.random.randn(layer.size, input.size)
                        genotype.connect(input, layer, w)
                        
        return genotype
