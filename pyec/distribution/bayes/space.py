"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
from pyec.config import Config
from pyec.distribution.bayes.net import BayesNet
from pyec.distribution.bayes.structure.proposal import StructureProposal        
from pyec.space import Space, Product
from pyec.util.TernaryString import TernaryString

from .variables import GaussianVariable, BinaryVariable
from .sample import DAGSampler

class BayesNetStructure(Space):
    """Space for Bayesian network structure search.
    
    Built on top of a variable space that determines the variables of the
    Bayes net. For example, a Euclidean space, say $\mathbb{R}^d$, would have
    $d$ :class:`GaussianVariable`s. Or, a Binary space would have
    :class:`BinaryVariable`s.
    
    The default choice of variable is controlled by the ``_mapping`` class
    variable based on the ``type`` field of the variable space. To have mixed
    networks, use a :class:`Product` space.
    
    :param space: The space of variables
    :type space: :class:`Space`
    
    """
    _mappings = {  # map space types to bayesian variables
        np.ndarray: GaussianVariable,
        TernaryString: BinaryVariable,
    }
    
    def __init__(self, space):
        super(BayesNetStructure, self).__init__(BayesNet)
        self.space = space
        self.sampler = DAGSampler()
        
        if hasattr(space, 'dim'):
            self.numVariables = space.dim
        else:
            raise ValueError("Cannot determine the number of dimensions "
                             "for BayseNetStructure space based on given "
                             "space; expected space to have property dim "
                             "indicating the number of required variables. ")
        
        self.config = Config(numVariables=self.numVariables,
                             variableGenerator=self.variable,
                             randomizer=self.randomize,
                             sampler=self.sample)
        self.proposal = StructureProposal(**self.config.__properties__)
        self.config.structureGenerator = self.proposal
        
        self.edges = []
        self.nedges = []
    
    def in_bounds(self, network, **kwargs):
        c = network.config
        return (isinstance(network, BayesNet) and
                len(network.variables) == self.numVariables and
                np.all([v.__class__ == self.variable(v.index,c).__class__
                        for v in network.variables]))
    
    def randomize(self, network=None):
        """Retrieve an initial random state for a BayesNetwork in this space.
        
        :param network: An unused parameter used by the caller
        :type param: :class:`BayesNet`
        :returns: A sample in ``self.space`` with an initial state for the
                  network
        
        """
        return self.space.random()
    
    def sample(self, network):
        """Retrieve a free sample from a BayesNetwork.
        
        :param network: The Bayes net to sample
        :type network: :class:`BayesNet`
        :returns: A sample in ``self.space`` drawn from the network's
                  distribution
        
        """
        return self.sampler(network)
        
    def variable(self, index, config):
        """Create a variable for the specified index with the provided config.
        
        :param index: The index identifying the variable to be created in the
                      network that will use it.
        :type index: ``int``
        :param config: Configuration parameters for the variable to use
        :type config: :class:`Config`
        :returns: A :class:`Variable` in the space
        
        """
        if self.space.type in self._mappings:
            return self._mappings[self.space.type](index, config)
        elif isinstance(self.space, Product):
            total = 0
            for space in self.space.spaces:
                total += space.dim
                if total > index:
                    if space.type in self._mappings:
                        return self._mappings[self.space.type](index, config)
                    else:
                        raise ValueError("Failed to locate variable mapping "
                                         "for subspace {0}".format(space))
        
        raise ValueError("Failed to locate variable mapping "
                         "for space {0}".format(self.space))
            
        
    def proportion(self, smaller, larger, index):
        return  2.0 ** ((len(smaller.edges) + len(smaller.nedges)) -
                        (len(larger.edges) + len(larger.nedges)))
    
    def area(self, **kwargs):
        return 1.0
    
    def random(self):
        network = BayesNet(**self.config.__properties__)
        for i in xrange(5):
            self.proposal.search(network)
        return network
    
    def extent(self):
        return self.nedges, self.edges
        #raise NotImplementedError("Bayesian networks do not have a "
        #                          "well-defined notion of spatial extent; "
        #                          "If you have something in mind, please "
        #                          "subclass the space or submit a pull "
        #                          "request.")

    def hash(self, point):
        point.computeEdgeStatistics()
        return str(point.edges)
    
    def copy(self, point):
        return BayesNet.parse(str(point), self.config)


class BayesNetFixedEdges(BayesNetStructure):
    """A subset of :class:`BayesNetStructure` defined by requiring that
    one or more edges be present.
    
    :param space: The probability space for the Bayesian network. Passed to
                  the parent constructor.
    :type space: :class:`Space`
    :param edges: A list of edges as found in ``BayesNet.edges`` which must be
                  present for all networks in this set.
    :type edges: List of tuples
    :param edges: A list of edges NOT found in ``BayesNet.edges`` for all
                  networks in this set
    
    """
    def __init__(self, space, edges, nedges):
        super(BayesNetFixedEdges, self).__init__(space)
        self.edges = edges
        self.nedges = nedges
        self._area = None
    
    def in_bounds(self, x, **kwargs):
        """Checks ``BayesNetStructure.in_bounds`` and then checks for all
        edges in the set.
        
        """
        if "index" in kwargs:
            index = kwargs["index"]
            if index in self.edges:
                return x.hasEdge(*index)
            elif index in self.nedges:
                return not x.hasEdge(*index)
        
        return (super(BayesNetFixedEdges, self).in_bounds(x) and
                np.all([x.hasEdge(*edge) for edge in self.edges]) and
                np.all([not x.hasEdge(*edge) for edge in self.nedges]))

    def area(self, **kwargs):
        if self._area is not None:
            return self._area
        
        if (self.parent is not None and
            self.owner is not None):
            self._area = (self.parent.area() *
                          self.owner.proportion(self, self.parent, None))
        else:
            self._area = 1.0
            
        return self._area
    
    def random(self):
        network = BayesNet(**self.config.__properties__)
        network.edgeRep = self.edges
        network.initialize()
        network = self.proposal.search(network)
        while not self.in_bounds(network):
           network = self.proposal.search(network)
        return network
    