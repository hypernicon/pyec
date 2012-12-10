"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
from pyec.config import Config
from pyec.util.TernaryString import TernaryString
from scipy.special import erf

class Region(object):
    """Abstraction for a subset of a search domain."""
    pass


class Space(Region):
    """Abstraction for a search domain.
    
    The space object is used to contain all information about a specific 
    domain, including contraints.
    
    In the case of evolutionary computation, the space is the genotype.
    If the phenotype differs from the genotype, then override 
    ``Space.convert`` to perform the conversion; the default conversion
    is the identity.
    
    :param cls: A class for objects in the space. type will be checked
    :type cls: Any type that can be passed to ``isinstance``
    
    """
    def __init__(self, cls):
        self.type = cls
        self.parent = None # An immediate supercontainer, if needed
        self.owner = None # The space this subspace/region is in, if any
        
    def area(self, **kwargs):
        """Return the area of the space.
        
        :returns: ``float``, the area/volume/measure of the space
        
        """
        return 1.0
        
    def convert(self, x):
        """Given a point in the space, convert it to a point that can
        be sent to the fitness function. This is mainly useful if the 
        space being searched differs from the domain of the fitness /
        cost function, as is the case for genotype to phenotype mapping
        
        :param x: The point to convert
        :type x:  An object with type ``self.type``
        :returns: The converted point, ready to be passed to the fitness
        
        """
        if not self.in_bounds(x):
            cname = self.__class__.__name__
            raise ValueError("Type mismatch in {0}.convert".format(cname))
        return x
   
    def extent(self):
        """Return a lower and upper vector the extent of the space.
        
        :returns: A tuple with (lower, upper) bounds for the space
        
        """
        raise NotImplementedError("Not all spaces have well-defined extent")
   
    def in_bounds(self, x):
        """Check whether a point is inside of the constraint region.
        
        Should first check type, then check constraint.
        
        :param x: A point in the space
        :type x: Should be an instance of `class`
        
        """
        return isinstance(x, self.type)

    def random(self):
        """Return a random point in the space"""
        raise NotImplementedException
        
    def hash(self, point):
        """Return a hash value for a point in the space.
        
        :param point: Any point in this space
        :type point: ``self.type``
        :returns: The hashed value for the point
        
        """
        return hash(point)


class Euclidean(Space):
    """A Euclidean space of fixed finite dimension.
    
    Uses a numpy array with dtype 64-bit floats.
    
    :param dim: The dimension of the space
    :type dim: ``int``
    :param scale: The scale for the space, integer or array
    :type scale: ``int`` for spherical space, or ``numpy.ndarray``
    
    """
    def __init__(self, dim=1, center=0.0, scale=1.0):
        super(Euclidean, self).__init__(np.ndarray)
        try:
            bottom = center - scale
        except Exception:
            raise ValueError("Mismatched center or scale in Euclidean")
        self.center = center
        self.scale = scale
        self.dim = dim
        
        if isinstance(center, np.ndarray):
            if (dim,) != np.shape(center):
                raise ValueError("Dimension of center doesn't match dim")
        
        if isinstance(scale, np.ndarray):
            if (dim,) != np.shape(scale):
                raise ValueError("Dimension of scale array doesn't match dim")
    
    def gaussInt(self, z):
        # x is std normal from zero to abs(z)
        x = .5 * erf(np.abs(z)/np.sqrt(2))
        return .5 + np.sign(z) * x
    
    def proportion(self, smaller, larger, index):
        """Assume ``smaller`` is a hyperrectangle, and ``larger`` is either
        a euclidean space or a hyperrectangle containing ``smaller``.
        
        To handle something more general, we would need to integrate somehow,
        either monte carlo or decomposing into hyperrectangles.
        
        :param smaller: A Hyperrectangle in this space
        :type smaller: :class:`Hyperrectangle`
        :param larger: A Hyperrectangle or this space, either way containing
                       ``smaller``
        :type larger: :class:`Hyperrectangle` or :class:`Euclidean`
        :returns: The ratio of ``smaller``'s volume over ``larger``'s.
        """
        try:
            center = self.center[index]
            scale = self.scale[index]
        except:
            center = self.center
            scale = self.scale
        slower = smaller.center[index] - smaller.scale[index]
        supper = smaller.center[index] + smaller.scale[index]
        if isinstance(larger, Hyperrectangle):
            llower = larger.center[index] - larger.scale[index]
            lupper = larger.center[index] + larger.scale[index]
        else: # Hyperrectangle 
            llower = -np.inf
            lupper = np.inf
        slow = self.gaussInt((slower - center) / scale) 
        shigh = self.gaussInt((supper - center) / scale)
        llow = self.gaussInt((llower - center) / scale) 
        lhigh = self.gaussInt((lupper - center) / scale)
        return (shigh - slow) / (lhigh - llow)
      
    def extent(self):
        upper = np.zeros(self.dim)
        lower = np.zeros(self.dim)
        upper.fill(np.inf)
        lower.fill(-np.inf)
        return lower, upper
      
    def random(self):
        """Get a random point in Euclidean space. Use the constraint to 
        generate a random point in the space if possible, otherwise
        use a zero-centered elliptical gaussian scaled by ``self.scale``.
        
        """
        test = self.center + self.scale * np.random.randn(self.dim)
        return test
                
    def hash(self, point):
        parts = [((point+i)**2).sum() for i in np.arange(10)]
        return ",".join([str(pt) for pt in parts])
    
    def in_bounds(self, point):
        return isinstance(point, self.type) and np.shape(point) == (self.dim,)


class Hyperrectangle(Euclidean):
    """A Hyperrectangle constraint region within Euclidean space."""
    _area = None
    
    def in_bounds(self, y):
        return (np.abs(np.array(y) - self.center) <= self.scale).all()
   
    def extent(self):
        return self.center - self.scale, self.center + self.scale

    def proportion(self, smaller, larger, index):
        return smaller.scale[index] / larger.scale[index]

    def area(self, **kwargs):
        if self._area is not None:
            return self._area
        
        if ("index" in kwargs and
            self.parent is not None and
            self.owner is not None):
            self._area = (self.parent.area() *
                          self.owner.proportion(self,
                                                self.parent,
                                                kwargs["index"]))
        else:
            # Lebesgue
            self._area = (2*self.scale).prod()
            
        return self._area
   
    def random(self):
       base = np.random.random_sample(np.shape(self.center))
       return self.center - self.scale + 2 * self.scale * base


class Binary(Space):
    """A binary space of fixed finite dimension.
    
    Uses a :class:`TernaryString` as a representation.
    
    :param dim: The dimension of the space
    :type dim: ``int``
    
    """
    _area = None
    
    def __init__(self, dim=1):
        super(Binary, self).__init__(TernaryString)
        self.dim = dim
        
    def area(self):
        return 2.0 ** self.dim
            
    def extent(self):
        return TernaryString(0L, 0L, self.dim), TernaryString(-1L, 0L, self.dim)
            
    def random(self):
        """Get a random point in binary space. Use the constraint to 
        generate a random point in the space if possible, otherwise
        use a random byte string.
        
        """
        return TernaryString.random(.5, self.dim)


class BinaryReal(Binary):
    """A binary genotype with a Euclidean phenotype.
    
    The conversion is scaled and centered. The formula is 
    ``center - scale + 2 * scale * converted_bits`` where
    ``converted_bits`` is obtained by interpreting ``bitDepth``
    bits as a fixed point decimal number between 0 and 1.
    
    :param realDim: How many real dimensions
    :type realDim: ``int``
    :param bitDepth: How many bits per number?
    :type bitDepth: ``int``
    :param center: The center point of converted values
    :type center: ``float`` or ``numpy.ndarray``
    :param scale: The scale of the space, from the center to the sides
    :type scale: ``float`` or ``numpy.ndarray``
    
    """
    
    def __init__(self, realDim=1, bitDepth=16, center=0.0, 
                 scale=1.0):
        self.bitLength = realDim * bitDepth
        super(BinaryReal, self).__init__(self.bitLength)
        self.realDim = realDim
        self.bitDepth = bitDepth
        self.center = center
        self.scale = scale
        self.adj = center - scale
        self.scale2 = 2 * scale
        
        if isinstance(center, np.ndarray):
            if shape(center) != realDim:
                raise ValueError("Dimension of center doesn't match dim")
        
        if isinstance(scale, np.ndarray):
            if shape(scale) != realDim:
                raise ValueError("Dimension of scale array doesn't match dim")
        
    def convert(self, x):
        if not isinstance(x, self.type):
            cname = self.__class__.__name__
            raise ValueError("Type mismatch in {0}.convert".format(cname))
        
        if x.length < self.bitLength:
            err = "Not enough bits in {0}; needed {1}"
            err.format(x, self.bitLength)
            raise ValueError(err)
            
        ret = np.zeros(self.realDim, dtype=np.float)
        
        idx = 0
        for i in xrange(self.realDim):
            val = 0.0
            current = 0.5
            for j in xrange(self.bitDepth):
                val += current * x[idx]
                current /= 2.0
                idx += 1
            ret[i] = val
            
        return self.adj + self.scale2 * ret


class BinaryRectangle(Binary):
    """A binary constraint generated by a :class:`TernaryString` whose
    ``known`` value specifies the constrained bits and whose ``base``
    contains the constraints at those bits.
   
    :param spec: A :class:`TernaryString` whose
                 ``known`` value specifies the constrained bits and 
                 whose ``base`` contains the constraints at those bits
    :type spec: :class:`TernaryString`
         
    """
    def __init__(self, spec):
        if not isinstance(spec, TernaryString):
           raise ValueError("BinaryRectangle expects a TernaryString")
           
        self.spec = spec
        dim = spec.length
        super(BinaryRectangle, self).__init__(dim)
       
    def in_bounds(self, x):
        """Test containment; x must "know" more than spec, and be equal at 
        the known bits.
       
        :param x: The point to test
        :type x: :class:TernaryString
        :returns: A ``bool``, ``True if ``x`` is in the space, ``False``
                  otherwise
       
        """
        return self.spec < x
    
    def extent(self):
        lower = 0L | (self.spec.known & self.spec.base)
        upper = -1L & (self.spec.known & self.spec.base)
        return (TernaryString(lower, -1L, self.spec.length),
                TernaryString(upper, -1L, self.spec.length))
    
    def area(self, **kwargs):
        """Count the number of known bits"""
        if self._area is not None:
            return self._area
        
        if self.parent is not None:
            self._area = .5 * self.parent.area()
        else:
            # Lebesgue
            mask = 1L
            total = 0.0
            for i in xrange(self.spec.length):
                total += (mask & self.spec.known) > 0
            self._area = total
            
        return self._area
       
    def random(self):
        """Return a random TernaryString conforming to the constraint.
       
        :returns: A :class:`TernaryString`
        
        """
        test = super(BinaryRectangle, self).random()
        base = (~self.spec.known & test.base) 
        base |= (self.spec.known & self.spec.base)
        test.base = base
        return test 


class BayesianNetworks(Space):
    """Space for Bayesian network structure search.
    
    """
    def __init__(self,
                 numVariables,
                 variableGenerator,
                 structureGenerator,
                 randomizer,
                 sampler):
        from pyec.distribution.bayes.net import BayesNet
        from pyec.distribution.bayes.structure.proposal import StructureProposal
        super(BayesianNetworks, self).__init__(BayesNet)
        self.config = Config(numVariables=numVariables,
                             variableGenerator=variableGenerator,
                             structureGenerator = structureGenerator,
                             randomizer=randomizer,
                             sampler=sampler)
        self.proposal = StructureProposal(self.config)
        
    def area(self, **kwargs):
        return 2.0 ** (self.config.numVariables ** 2)
    
    def random(self):
        return self.proposal()
    
    def extent(self):
        return TernaryString(0L, 0L, self.dim), TernaryString(-1L, 0L, self.dim)
    