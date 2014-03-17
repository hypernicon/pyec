"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import copy
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
    
    def __str__(self):
        return "{0}".format(self.__class__.__name__)
        
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
        # checking in bounds may be expensive
        #if not self.in_bounds(x):
        #    cname = self.__class__.__name__
        #    raise ValueError("Type mismatch in {0}.convert".format(cname))
        return x
   
    def extent(self):
        """Return a lower and upper vector the extent of the space.
        
        :returns: A tuple with (lower, upper) bounds for the space
        
        """
        raise NotImplementedError("Not all spaces have well-defined extent")
   
    def in_bounds(self, x, **kwargs):
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

    def copy(self, point):
        """Make a safe copy of a point in the space.
        
        :params point: A point in the space
        :type point: ``space.type``
        :returns: A copy of the point
        
        """
        return copy.copy(point)
    
    def discard(self, point):
        """Allow any type-specific deletion or cleanup.
        
        :params point: A point in the space
        :type point: ``space.type``
        
        """
        pass


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
        if not isinstance(center, np.ndarray):
            center = center * np.ones(dim)
        if not isinstance(scale, np.ndarray):
            scale = scale * np.ones(dim)
        try:
            bottom = center - scale
        except Exception:
            raise ValueError("Mismatched center or scale in Euclidean")
        self.center = center
        self.scale = scale
        self.dim = dim
        
        if (dim,) != np.shape(center):
            raise ValueError("Dimension of center doesn't match dim")
        
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
        center = self.center[index]
        scale = self.scale[index]
        slower = smaller.lower[index]
        supper = smaller.upper[index]
        if isinstance(larger, Hyperrectangle):
            llower = larger.lower[index]
            lupper = larger.upper[index]
        else: # Euclidean
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
    
    def in_bounds(self, point, **kwargs):
        return isinstance(point, self.type) and np.shape(point) == (self.dim,)

    def copy(self, point):
        return np.copy(point)


class Hyperrectangle(Euclidean):
    """A Hyperrectangle constraint region within Euclidean space.
    
    :param lower: A ``numpy.ndarray`` for the lower boundary of the
                  hyperrectangle
    :type lower: ``numpy.ndarray``
    :param upper: A ``numpy.ndarray`` for the upper boundary of the
                  hyperrectangle
    :type upper: ``numpy.ndarray``
    
    """
    _area = None
    
    def __init__(self, lower, upper):
        dim = len(lower)
        if (upper < lower).any():
            raise ValueError("Upper boundary cannot be below lower boundary.")
        scale = .5 * (upper - lower)
        center = lower + scale 
        super(Hyperrectangle, self).__init__(dim, center, scale)
        self.lower = lower
        self.upper = upper
    
    def in_bounds(self, y, **kwargs):
        if "index" in kwargs:
            index = kwargs["index"]
            return self.lower[index] <= y[index] <= self.upper[index]
        return (self.lower <= y).all() and (y <= self.upper).all()
   
    def extent(self):
        return self.lower, self.upper

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
            #self._area = (2*self.scale).prod()
            self._area = 1.0
            
        return self._area
   
    def random(self):
       base = np.random.random_sample(np.shape(self.center))
       return self.lower + 2 * self.scale * base


class IntegerRectangle(Space):
    """An integer space of fixed finite dimension and size.
    
    Uses a numpy array with dtype 64-bit integers.
    
    :param dim: The dimension of the space
    :type dim: ``int``
    :param scale: The scale for the space, integer or array
    :type scale: ``int`` for spherical space, or ``numpy.ndarray``
    
    """
    def __init__(self, lower, upper):
        super(IntegerRectangle, self).__init__(np.ndarray)
        dim = len(lower)
        if (upper < lower).any():
            raise ValueError("Upper boundary cannot be below lower boundary.")
        if upper.dtype != np.int64 or lower.dtype != np.int64:
            raise ValueError("Expected 64 bit integers for boundaries")
        self.lower = lower
        self.upper = upper
        self.bound = (upper-lower).max() ** 2
        self.scale = upper - lower
        self._area = None
        
    def __str__(self):
        return "{0} {1} {2}".format(self.__class__.__name__, list(self.lower), list(self.upper))
    
    def in_bounds(self, y, **kwargs):
        if not isinstance(y, np.ndarray) or y.dtype != np.int64:
            return False
        if "index" in kwargs:
            index = kwargs["index"]
            return self.lower[index] <= y[index] <= self.upper[index]
        return (self.lower <= y).all() and (y <= self.upper).all()
   
    def extent(self):
        return self.lower, self.upper

    def proportion(self, smaller, larger, index):
        return smaller.scale[index] / float(larger.scale[index])

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
            #self._area = (2*self.scale).prod()
            self._area = 1.0
            
        return self._area
   
    def random(self):
       base = np.random.random_integers(0,self.bound,size=np.shape(self.lower))
       return self.lower + (base % (self.scale))
                
    def hash(self, point):
        parts = [((point+i)**2).sum() for i in np.arange(10)]
        return ",".join([str(pt) for pt in parts])
    
    def copy(self, point):
        return np.copy(point)


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
        return 1.0
            
    def extent(self):
        return TernaryString(0L, 0L, self.dim), TernaryString(-1L, 0L, self.dim)
            
    def random(self):
        """Get a random point in binary space. Use the constraint to 
        generate a random point in the space if possible, otherwise
        use a random byte string.
        
        """
        return TernaryString.random(self.dim)
    
    def in_bounds(self, x, **kwargs):
        return isinstance(x, TernaryString) and x.length == self.dim


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
        if not isinstance(center, np.ndarray):
            center = center * np.ones(realDim)
        self.center = center
        if not isinstance(scale, np.ndarray):
            scale = scale * np.ones(realDim)
        self.scale = scale
        self.adj = center - scale
        self.scale2 = 2 * scale
        
        if np.shape(center) != (realDim,):
            raise ValueError("Dimension of center doesn't match dim")
        
        if np.shape(scale) != (realDim,):
            raise ValueError("Dimension of scale array doesn't match dim")
    
        # nb we could still use a cache for higher bit depth,
        # we just have to be more careful -- break it into groups of
        # 16 and then combine
        self.useCache = self.bitDepth <= 16
        if self.useCache:
            self._cache = [self.convertOne(i) for i in xrange(1 << self.bitDepth)]
        else:
            self._cache = None
            
        self.mask = (1L << self.bitDepth) - 1L
    
    def convertOne(self, x):
        val = 0.0
        current = 0.5
        mask = 1L
        for j in xrange(self.bitDepth):
            val += current * (x & mask != 0)
            current /= 2.0
            mask <<= 1
        return val
        
       
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
            nextIdx = idx + self.bitDepth
            b = x[idx:nextIdx]
            b = b.base & b.known & self.mask
            idx = nextIdx
            ret[i] = self.useCache and self._cache[b] or self.convertOne(b)
            
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
       
    def in_bounds(self, x, **kwargs):
        """Test containment; x must "know" more than spec, and be equal at 
        the known bits.
       
        :param x: The point to test
        :type x: :class:TernaryString
        :returns: A ``bool``, ``True if ``x`` is in the space, ``False``
                  otherwise
       
        """
        if "index" in kwargs:
            index = 1L << kwargs["index"]
            if (self.spec.known & index) == 0:
                return True
            elif (x.known & index) == 0:
                return False
            else:
                return (self.spec.base & index) == (x.base & index)
        return self.spec < x
    
    def extent(self):
        lower = 0L | (self.spec.known & self.spec.base)
        upper = -1L & (self.spec.known & self.spec.base | ~self.spec.known)
        return (TernaryString(lower, self.spec.known, self.spec.length),
                TernaryString(upper, self.spec.known, self.spec.length))
    
    def area(self, **kwargs):
        """Count the number of known bits"""
        if self._area is not None:
            return self._area
        
        if self.parent is not None:
            self._area = .5 * self.parent.area()
        else:
            # Lebesgue
            mask = 1L
            total = 1.0
            for i in xrange(self.spec.length):
                total *= 2.0 ** (-((mask & self.spec.known) > 0))
                mask <<= 1
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

class Union(Space):
    """A union of multiple regions. Represents the space $\bigcup_i spaces_i$.
    
    :param spaces: A list of spaces for which the union is to be taken.
    :type spaces: A ``list`` of :class:`Space` instances
    
    """
    def __init__(self, spaces):
        if np.any([isinstance(space, Space) for space in spaces]):
            raise ValueError("Union space takes a list of instances of Space.")
        self.spaces = spaces
    
    @property
    def dim(self):
        for space in self.spaces:
            if hasattr(space, 'dim'):
                return space.dim
        raise ValueError("Asked for dimension of union space, but no member of"
                         "the union has a dimension.")
    
    def random(self):
        idx = np.random.randint(0,len(self.spaces))
        return self.spaces[idx].random()
    
    def area(self):
        return 1.0
    
    def in_bounds(self, x):
        for space in self.spaces:
            if space.in_bounds(x):
                return True
        return False
    
    def hash(self, x):
        return self.spaces[0].hash(x)


class Complement(Space):
    """A complement of one region within another. Represents the space
    $base\setminus subtrahend$.
    
    :param base: The space within which the complement is to be taken.
    :type base: :class:`Space`
    :param subtrahend: The space to be removed from ``base``.
    :type subtrahend: :class:`Space`
    
    """
    def __init__(self, base, subtrahend):
        if not isinstance(base, Space) or not isinstance(subtrahend, Space):
            raise ValueError("Both base and subtrahend for Complement must"
                             " be instances of Space.")
        self.base = base
        self.subtrahend = subtrahend
        
    @property
    def dim(self):
        if hasattr(self.base, 'dim'):
            return self.base.dim
        else:
            raise ValueError("Requested dimension of complement space, but "
                             "the base space does not have property 'dim'")
    
    def random(self):
        """Use rejection sampling to sample the complement"""
        attempts = 0
        while attempts < 1000:
            rnd = self.base.random()
            if not self.subtrahend.in_bounds():
                return rnd
            attempts += 1
        raise RuntimeError("Failed to sample Complement afer 1000 attempts.")
            

    def area(self):
        return self.base.area() - self.subtrahend.area()
    
    def in_bounds(self, x, **kwargs):
        """A point is in the complement if it is in the ``base`` but not in
        the ``subtrahend``.
        
        """
        return (self.base.in_bounds(x, **kwargs) and
                not self.subtrahend.in_bounds(x, **kwargs))
    
    def hash(self, x):
        return self.base.hash(x)
    

class Product(Space):
    """A topological product space formed from the Cartesian product
    of multiple spaces.
    
    :param spaces: The spaces from which the product is to be formed
    :type spaces: ``list`` of :class:`Space` objects
    
    """
    def __init__(self, *spaces):
        super(Product, self).__init__(list)
        self.spaces = spaces
     
    @property
    def dim(self):
        dim = 0
        for space in self.spaces:
            if hasattr(space, 'dim'):
                dim += space.dim
            else:
                raise ValueError("Requested dimension of product space, but "
                                 "one or more subordinate spaces does not "
                                 "have property 'dim' used to look up the "
                                 "space's dimension.")
       
    def random(self):
        return [space.random() for space in self.spaces]
    
    def area(self, **kwargs):
        return float(np.prod([space.area(**kwargs) for space in self.spaces]))
    
    def extent(self):
        extents = [space.extent() for space in self.spaces]
        lowers = [lower for lower,upper in extents]
        uppers = [upper for upper in extents]
        return lowers, uppers
    
    def in_bounds(self, x):
        return np.ndarray([space.in_bounds(y)
                           for y,space in zip(x,self.spaces)]).all()
    
    def hash(self, x):
        return "|:|".join([space.hash[y] for y,space in zip(x,self.spaces)])


class NumpyProduct(Product):
    def __init__(self, *spaces):
        super(NumpyProduct, self).__init__(*spaces)
        for space in spaces:
            if not issubclass(space.type, np.ndarray):
                raise ValueError("Numpy product space must be composed of spaces"
                                 " with type numpy, not {0}".format(space.config.type))
        self.type = np.ndarray

    def convert(self, x):
        return np.concatenate(x)
    
    def extent(self):
        lowers, uppers = super(NumpyProduct, self).extent()
        return np.concatenate(lowers), np.concatenate(uppers)
    
    @property
    def dim(self):
        return sum([space.dim for space in self.spaces])


class EndogeneousProduct(Product):
    """A product space for which only the first portion of the space is to be
    passed to the objective function. Other spaces represent endogeneous
    parameters for mutation, as in Evolution Strategies.
    
    """
    def convert(self, x):
        """Return the value of ``x`` in the first space.
        
        :param x: An element in the space
        :type x: ``list``
        
        """
        return x[0]

class LayeredSpace(Space):
    """A Mixin for layered spaces, that is, hierarchical product spaces. This is
    used for Neural Network spaces within Evolutionary Annealing.
    
    """
    def extractLayers(self, x):
        raise NotImplementedError("Subclasses must implement this.")
    
    def layerFactor(self):
        """A proportional weight for this layer. Two regions in the same
        layer of a layered space may be compared for proportional weight
        using the ratio of their output on this function.
        
        :returns: a ``float`` indicating the weight in this region
        
        """
        return 1.0
    
    def layers(self, x):
        """Pull out the layer features for comparison from the argument.
        
        :param x: An instance of ``self.type``
        :type x: ``self.type``
        :returns: A ``list`` of objects
        
        """
        return x
    
    def wrapLayer(self, region):
        """Wrap a region with a wrapper that will unpack the point
        for comparison within that region.
        
        :param region: An instance of :class:`Space`
        :type region: :class:`Space`
        :returns: a :class:`LayerWrapper` that wraps region if necessary, the
                  region itself if no wrapping is needed.
        
        """
        return region
    
    
class LayerWrapper(Space):
    """Wrap a region so that an object can be converted with a call to
    ``unwrap`` before calling the region's methods.
    
    :param space: The space to wrap
    :type space: :class:`Space`
    
    """
    def __init__(self, space):
        self.wrapped = space
    
    def __str__(self):
        return "{0} for {1}".format(self.__class__.__name__, self.wrapped)
    
    def __repr__(self):
        return str(self)
    
    def unwrap(x):
        raise NotImplementedException("LayerWrapper.unwrap() is abstract")
    
    def random(self):
        raise NotImplementedException("LayerWrapper.random() is not allowed")
    
    def extent(self):
        return self.wrapped.extent()
    
    def copy(self, x):
        raise NotImplementedException("LayerWrapper.copy() is not allowed")
    
    def hash(self, x):
        raise NotImplementedException("LayerWrapper.hash() is not allowed")
    
    def in_bounds(self, x, **kwargs):
        return self.wrapped.in_bounds(self.unwrap(x), **kwargs)
    
    def area(self, **kwargs):
        return self.wrapped.area(**kwargs)
    
    def proportion(self, smaller, larger, index):
        return self.wrapped.proportion(smaller, larger, index)
    
    @property
    def type(self):
        return self.wrapped.type
    
    def get_owner(self):
        return self.wrapped.owner
    
    def set_owner(self, owner):
        self.wrapped.owner = owner
    
    owner = property(get_owner, set_owner)
    
    def get_parent(self):
        return self.wrapped.parent
    
    def set_parent(self, parent):
        self.wrapped.parent = parent
    
    parent = property(get_parent, set_parent)