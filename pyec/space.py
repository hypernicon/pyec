"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
from pyec.util.TernaryString import TernaryString

class Space(object):
    """Abstraction for a search domain.
    
    The space object is used to contain all information about a specific 
    domain, including contraints.
    
    In the case of evolutionary computation, the space is the genotype.
    If the phenotype differs from the genotype, then override 
    ``Space.convert`` to perform the conversion; the default conversion
    is the identity.
    
    :param cls: A class for objects in the space. type will be checked
    :type cls: Any type that can be passed to ``isinstance``
    :param constraint: A :class:`Constraint` object for this space. Otherwise, 
                       the space is assumed to be unconstrained.
    :type constraint: :class:`Constraint`
    
    """
    def __init__(self, cls, constraint=None):
        if constraint is not None:
            if not constraint.compatible(self):
                cname = "{0}[{1}]"
                cname = cname.format(self.__class__.__name__, cls.__name__)
                err = "Incompatible constraint for space {0}".format(cname)
                raise ValueError(err)

            self.bounded = False
        else:
            self.bounded = True
        self.constraint = constraint
        self.type = cls
        
    def convert(self, x):
        """Given a point in the space, convert it to a point that can
        be sent to the fitness function. This is mainly useful if the 
        space being searched differs from the domain of the fitness /
        cost function, as is the case for genotype to phenotype mapping
        
        :param x: The point to convert
        :type x:  An object with type ``self.type``
        :returns: The converted point, ready to be passed to the fitness
        
        """
        if not isinstance(x, self.type):
            cname = self.__class__.__name__
            raise ValueError("Type mismatch in {0}.convert".format(cname))
        return x
   
    def in_bounds(self, x):
        """Check whether a point is inside of the constraint region.
        
        Should first check type, then check constraint.
        
        :param x: A point in the space
        :type x: Should be an instance of `class`
        
        """
        if not isinstance(x, self.type):
            return False
        
        if self.constraint is None:
            return True
        else:
            return self.constraint(x)
            
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
    :param constraint: The constraint region for the space
    :type constraint: :class:`Constraint`
    
    """
    def __init__(self, dim=1, center=0.0, scale=1.0, constraint=None):
        super(Euclidean, self).__init__(np.ndarray, constraint)
        self.center = center
        self.scale = scale
        self.dim = dim
        
        if isinstance(center, np.ndarray):
            if shape(center) != dim:
                raise ValueError("Dimension of center doesn't match dim")
        
        if isinstance(scale, np.ndarray):
            if shape(scale) != dim:
                raise ValueError("Dimension of scale array doesn't match dim")
        
        if constraint is not None:
            if hasattr(constraint, 'center'):
                if isinstance(self.center, np.ndarray):
                    fail = (self.center != constraint.center).any()
                else:
                    fail = self.center != constraint.center
                if fail:
                    err = "Constraint center doesn't match space center"
                    raise ValueError(err)
                    
            if hasattr(constraint, 'scale'):
                if isinstance(self.scale, np.ndarray):
                    fail = (self.scale != constraint.scale).any()
                else:
                    fail = self.scale != constraint.scale
                if fail:
                    err = "Constraint scale doesn't match space scale"
                    raise ValueError(err)
                    
            if hasattr(constraint, 'dim') and self.dim != constraint.dim:
                raise ValueError("Constraint dimension doesn't match space")
            
        
    def random(self):
        """Get a random point in Euclidean space. Use the constraint to 
        generate a random point in the space if possible, otherwise
        use a zero-centered elliptical gaussian scaled by ``self.scale``.
        
        """
        if hasattr(self.constraint, 'random'):
            return self.constraint.random()
        else:
            test = self.scale * np.random.randn(self.dim)
            if self.constraint is not None:
                while not self.constraint.in_bounds(test):
                    test = self.scale * np.random.randn(self.dim)
                return test
            else:
                return test
                
    def hash(self, point):
        parts = [((point+i)**2).sum() for i in np.arange(10)]
        return ",".join([str(pt) for pt in parts])
        

class Binary(Space):
    """A binary space of fixed finite dimension.
    
    Uses a :class:`TernaryString` as a representation.
    
    :param dim: The dimension of the space
    :type dim: ``int``
    :param constraint: The constraint region for the space
    :type constraint: :class:`Constraint`
    
    """
    
    def __init__(self, dim=1, constraint=None):
        super(Binary, self).__init__(TernaryString, constraint)
        self.dim = dim
        self.constraint = constraint
        if hasattr(constraint, 'dim') and constraint.dim != dim:
            raise ValueError("Constraint has mismatched dimension")
            
    def random(self):
        """Get a random point in binary space. Use the constraint to 
        generate a random point in the space if possible, otherwise
        use a random byte string.
        
        """
        if hasattr(self.constraint, 'random'):
            return self.constraint.random()
        else:
            test = TernaryString.random(.5, self.dim)
            if self.constraint is not None:
                while not self.constraint.in_bounds(test):
                    test = TernaryString.random(.5, self.dim)
                return test
            else:
                return test

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
                 scale=1.0, constraint=None):
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
