"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

try:
    import theano
except ImportError:
    print "Could not find Theano. Falling back on CPU-based RNN instance. If you want",
    print "to use a GPU, please `easy_install Theano`"
    from .net import RnnEvaluator
else:    
    
    def RnnEvaluator(genotype):
        """Build a Theano function that computes the internal state of the network
        when called.
        
        """
        states = dict([("s{0}".format(layer.id), theano.vector("s{0}".format(layer.id)))
                       for layer in genotype.layer])
        
        for layer in genotype.layer:
            weights = []
            for source in layer.inLinks:
                weights.append((genotype.links((source, layer)),states["s{0}".format(source.id)]))
    
