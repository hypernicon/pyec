"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from pyec.config import Config as _
from pyec.space import BinaryReal, Binary, Euclidean
from pyec.distribution.bayes.mutators import *
from pyec.distribution.bayes.space import BayesNetStructure
from pyec.distribution.bayes.structure.proposal import StructureProposal
from pyec.distribution.ec.mutators import *
from pyec.distribution.ec.selectors import *

import logging
log = logging.getLogger(__file__)

"""Mainly, these files are just examples of how to make a genetic algorithm
work. Genetic algorithms are just convolutions of standard components.

"""

"""``SimpleGeneticAlgorithm`` uses proportional selection with one point
crossover and some mutation, all in a binary encoding such as provided by
``Binary`` and ``BinaryReal``.

The following definition says to use proportional selection twice (with the
second selection ignoring the results of the first, ``>> 1``),
followed by one-point crossover, followed by Bernoulli mutation (bit-flipping).
The genotype is set as :class:`BinaryReal` (the ``space``), which generates
bit strings and produces a Euclidean phenotype through the ``convert`` method
of :class:`BinaryReal`.

"""
SimpleGeneticAlgorithm = (
   Proportional << ((Proportional >> 1) <<
                     Crossover[_(crosser=OnePointDualCrosser)])
   << Bernoulli
)[_(space=BinaryReal(realDim=5))]


"""``GeneticAlgorithm`` uses tournament selection over the entire population,
uniform crossover, and Bernoulli mutation.

"""
GeneticAlgorithm = (
   Tournament << ((Tournament >> 1) << Crossover) << Bernoulli
)[_(space=Binary(dim=100))]


"""``RealGeneticAlgorithm`` uses linear ranking selection, uniform crossover,
and Gaussian mutation.

"""
RealGeneticAlgorithm = (
   Ranking << ((Ranking >> 1) << Crossover) << Gaussian
)[_(space=Euclidean(dim=5))]

"""``ElitistGeneticAlgorithm`` shows how to apply elitism; in this case, the top
10% of the population will be preserved for the next generation.

"""
ElitistGeneticAlgorithm = (.1 * Elitist) | (.9 * GeneticAlgorithm)


BayesGeneticAlgorithm = (
  Tournament << ((Tournament >> 1) << Crossover[_(crosser=UniformBayesCrosser)])
  << StructureMutator
)[_(space=BayesNetStructure(space=Binary(dim=25)))]