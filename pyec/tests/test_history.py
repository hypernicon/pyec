"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
from pyec.config import Config
import unittest

class TestHistory(unittest.TestCase):
   def test_base(self):
      cfg = Config()
      history = History(cfg)
      self.assertEqual(history.evals, 0)
      self.assertEqual(history.printEvery, 1000000000000L)
      self.assertEqual(history.updates, 0)
      self.assertEqual(history.minScore, np.inf)
      self.assertTrue(history.minSolution is None)
      self.assertEqual(history.maxScore, -np.inf)
      self.assertTrue(history.maxSolution is None)
      self.assertTrue(history.empty())
      self.assertTrue(history.config is cfg)
      self.assertTrue(history.cache is not None)
      
      