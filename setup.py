#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

try:
    from setuptools import setup, find_packages
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()
    from setuptools import setup, find_packages




setup(name='PyEC',
      version='0.2.5',
      description='Evolutionary computation package',
      author='Alan J Lockett',
      install_requires=[
         'numpy >= 1.5.1',
         'scipy >= 0.8.0'
      ],
      packages=find_packages(exclude=['ez_setup']),
      include_package_data=True,
      url='http://www.alockett.com/pyec/docs/0.2/index.html'
)

"""
Note, simpleapi has a requirement on the pstats module, which isn't included in the default Ubuntu
python standard library.  It's a long standing licensing issue, and the workaround is to manually
install python-profiler

sudo apt-get install python-profiler

This is Ubuntu only!!

"""
