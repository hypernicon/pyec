"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""This module provides the class :class:`RunStats`, which can be used to track performance 
of repeated behaviors. The module maintains separate timers for different key names.

"""

from time import time

class RunStats(object):
   """
      A simple stats recording tool that can be used to aggregate the
      amount of time spent in various methods of an optimizer. Keeps track 
      of variables by key names and outputs the time spent between `start` 
      and `stop` for each key. For a recorded key, the average time spent 
      between `start` and `stop` can be retrieved by [], like so::
      
         def checkStats():     
            stats = RunStats()
            # start the timer
            stats.start("test")
            .. .
            # stop the timer
            stats.stop("test")
            # print just the key "test"
            print stats["test"]
            # print all
            print stats


   """
   totals = {}
   times = {}
   counts = {}
   recording = True
   
   def start(self, key):
      """
         Start recording time for `key`.
         
         :param key: A name for the key.
         :type key: str
      """
      if not self.recording: return
      if not self.totals.has_key(key):
         self.totals[key] = 0.0
         self.counts[key] = 0
      self.times[key] = time()

   def stop(self, key):
      """
         Stop recording time for `key`.
         
         :param key: The previously started key that is to be stopped.
         :type key: str
      """
      if not self.recording: return
      now = time()
      self.totals[key] += now - self.times[key]
      del self.times[key]
      self.counts[key] += 1
      
   def __getitem__(self, key):
      return self.totals[key] / self.counts[key]
      
   def __str__(self):
      ret = ""
      for key,val in sorted(self.totals.items(), key=lambda x: x[0]):
         ret += "%s: %.9f\n" % (key, self[key])
      return ret

