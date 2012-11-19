"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from time import time

class LLNode(object):
   before = None
   after = None
   value = None

class LinkedList(object):
   first = None
   last = None
   
   def append(self, value):
      node = LLNode()
      node.value = value
      node.before = self.last
      node.after = None
      self.last = node
      if self.first is None:
         self.first = node
      if node.before is not None:
         node.before.after = node

   def remove(self, node):
      if node == self.first:
         self.first = node.after
      if node == self.last:
         self.last = node.before
      if node.before is not None:
         node.before.after = node.after
      if node.after is not None:
         node.after.before = node.before

class LRUCache(object):

   def __init__(self, size=10000):
      self.maxSize = size
      self.times = LinkedList()
      self.objects = {}
      self.timeMap = {}
   
   def clear(self):
      self.times = LinkedList()
      self.objects = {}
      self.timeMap = {}  
    
   def __setitem__(self, key, val):
      if self.timeMap.has_key(key):
         node = self.timeMap[key]
         self.times.remove(node)
      self.objects[key] = val
      self.times.append(key)
      self.timeMap[key] = self.times.last
      while len(self.objects) > self.maxSize:
         lru = self.times.first
         del self.timeMap[lru.value]
         del self.objects[lru.value]
         self.times.remove(lru)
         
   def __getitem__(self, key):
      return self.objects[key]

   def __len__(self):
      return len(self.objects)
      
   def has_key(self, key):
      return self.objects.has_key(key)
         