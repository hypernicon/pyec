"""
Copyright (C) 2012 Alan J Lockett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
Parse .net format for Bayes nets and return a bayes net

"""

from pyec.config import Config
from pyec.distribution.bayes.net import *
from pyec.distribution.bayes.structure.proposal import StructureProposal

class BayesParser(object):

   def __init__(self):
      self.variables = {}
      self.indexMap = {}
      self.revIndexMap = {}
      self.index = 0
   
   def processLine(self, line):
      line = line.strip()
      if line == "":
         return None
      line = line[1:-1]
      parts = line.split(" ")
      if parts[0] == "var":
         name = parts[1].strip("' ")
         vals = " ".join(parts[2:])
         vals = vals.strip("'()").split(" ")
         vals = [v.strip("() \t\r\n") for v in vals]
         vals = [v for v in vals if v != ""]
         self.variables[name] = {'vals':vals, 'parents':None, 'cpt':None}
         self.indexMap[name] = self.index
         self.revIndexMap[self.index] = name
         self.index += 1
      elif parts[0] == "parents":
         name = parts[1].strip("'").strip()
         parts2 = line.split("'(")
         if len(parts2) == 1:
            parts2 = line.split("(")
            parstr = parts2[1]
            cptstr = "(".join(parts2[2:])
         else:
            parstr = parts2[1]
            cptstr = parts2[2]
         parents = parstr.strip(") \n").split(" ")
         parents = [parent for parent in parents if parent != ""]
         sortedParents = sorted(parents, key=lambda x: self.indexMap[x])
         self.variables[name]['parents'] = sortedParents
         cpt = {}
         if len(parents) == 0:
            vals = cptstr[:-2].strip("( )\r\n\t").split(" ")
            vals = array([float(v) for v in vals][:-1])
            cpt[""] = vals
         else:
            rows = cptstr[:-2].split("((")
            for row in rows:
               row = row.strip(") \r\n\t")
               if row == "": continue
               cfg, vals = row.split(")")
               keys = [c for c in cfg.split(" ") if c != ""]
               keyStr = [[]] * len(parents)
               for j, key in enumerate(keys):
                  options = self.variables[parents[j]]['vals']
                  idx = options.index(key) + 1
                  keyStr[sortedParents.index(parents[j])] = idx
               keyStr = ",".join([str(i) for i in array(keyStr)])
               vals = vals.strip().split(" ")
               vals = array([float(v) for v in vals][:-1])
               cpt[keyStr] = vals
         self.variables[name]['cpt'] = cpt
      else:
         return False   
            
   def parse(self, fname):
      f = open(fname)
      totalLine = ""
      done = False
      for line in f:
         totalLine += line
         lefts = len(totalLine.split("("))
         rights = len(totalLine.split(")"))
         if lefts == rights:
            self.processLine(totalLine)
            totalLine = ""
      categories = [[]] * self.index
      for name, idx in self.indexMap.iteritems():
         categories[idx] = self.variables[name]['vals']
      
      cfg = Config()
      cfg.numVariables = len(self.variables)
      cfg.variableGenerator = MultinomialVariableGenerator(categories)
      cfg.randomizer = MultinomialRandomizer()
      cfg.sampler = DAGSampler()        
      cfg.structureGenerator = StructureProposal(cfg)
      
      net = BayesNet(cfg)
      for variable in net.variables:
         variable.tables = self.variables[self.revIndexMap[variable.index]]['cpt']
         #print names[variable.index], self.variables[self.revIndexMap[variable.index]]['parents']
         variable.known = [self.indexMap[parent] for parent in self.variables[self.revIndexMap[variable.index]]['parents']]
         variable.known = sorted(variable.known)
         variable.parents = dict([(i, net.variables[i]) for i in variable.known])
      net.dirty = True
      net.computeEdgeStatistics()
      """
      for variable in net.variables:
         print "(var ", self.revIndexMap[variable.index], " (", " ".join(variable.categories[variable.index]), "))"
      
      for variable in net.variables:
         print "(parents ", self.revIndexMap[variable.index], " (", " ".join([self.revIndexMap[i] for i in variable.known]), ") "
         for key, val in variable.tables.iteritems():
            if key == "":
               expanded = ""
            else:
               cfg = array([int(num) for num in key.split(",")])
               expanded = " ".join(self.variables[self.revIndexMap[variable.known[k]]]['vals'][c-1] for k,c in enumerate(cfg))
            
            total = val.sum()
            vals = " ".join([str(i) for i in val])
            print "((", expanded, ") ", vals, (1. - total), ")"
         print ")"
      """
      return net