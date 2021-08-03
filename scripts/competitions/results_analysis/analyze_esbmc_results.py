#!/usr/bin/env python

import sys, copy, random

import cPickle as pickle
import gzip

def read_strats(data_file_names):

  solvers = {}   # maps solvers to expers where each exper is a {solved_problems:status} dict
  
  # loading
  for name in data_file_names:
    with open(name,"r") as f:
      for line in f:
        _,prob,_,solver,_,config,_,_,time,_,memory,stat = line.split(",")[:12]

        if ".rm" in prob:
          pass

        solver_name = solver+"_"+config

	if "SMT" in prob:
		logic = prob.split("/")[1]
		solver_name = logic+"_"+solver_name

        if solver_name in solvers:
          exper = solvers[solver_name]
        else:
          exper = {}
          solvers[solver_name] = exper
            
        if prob in exper: # if there were multiple runs of the same strategy, conut
          pass
        else:
          exper[prob] = stat.strip() 

  return solvers

if __name__ == "__main__":
  
  solvers = read_strats(["Job42771_info.csv"])
  #del solvers["solver_configuration"]
  
  m = max({len(k) for k in solvers.keys()}) 

  # map from problem to solvers that solve it
  solved = {}

  for (solver,exper) in sorted(solvers.items()):
    count = 0
    for (prob,stat) in exper.items(): 
      if stat=="Refutation" or stat=="Theorem" or stat=="Unsatisfiable" or stat=="unsat":
        count +=1
        if prob not in solved:
          solved[prob] = set()
        solved[prob].add(solver)
    print solver,"\t:",count
             
  #sys.exit(0)

  uniques = {}
  for (prob,solutions) in solved.items():    
    if len(solutions)==1:
      s = solutions.pop()
      if s not in uniques:
        uniques[s] = set()
      uniques[s].add(prob)

  print "Uniques numbers:"
  for (solver,u) in sorted(uniques.items()):
    print "\t",solver,"\t:",len(u) 

  sys.exit(0)
  print "Unique problems:"
  for (solver,u) in sorted(uniques.items()):
    print "\t",solver,":"
    for prob in u:
      print "\t\t",prob
