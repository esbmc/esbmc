#!/usr/bin/env python

import sys, copy, random

import gzip

def read_strats(file_name):

  print(file_name)

  solved_by_solver = set()  
  number_solved_positive = 0;
  number_solved_negative = 0;

  with open(file_name,"r") as f:
    count = 0;
    for line in f:
      if count > 2:
        benchmark, category, expected_status, status, _, _, _ = line.split("\t")[:12]
        if status.startswith(expected_status) :
          if expected_status == "true" :
            number_solved_positive = number_solved_positive + 1 
          else :
            number_solved_negative = number_solved_negative + 1       
          solved_by_solver.add(benchmark + "_" + category)      
      count = count + 1

  return solved_by_solver

if __name__ == "__main__":
  
  solved1 = read_strats(sys.argv[1])
  solved2 = read_strats(sys.argv[2])
  
  print("total number solved by solver1 " + str(len(solved1)))
  print("total number solved by solver2 " + str(len(solved2)))

  uniques_1 = set()
  uniques_2 = set()

  for benchmark in solved1:
    if benchmark not in solved2:
      uniques_1.add(benchmark)


  for benchmark in solved2:
    if benchmark not in solved1:
      uniques_2.add(benchmark)    

  print("total number of uniques solved by solver1 " + str(len(uniques_1)))
  print("total number of uniques solved by solver2 " + str(len(uniques_2)))

  print("\n\nUniques 1:\n")
  for b in uniques_1:
    print("  " + b)

  print("\n\nUniques 2:\n")
  for b in uniques_2:
    print("  " + b)