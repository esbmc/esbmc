#!/usr/bin/env python

import os

from sys import argv
#from os import listdir
#from os.path import isfile, join, splitext

script, dire, printopt = argv

onlyfiles = [f for f in os.listdir(dire) if os.path.isfile(os.path.join(dire, f))]
onlyfiles.sort()

for filename in onlyfiles:

  if(filename == "Makefile.am"):
    continue;

  if(filename == "Makefile.in"):
    continue;

  if(filename == "transform.py"):
    continue;

  name, ext = os.path.splitext(filename)

  if(ext != ".h" and ext != ".hs"):
    print "Remove this file, it's not needed: " + filename

  if(ext == ".h"):
    new_filename = name + ".hs"
    if(os.path.exists(new_filename)):
      os.remove(new_filename)

    os.rename(filename, new_filename)
    filename = new_filename

  if(filename.startswith('clang_') != True):
    new_name = "clang_" + name
    new_filename = new_name + ".hs"

    if(os.path.exists(new_filename)):
      os.remove(new_filename)

    os.rename(filename, new_filename)

  if(printopt == "0"):
    print "{ \"" + name[6:len(name)] + ".h\", " + name + "_buf, &" + name + "_buf_size},"
  elif(printopt == "1"):	
    print "extern char " + name + "_buf[];"
    print "extern unsigned int " + name + "_buf_size;\n"
  else:
    print filename
