#!/bin/python
#############################
# Script to add timeout and/or memlimit parameter to test.desc
##############################
import fnmatch
import os
import shutil
import sys
import os
from sys import argv
import xml.etree.ElementTree as ET

STR_OPT = 'item_05_option_to_run_esbmc'
TIMEOUT_CONST = '3600s'
MEMLIMIT_CONST = '1g'
global TIMEOUT, MEMLIMIT

def error(message):
    sys.stderr.write("error: %s\n" % message)
    #sys.exit(1)

error_file = "results_error.log"
f = open(error_file, 'w')

def ChangeExecutionParameters(path):
    """
    Parse tree
    """
    print "##### File: " + path + " #####"
    try:
        tree = ET.parse(path)
    except IOError as e:
        error("Could not open test.desc")
        return
    #XML file
    root = tree.getroot()

    res = root
    ite_par = res.find(STR_OPT)
    ite_param = ite_par.text
    #TIMEOUT code
    print "## Actual parameter: " + ite_param
    timeout = ite_param.split();
    print "## Changing the timeout parameter... "
    if (TIMEOUT == -1):
       try:
         pos = timeout.index("--timeout") #unwind option
         del timeout[pos]
         del timeout[pos]
         timeout =  " ".join(timeout)
         ite_par.text = timeout
         print "New parameter: " + ite_par.text
       except:
         print "Warning: Can not change timeout parameter"
         pass

    else:
      try:
        #Verification if exists timeout parameter
        pos = timeout.index("--timeout") #unwind option
        timeout[pos+1] = TIMEOUT
        print "-> Timeout parameter already exists..."
        print "-> Changing the timeout value to: " + TIMEOUT
        timeout =  " ".join(timeout)
        ite_par.text = timeout
        print "New parameter: " + ite_par.text
      except:
        #timeout parameter does not exist. Creating...
        print "-> Adding timeout parameter"
        ite_par.text = ite_param + " --timeout " + TIMEOUT
        print "New parameter: " + ite_par.text

    #MEMORY limit parameter code
    ite_param = ite_par.text
    print "## Changing the memory limit parameter... "
    memlimit = ite_param.split();
    if (MEMLIMIT == -1):
       try:
         pos = memlimit.index("--memlimit") #unwind option
         del memlimit[pos]
         del memlimit[pos]
         memlimit =  " ".join(memlimit)
         ite_par.text = memlimit
         print "New parameter: " + ite_par.text
       except:
         print "Warning: Can not change memory limit parameter"
         pass
    else:
      try:
        #Verification if exists timeout parameter
        pos = memlimit.index("--memlimit") #unwind option
        memlimit[pos+1] = MEMLIMIT
        print "-> Memlimit parameter already exists..."
        print "-> Changing the memlimit value to: " + MEMLIMIT
        memlimit =  " ".join(memlimit)
        ite_par.text = memlimit
        print "New parameter: " + ite_par.text
      except:
        #memlimit parameter does not exist. Creating...
        print "-> Adding memlimit parameter"
        ite_par.text = ite_param + " --memlimit " + MEMLIMIT
        print "New parameter: " + ite_par.text

    tree.write(path,encoding="utf-8",xml_declaration=True)
    #adding new empty line into file
    print
    try:
      open(path,"a").write("\n")
      pass
    except :
      pass


def main():
  #if len(sys.argv) < 2:
  #  print "usage: %s <root PATH>" % argv[0]
  #  sys.exit(1)
  #path = argv[1];
  global TIMEOUT, MEMLIMIT
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('path', help="Root PATH including all suites")
  parser.add_argument('--timeout', nargs='?', const=TIMEOUT_CONST, default=-1, \
                                   help="configure time limit, integer followed by {s,m,h}")
  parser.add_argument('--memlimit', nargs='?', const=MEMLIMIT_CONST, default=-1, \
                           help="configure memory limit, of form \"100m\" or \"2g\"")
  MEMLIMIT = parser.parse_args().memlimit
  TIMEOUT = parser.parse_args().timeout
  path = parser.parse_args().path;
#  print TIMEOUT
#  print MEMLIMIT

  matches = []
  print "Searching for test.desc files in: " + path
  for root, dirnames, filenames in os.walk(path):
   for filename in fnmatch.filter(filenames, 'test.desc'):
      print
      ac_path = os.path.join(root, filename)
      matches.append(ac_path)
      ChangeExecutionParameters(ac_path)

if __name__ == "__main__":
    main()
