#!/bin/python
#############################
# Script to add timeout parameter to test.desc
##############################
import fnmatch
import os
import shutil
import sys
import os
from sys import argv
import xml.etree.ElementTree as ET

STR_OPT = 'item_05_option_to_run_esbmc'
TIMEOUT = '3600s'

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
    print "## Actual parameter: " + ite_param
    timeout = ite_param.split();
    print "## Changing the parameter... "
    try:
       pos = timeout.index("--timeout") #unwind option
       timeout[pos+1] = TIMEOUT
       print "-> Timeout parameter already exists..."
       print "-> Changing the timeout value to: " + TIMEOUT 
       timeout =  " ".join(timeout)
       ite_par.text = timeout
       print "New parameter: " + ite_par.text
    except:
       print "-> Adding timeout parameter"
       ite_par.text = ite_param + " --timeout " + TIMEOUT
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
  if len(sys.argv) < 2:
    print "usage: %s <root PATH>" % argv[0]
    sys.exit(1)
  path = argv[1];

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
