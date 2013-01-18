#!/bin/python
#############################
# Script to add timeout parameter to test.desc
##############################

import sys
import os
from sys import argv
import xml.etree.ElementTree as ET

#STR_OPT = 'item_04_esbmc-option'
STR_OPT = 'item_05_option_to_run_esbmc'
TIMEOUT = '3600'

def error(message):
    sys.stderr.write("error: %s\n" % message)
    #sys.exit(1)

error_file = "resultados_error.log"
f = open(error_file, 'w')



def ChangeExecutionParameters(path):
    """
    Parse tree
    """
    os.chdir(path)
    print "##### Directory: " + path
    print
    try:
        tree = ET.parse("test.desc")
    except IOError as e:
        error("Could not open test.desc")
        return
    #XML file
    root = tree.getroot()

    res = root
    ite_par = res.find(STR_OPT)
    ite_param = ite_par.text 
    print ite_param
    timeout = ite_param.split();
    try:
       pos = timeout.index("--timeout") #unwind option
       timeout[pos+1] = TIMEOUT
       print pos
       timeout =  " ".join(timeout)
       ite_par.text = timeout
       print ite_par.text
    except:
       print "---> Writing test.desc"
       ite_par.text = ite_param + " --timeout " + TIMEOUT
       print ite_par.text

    tree.write('test.desc',encoding="utf-8",xml_declaration=True)
    #adding new empty line
    print
    try:
      open("test.desc","a").write("\n")
      pass
    except :
      pass


def main():
    if len(sys.argv) < 2:
      print "usage: %s <PATH to test suite>" % argv[0]
      sys.exit(1)
    path = argv[1];
    listing = os.listdir(path)
    listing.sort() #sort files
    os.chdir(path)
    for infile in listing:
      if os.path.isdir(infile):
        ChangeExecutionParameters(infile)
        print
        os.chdir("..")
          
if __name__ == "__main__":
    main()
