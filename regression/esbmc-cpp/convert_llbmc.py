#!/bin/python
#############################
# Script to create new file testllvm.desc from test.desc
# If the file does not exist, it will creat it
##############################
import os
import xml.etree.ElementTree as ET
import sys
from copy import copy
from sys import argv

def show_info(path):
    #changing path
    os.chdir(path)
    #printing path
    print "#############################################"
    print path
    print "#############################################"
    #try to open test.desc
    try:
   	tree = ET.parse("test.desc")
    except IOError as e:
        sys.exit("Could not open test.desc")

    #XML file
    root = tree.getroot()

    print "---> TEST.DESC:"
    #getting the options from file test.desc
    opt = root.find("item_05_option_to_run_esbmc").text
    expected = root.find("item_06_expected_result");
    unw = opt.split()
    #print unw

    try:
      pos = unw.index("--unwind") #unwind option

      if (pos+1) < len(unw):
         unw_value = unw[pos+1] #value
      else:
         sys.exit("Parameter unwind error") 
    except:
      unw_value=str(10)

    print "unwind: " + unw_value

    ########### testllvm.desc ###############
    #try to open testllvm.desc
    existsll = True
    try:
        tree = ET.parse("testllvm.desc")
        root = tree.getroot()
    except IOError as e:
    	existsll = False
    print "---> testllvm.desc:"
    print "---> Checking testllvm.desc" 
    llbmc_opt = root.find("item_05_option_to_run_esbmc")
    if not existsll: #file does not exist
        print "---> testllvm.desc does not exist, creating..." 
        llbmc_file = root.find("item_04_file_C_to_test")
        fileName, fileExtension = os.path.splitext(llbmc_file.text)
        llbmc_file.text = fileName + ".bc"
        
        llbmc_opt.text = "--ignore-missing-function-bodies --max-loop-iterations= --no-max-loop-iterations-checks"
    else:
        #expected
        expected_llbmc = root.find("item_06_expected_result");
        expected_llbmc.text = expected.text

    #getting the options from file test.desc
    #max-loop
    llbmc_s = llbmc_opt.text.split()
    llbmc_s[1] = "--max-loop-iterations=" + str(unw_value)
    changed = " ".join(llbmc_s)
    llbmc_opt.text = changed
    
    print "---> Writing testllvm.desc"
    tree.write('testllvm.desc',encoding="utf-8",xml_declaration=True)
    #adding new empty line    
    print
    
    try:
        open("testllvm.desc","a").write("\n")
    except :
        pass
#    raw_input("Press ENTER to next")

def main():
    path = argv[1];
    listing = os.listdir(path)
    listing.sort() #sort files
    os.chdir(path)
    for infile in listing:
        if os.path.isdir(infile):
          show_info(infile)
          os.chdir("..")
          
if __name__ == "__main__":
    main()

