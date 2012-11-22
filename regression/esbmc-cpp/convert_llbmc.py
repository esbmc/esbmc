#!/bin/python
import os
import xml.etree.ElementTree as ET
import sys
from sys import argv
#from os.path import expanduser
#home = expanduser("~")


def show_info(path):
    #changing path
    os.chdir(path)
    print "#############################################"
    print path
    print "#############################################"
    tree = ET.parse("test.desc")
    root = tree.getroot()

    print "---> TEST.DESC:"
    #getting the options from file test.desc
    opt = root.find("item_05_option_to_run_esbmc").text
    expected = root.find("item_06_expected_result");
    print "---> " + opt
    print "---> " + expected.text
    unw = opt.split()
    print unw
    pos = unw.index("--unwind") #unwind option
    if (pos+1) < len(unw):
       unw_value = unw[pos+1] #value
       print unw_value

    #testllvm.desc
    tree = ET.parse("testllvm.desc")
    root = tree.getroot()

    print "---> TEST.DESC:"
    #getting the options from file test.desc
    llbmc_item = root.find("item_05_option_to_run_esbmc") 
    llbmc_opt = llbmc_item.text
    expected_llbmc = root.find("item_06_expected_result");
    print "---> " + llbmc_opt
    print "---> " + expected_llbmc.text
    print llbmc_opt
    llbmc_s = llbmc_opt.split()
    llbmc_s[1] = "--max-loop-iterations=" + str(unw_value)
    changed = " ".join(llbmc_s)
    llbmc_item.text = changed

    expected_llbmc.text = expected.text
    tree.write('testllvm.desc',encoding="UTF-8")
    
    os.system("echo '' >> testllvm.desc")

    


   # raw_input("Press ENTER to next")

def main():
    path = argv[1];
    print path    
    listing = os.listdir(path)
    listing.sort()
#    raw_input()
    for infile in listing:
        if os.path.isdir(path+infile):
          #print infile
          show_info(path+infile)
    
if __name__ == "__main__":
    main()


