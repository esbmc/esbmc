#!/usr/bin/env python

import os
import commands
import tempfile
import atexit
import shutil
import subprocess
import argparse
import multiprocessing

def run_command(cmd):
  print cmd
  proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
  for line in iter(proc.stdout.readline, ""):
    print line,

if not os.path.isdir('./src'):
  print "Please run from the top level dir"
  exit(1)

if not commands.getoutput('uname -m') == 'x86_64':
  print "You must build ESBMC releases on a 64 bit machine, sorry"
  exit(1)

if not commands.getoutput('uname -s') == 'Linux':
  print "Not building on linux is not a supported operation; good luck"
  exit(1)

parser = argparse.ArgumentParser()
parser.add_argument("--clangdir", help="Path to the clang directory")
parser.add_argument("--llvmdir", help="Path to the llvm directory")
parser.add_argument("--boostpython", help="boost_python lib name")
parser.add_argument("-a", "--arch", help="Either 32 or 64 bits", type=int, choices=[32, 64], default=64)
parser.add_argument("--compiler", help="Compiler to preprocess the benchmarks", choices=['g++', 'clang++'], default='g++')

args = parser.parse_args()
clangdir = args.clangdir
llvmdir = args.llvmdir
arch = args.arch
compiler = args.compiler
boostpython = args.boostpython

# Procedure:
# 1) Tell the user what's about to happen
# 2) Produce cleaned tarball via export-pub
# 3) For each target:
# 3a) Extract to an arbitary dir
# 3b) Run configure apropriately
# 3c) make
# 3d) Manufacture release tree
# 3e) Tar up
# 4) Clear up

print "Will now produce a source distribution and compile into a release. Any local uncommitted changes will be ignored"

with open('src/esbmc/version', 'r') as version_file:
  esbmcversion = version_file.read()
print 'ESBMC version: ' + esbmcversion

# 2

tmpfile = tempfile.mktemp()
srcdir = tempfile.mkdtemp()
builddir = tempfile.mkdtemp()
destdir = tempfile.mkdtemp()
here = os.getcwd()
releasedir = here + '/.release'

if not os.path.isdir(releasedir):
  os.mkdir(releasedir)

@atexit.register
def fin():
  shutil.rmtree(srcdir)
  shutil.rmtree(builddir)
  shutil.rmtree(destdir)

print 'Exporting public version of ESBMC, please wait.'
print run_command('src/scripts/export-pub.sh -n buildrelease ' + tmpfile)

# 3

# 3a) Extract to an arbitary dir
run_command("tar -xzf" + tmpfile + " -C " + srcdir)

def do_build(releasename, configureflags):
  curbuilddir = builddir + "/" + releasename
  os.mkdir(curbuilddir)
  os.chdir(curbuilddir)

  destname = destdir + "/" + releasename
  os.mkdir(destname)
  run_command(srcdir + '/buildrelease/src/configure --prefix=' + destname + " " + configureflags)

  # build
  run_command('make -j' + str(multiprocessing.cpu_count()+1))
  run_command('make install')

  print os.getcwd()
  print releasedir
  print curbuilddir

  # Make tar
  os.chdir(destdir)
  run_command('tar -czf ' + releasedir + "/" + releasename + '.tgz ' + releasename)

solver_opts = "--disable-yices --disable-cvc4 --disable-mathsat --enable-z3 --enable-boolector"
cxxflags = "-DNDEBUG -O3"
cflags = "-DNDEBUG -O3"

if arch == 32:
  cxxflags += " -m32"
  cflags += " -m32"

configure_str = "CXX=" + compiler + " "
configure_str += "CXXFLAGS=\"" + cxxflags + "\" "
configure_str += "CFLAGS=\"" + cflags + "\" "

if clangdir:
  configure_str += "--with-clang-libdir=" + clangdir + " "

if llvmdir:
  configure_str += "--with-llvm=" + llvmdir + " "

if boostpython:
  configure_str += "--with-boost-python-libname=" + boostpython + " "

configure_str += "--enable-esbmc --enable-shared --enable-python "

# Build dynamic
do_build('esbmc-v' + esbmcversion + '-linux-' + str(arch), configure_str)

# Build static
do_build('esbmc-v' + esbmcversion + '-linux-static-' + str(arch), configure_str + " --enable-static-link")
