#!/usr/bin/env python

import os
import argparse
import shlex
import subprocess

# strings
esbmc_path = "./esbmc "
witness_path = "error-witness.graphml "

# ESBMC default commands: this is the same for every submission
esbmc_dargs = "--no-div-by-zero-check --force-malloc-success --context-bound 7 "
esbmc_dargs += "--clang-frontend "
esbmc_dargs += "--witness-output " + witness_path

# ESBMC specific commands: this is different for every submission
esbmc_fp    = "--floatbv --mathsat --no-bitfields "
esbmc_kind  = "--floatbv --unlimited-k-steps --z3 --k-induction-parallel "
esbmc_falsi = "--floatbv --unlimited-k-steps --z3 --falsification "
esbmc_incr  = "--floatbv --unlimited-k-steps --z3 --incremental-bmc  "
esbmc_fixed = "--unwind 128 --no-unwinding-assertions --boolector "

command_line = esbmc_path + esbmc_dargs

# Options

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--arch", help="Either 32 or 64 bits", type=int, choices=[32, 64], default=32)
parser.add_argument("-v", "--version", help="Prints ESBMC's version", action='store_true')
parser.add_argument("-p", "--propertyfile", help="Path to the property file")
parser.add_argument("benchmark", nargs='?', help="Path to the benchmark")
parser.add_argument("-s", "--strategy", help="ESBMC's strategy", choices=["kinduction", "fp", "falsi", "incr", "fixed"], default="incr")

args = parser.parse_args()

arch = args.arch
version = args.version
property_file = args.propertyfile
benchmark = args.benchmark
strategy = args.strategy

if version == True:
  os.system(esbmc_path + "--version")
  exit(0)

if property_file is None:
  print "Please, specify a property file"
  exit(1)

if benchmark is None:
  print "Please, specify a benchmark to verify"
  exit(1)

# Add arch
if arch == 32:
  command_line += "--32 "
else:
  command_line += "--64 "

# Add strategy
if strategy == "kinduction":
  command_line += esbmc_kind
elif strategy == "fp":
  command_line += esbmc_fp
elif strategy == "falsi":
  command_line += esbmc_falsi
elif strategy == "incr":
  command_line += esbmc_incr
elif strategy == "fixed":
  command_line += esbmc_fixed
else:
  print "Unknown strategy"
  exit(1)

# Parse property files
is_memsafety = False
is_overflow = False
is_reachability = False

f = open(property_file, 'r')
property_file_content = f.read()

if "CHECK( init(main()), LTL(G valid-free) )" in property_file_content:
  is_memsafety = True
elif "CHECK( init(main()), LTL(G ! overflow) )" in property_file_content:
  is_overflow = True
elif "CHECK( init(main()), LTL(G ! call(__VERIFIER_error())) )" in property_file_content:
  is_reachability = True
else:
  # We don't support termination
  print "Unsupported Property"
  exit(1)

if is_overflow:
  command_line += "--overflow-check "
elif is_memsafety:
  command_line += "--memory-leak-check "
elif is_reachability:
  command_line += "--no-pointer-check --no-bounds-check --error-label ERROR "

# Call ESBMC
command_line += benchmark

print "Verifying with ESBMC "
print "Command: " + command_line

args = shlex.split(command_line)

p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
(stdout, stderr) = p.communicate()

""" DEBUG output
print stdout
print stderr

print "\n~~~~~~~~~~~~~~~~"
stdout_split = stdout.split('\n')
for i in range(1, 10):
  print stdout_split[len(stdout_split) -(10-i)]
print "~~~~~~~~~~~~~~~~\n"
"""

# Parse output
if "Timed out" in stdout:
  print "Timed out"
  exit(1)

# Error messages:
memory_leak = "dereference failure: forgotten memory"
invalid_pointer = "dereference failure: invalid pointer"
access_out = "dereference failure: Access to object out of bounds"
dereference_null = "dereference failure: NULL pointer"
invalid_object = "dereference failure: invalidated dynamic object"
invalid_object_free = "dereference failure: invalidated dynamic object freed"
invalid_pointer_free = "dereference failure: invalid pointer freed"
free_error = "dereference failure: free() of non-dynamic memory"
bounds_violated = "array bounds violated"
free_offset = "Operand of free must have zero pointer offset"

if "VERIFICATION FAILED" in stdout:
  if "unwinding assertion loop" in stdout:
    print "UNKNOWN"
    exit(1)

  if is_memsafety:
    if memory_leak in stdout:
      print "FALSE_MEMTRACK"
      exit(0)

    if invalid_pointer_free in stdout:
      print "FALSE_FREE"
      exit(0)

    if invalid_object_free in stdout:
      print "FALSE_FREE"
      exit(0)

    if invalid_pointer in stdout:
      print "FALSE_DEREF"
      exit(0)

    if dereference_null in stdout:
      print "FALSE_DEREF"
      exit(0)

    if free_error in stdout:
      print "FALSE_FREE"
      exit(0)

    if access_out in stdout:
      print "FALSE_DEREF"
      exit(0)

    if invalid_object in stdout:
      print "FALSE_DEREF"
      exit(0)

    if bounds_violated in stdout:
      print "FALSE_DEREF"
      exit(0)

    if free_offset in stdout:
      print "FALSE_FREE"
      exit(0)

  if is_overflow:
    print "FALSE_OVERFLOW"
    exit(0)

  if is_reachability:
    print "FALSE"
    exit(0)

if "VERIFICATION SUCCESSFUL" in stdout:
  print "TRUE"
  exit(0)

print "UNKNOWN"
