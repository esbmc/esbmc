#!/usr/bin/env python

import os
import argparse
import shlex
import subprocess

class Result:
  err_timeout = 1
  err_unwinding_assertion = 2
  success = 3
  fail_deref = 4
  fail_memtrack = 5
  fail_free = 6
  fail_reach = 7
  fail_overflow = 8
  unknown = 9

# Function to run esbmc
def run_esbmc(command_line):
  print "Verifying with ESBMC "
  print "Command: " + command_line

  args = shlex.split(command_line)

  p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  (stdout, stderr) = p.communicate()

  print stdout
  """ DEBUG output
  print stderr
  """

  return stdout

def parse_result(output):

  # Parse output
  if "Timed out" in output:
    return Result.err_timeout

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

  if "VERIFICATION FAILED" in output:
    if "unwinding assertion loop" in output:
      return Result.err_unwinding_assertion;

    if is_memsafety:
      if memory_leak in output:
        return Result.fail_memtrack

      if invalid_pointer_free in output:
        return Result.fail_free

      if invalid_object_free in output:
        return Result.fail_free

      if invalid_pointer in output:
        return Result.fail_deref

      if dereference_null in output:
        return Result.fail_deref

      if free_error in output:
        return Result.fail_free

      if access_out in output:
        return Result.fail_deref

      if invalid_object in output:
        return Result.fail_deref

      if bounds_violated in output:
        return Result.fail_deref

      if free_offset in output:
        return Result.fail_free

    if is_overflow:
      return Result.fail_overflow

    if is_reachability:
      return Result.fail_reach

  if "VERIFICATION SUCCESSFUL" in output:
    return Result.success

  return Result.unknown

def get_result_string(result):
  if result == Result.err_timeout:
    return "Timed out"

  if result == Result.err_unwinding_assertion:
    return "Unknown"

  if result == Result.fail_memtrack:
    return "FALSE_MEMTRACK"

  if result == Result.fail_free:
    return "FALSE_FREE"

  if result == Result.fail_deref:
    return "FALSE_DEREF"

  if result == Result.fail_overflow:
    return "FALSE_OVERFLOW"

  if result == Result.fail_reach:
    return "FALSE_REACH"

  if result == Result.success:
    return "TRUE"

  if result == Result.unknown:
    return "Unknown"

  exit(0)

# strings
esbmc_path = "./esbmc "
witness_path = "error-witness.graphml "

# ESBMC default commands: this is the same for every submission
esbmc_dargs = "--no-div-by-zero-check --force-malloc-success --context-bound 7 "
esbmc_dargs += "--clang-frontend "
esbmc_dargs += "--witness-output " + witness_path

# ESBMC specific commands: this is different for every submission
esbmc_fp    = "--floatbv --mathsat --no-bitfields --unwind 1 --no-unwinding-assertions "
esbmc_kind  = "--floatbv --unlimited-k-steps --z3 --k-induction-parallel "
esbmc_falsi = "--floatbv --unlimited-k-steps --z3 --falsification "
esbmc_incr  = "--floatbv --unlimited-k-steps --z3 --incremental-bmc  "
esbmc_fixed = "--unroll-loops --unwind 160 --no-unwinding-assertions --boolector "

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

command_line += benchmark + " "

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
output = run_esbmc(command_line)

# Parse output
result = parse_result(output)
result_string = get_result_string(result)

print result_string

