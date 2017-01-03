#!/usr/bin/env python

import os
import argparse
import shlex
import subprocess
import time
import sys

# Start time for this script
start_time = time.time()

class Result:
  success = 1
  fail_deref = 2
  fail_memtrack = 3
  fail_free = 4
  fail_reach = 5
  fail_overflow = 6
  err_timeout = 7
  err_memout = 8
  err_unwinding_assertion = 9
  force_fp_mode = 10
  unknown = 11

  @staticmethod
  def is_fail(res):
    if res == Result.fail_deref:
      return True
    if res == Result.fail_free:
      return True
    if res == Result.fail_memtrack:
      return True
    if res == Result.fail_overflow:
      return True
    if res == Result.fail_reach:
      return True
    return False

  @staticmethod
  def is_out(res):
    if res == Result.err_memout:
      return True
    if res == Result.err_timeout:
      return True
    if res == Result.unknown:
      return True
    return False

class Property:
  reach = 1
  memory = 2
  overflow = 3
  termination = 4

class Unwindings:
  loops = [1, 64, 160]
  overflow = [1, 2, 32778]

# Function to run esbmc
def run_esbmc(cmd_line):
  print "Verifying with ESBMC "
  print "Command: " + cmd_line

  the_args = shlex.split(cmd_line)

  p = subprocess.Popen(the_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  (stdout, stderr) = p.communicate()

  """ DEBUG output
  print stdout
  print stderr
  """

  return stdout

def parse_result(the_output, prop):

  # Parse output
  if "Timed out" in the_output:
    return Result.err_timeout

  if "Out of memory" in the_output:
    return Result.err_memout

  if "try with Z3 or Mathsat" in the_output:
    return Result.force_fp_mode

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

  if "VERIFICATION FAILED" in the_output:
    if "unwinding assertion loop" in the_output:
      return Result.err_unwinding_assertion

    if prop == Property.memory:
      if memory_leak in the_output:
        return Result.fail_memtrack

      if invalid_pointer_free in the_output:
        return Result.fail_free

      if invalid_object_free in the_output:
        return Result.fail_free

      if invalid_pointer in the_output:
        return Result.fail_deref

      if dereference_null in the_output:
        return Result.fail_deref

      if free_error in the_output:
        return Result.fail_free

      if access_out in the_output:
        return Result.fail_deref

      if invalid_object in the_output:
        return Result.fail_deref

      if bounds_violated in the_output:
        return Result.fail_deref

      if free_offset in the_output:
        return Result.fail_free

      if " Verifier error called" in the_output:
        return Result.success

    if prop == Property.overflow:
      return Result.fail_overflow

    if prop == Property.reach:
      return Result.fail_reach

  if "VERIFICATION SUCCESSFUL" in the_output:
    return Result.success

  return Result.unknown

def get_result_string(the_result):
  if the_result == Result.fail_memtrack:
    return "FALSE_MEMTRACK"

  if the_result == Result.fail_free:
    return "FALSE_FREE"

  if the_result == Result.fail_deref:
    return "FALSE_DEREF"

  if the_result == Result.fail_overflow:
    return "FALSE_OVERFLOW"

  if the_result == Result.fail_reach:
    return "FALSE_REACH"

  if the_result == Result.success:
    return "TRUE"

  if the_result == Result.err_timeout:
    return "Timed out"

  if the_result == Result.err_unwinding_assertion:
    return "Unknown"

  if the_result == Result.err_memout:
    return "Unknown"

  if the_result == Result.unknown:
    return "Unknown"

  exit(0)

# strings
esbmc_path = "./esbmc "

# ESBMC default commands: this is the same for every submission
esbmc_dargs = "--no-div-by-zero-check --force-malloc-success --context-bound 7 "
esbmc_dargs += "--clang-frontend --state-hashing -Dldv_assume=__ESBMC_assume "

def get_command_line(strat, prop, arch, benchmark, fp_mode):
  command_line = esbmc_path + esbmc_dargs

  # Add witness arg
  command_line += "--witness-output " + os.path.basename(benchmark) + ".graphml "

  # Add strategy
  if strat == "kinduction":
    command_line += "--floatbv --unlimited-k-steps --z3 --k-induction "
  elif strat == "falsi":
    command_line += "--floatbv --unlimited-k-steps --z3 --falsification "
  elif strat == "incr":
    command_line += "--floatbv --unlimited-k-steps --z3 --incremental-bmc "
  elif strat == "fixed":
    command_line += "--floatbv --unroll-loops --no-unwinding-assertions --unwind 1 "
    if fp_mode: command_line += "--mathsat "
  else:
    print "Unknown strategy"
    exit(1)

  # Add arch
  if arch == 32:
    command_line += "--32 "
  else:
    command_line += "--64 "

  if prop == Property.overflow:
    command_line += "--no-pointer-check --no-bounds-check --overflow-check -D__VERIFIER_error=ESBMC_error "
  elif prop == Property.memory:
    command_line += "--memory-leak-check -D__VERIFIER_error=ESBMC_error "
  elif prop == Property.reach:
    command_line += "--no-pointer-check --no-bounds-check --error-label ERROR "

  # Benchmark
  command_line += benchmark
  return command_line

def verify(strat, prop):
  # Get command line
  esbmc_command_line = get_command_line(strat, prop, arch, benchmark, False)

  # Call ESBMC
  output = run_esbmc(esbmc_command_line)

  # Parse output
  result = parse_result(output, category_property)

  # We'll only recheck the fixed approach
  if strat != "fixed":
    return result

  # Retry in fp_mode?
  fp_mode = (result == Result.force_fp_mode)

  # We'll retry a number of times, however, the verification with
  # unwind 1 for forced fp mode failed, so we try again with 1 unwind
  retry = 1 - fp_mode

  unwinds = Unwindings.overflow if prop == Property.overflow else Unwindings.loops

  while retry != len(unwinds):
    # The new command is incomplete
    new_command_line = get_command_line(strategy, category_property, arch, benchmark, fp_mode)

    # Add memory out and timeout
    new_command_line += " --memlimit 13g --timeout "
    new_command_line += str(895 - (int) (round(time.time() - start_time)))

    # Replace unwind
    new_command_line = new_command_line.replace("unwind 1", "unwind " + str(unwinds[retry]))

    # If verifying overflows, abort on recursion
    if prop == Property.overflow:
      new_command_line += " --abort-on-recursion"

    # Run esbmc
    new_output = run_esbmc(new_command_line)
    new_result = parse_result(new_output, category_property)

    # If the result is either timeout or memory out, we give up
    if Result.is_out(new_result):
      break

    # Always keep the last _correct_ result
    result = new_result

    # retry next time with a bigger unwind
    retry += 1

  # Keep the previous result
  return result, fp_mode

def needs_validation(strat, prop, result, fp_mode):
  # If we're forcing floating-point mode, don't validate
  if fp_mode:
    return False

  # We only validate for fixed + reachability + false result
  if result == Result.fail_reach and strat == "fixed" and prop == Property.reach:
    return True

def get_cpa_command_line(prop, benchmark):
  command_line = "./scripts/cpa.sh -witness-validation "
  command_line += "-spec ../" + os.path.basename(benchmark) + ".graphml "
  command_line += " -spec " + prop + " "
  command_line += benchmark
  return command_line

def run_cpa(cmd_line):
  # The default result is to confirm the witness
  stdout = "VERIFICATION RESULT: FALSE"

  try:
    # Save current dir
    cwd = os.getcwd()

    # Change to CPA's dir
    os.chdir("./cpachecker/")

    # Checking if there is still enough time available
    elapsed_time = (int) (round(time.time() - start_time))
    remaining_time = 895 - elapsed_time

    if (remaining_time > 0):
      # Update CPA with timeout
      cmd_line += " -timelimit " + str(remaining_time) + "s"

      print "Verifying with CPA "
      print "Command: " + cmd_line

      the_args = shlex.split(cmd_line)

      p = subprocess.Popen(the_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      (stdout, stderr) = p.communicate()

      """ DEBUG output
      print stdout
      print stderr
      """

    # restore dir
    os.chdir(cwd)
  except:
    print("Unexpected error:", sys.exc_info()[0])

  return stdout

def parse_cpa_result(result):
  if "Verification result: FALSE" in result:
    return Result.fail_reach

  if "Verification result: TRUE" in result:
    return Result.success

  return Result.unknown

# Options
parser = argparse.ArgumentParser()
parser.add_argument("-a", "--arch", help="Either 32 or 64 bits", type=int, choices=[32, 64], default=32)
parser.add_argument("-v", "--version", help="Prints ESBMC's version", action='store_true')
parser.add_argument("-p", "--propertyfile", help="Path to the property file")
parser.add_argument("benchmark", nargs='?', help="Path to the benchmark")
parser.add_argument("-s", "--strategy", help="ESBMC's strategy", choices=["kinduction", "falsi", "incr", "fixed"], default="incr")

args = parser.parse_args()

arch = args.arch
version = args.version
property_file = args.propertyfile
benchmark = args.benchmark
strategy = args.strategy

if version:
  os.system(esbmc_path + "--version")
  exit(0)

if property_file is None:
  print "Please, specify a property file"
  exit(1)

if benchmark is None:
  print "Please, specify a benchmark to verify"
  exit(1)

# Parse property files
f = open(property_file, 'r')
property_file_content = f.read()

category_property = 0
if "CHECK( init(main()), LTL(G valid-free) )" in property_file_content:
  category_property = Property.memory
elif "CHECK( init(main()), LTL(G ! overflow) )" in property_file_content:
  category_property = Property.overflow
elif "CHECK( init(main()), LTL(G ! call(__VERIFIER_error())) )" in property_file_content:
  category_property = Property.reach
else:
  # We don't support termination
  print "Unsupported Property"
  exit(1)

result, fp_mode = verify(strategy, category_property)

# Check if we're going to validate the results
if needs_validation(strategy, category_property, result, fp_mode):
  cpa_command_line = get_cpa_command_line(property_file, benchmark)
  output = run_cpa(cpa_command_line)

  # If we found a property violation but CPA said that the path is not reachable
  # we ignore the result and present UNKNOWN. If CPA fails or confirms the result
  # we return FALSE
  if parse_cpa_result(output) == Result.success:
    result = Result.unknown

print get_result_string(result)
