#!/usr/bin/env python3

import os.path  # To check if file exists
import xml.etree.ElementTree as ET  # To parse XML
import os
import argparse
import shlex
import subprocess
import time
import sys
import resource
import re

# Start time for this script
start_time = time.time()
property_file_content = ""


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
    fail_memcleanup = 12

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
        if res == Result.fail_memcleanup:
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
    memcleanup = 5
    coverage = 6

# Function to run esbmc
def run(cmd_line):
  print("Verifying with ESBMC")
  print("Command: " + cmd_line)

  the_args = shlex.split(cmd_line)

  p = subprocess.Popen(the_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  (stdout, stderr) = p.communicate()

  print(stdout.decode())
  print(stderr.decode())

  return stdout


def parse_result(the_output, prop):

    # Parse output
    if "Timed out" in the_output:
        return Result.err_timeout

    if "Out of memory" in the_output:
        return Result.err_memout

    if "Chosen solver doesn\'t support floating-point numbers" in the_output:
        return Result.force_fp_mode

    # Error messages:
    if "VERIFICATION FAILED" in the_output:
        if "unwinding assertion loop" in the_output:
            return Result.err_unwinding_assertion

        if prop == Property.reach:
            return Result.fail_reach

    if "VERIFICATION SUCCESSFUL" in the_output:
        return Result.success

    return Result.unknown


def get_result_string(the_result):
    if the_result == Result.fail_reach:
        return "DONE"

    if the_result == Result.success:
        return "DONE"

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
esbmc_dargs = "--no-div-by-zero-check --force-malloc-success --goto-unwind "
esbmc_dargs += "--no-align-check --k-step 5 --floatbv --unlimited-k-steps "
esbmc_dargs += "--no-pointer-check --no-bounds-check --no-pointer-check "
# TODO: interval-analysis, contractor and pointer-check (problem with array index wrap)

def get_command_line(strat, prop, arch, benchmark, fp_mode):
    command_line = esbmc_path + esbmc_dargs

    # Add benchmark
    command_line += benchmark + " "

    # Add arch
    if arch == 32:
        command_line += "--32 "
    else:
        command_line += "--64 "

    # Add Test Case Arg
    command_line += "--generate-testcase "

    # Add strategy
    if strat == "kinduction":
        command_line += "--bidirectional "
    elif strat == "falsi":
        command_line += "--falsification "
    elif strat == "incr":
        command_line += "--incremental-bmc "
    else:
        print ("Unknown strategy")
        exit(1)

    if prop == Property.coverage:
        # TODO: Parallel solving?
        command_line += "--multi-property "
    if prop != Property.reach and prop != Property.coverage :
        print ("Unknown property")
        exit(1)

    return command_line


def verify(strat, prop, fp_mode):
    # Get command line
    esbmc_command_line = get_command_line(
        strat, prop, arch, benchmark, fp_mode)

    # Call ESBMC
    output = run(esbmc_command_line)

    res = parse_result(output, category_property)
    if(res == Result.force_fp_mode):
        return verify(strat, prop, True)

    # Parse output
    return res


# Options
parser = argparse.ArgumentParser()
parser.add_argument("-a", "--arch", help="Either 32 or 64 bits",
                    type=int, choices=[32, 64], default=32)
parser.add_argument("-v", "--version",
                    help="Prints ESBMC's version", action='store_true')
parser.add_argument("-p", "--propertyfile", help="Path to the property file")
parser.add_argument("benchmark", nargs='?', help="Path to the benchmark")
parser.add_argument("-s", "--strategy", help="ESBMC's strategy",
                    choices=["kinduction", "falsi", "incr"], default="incr")

args = parser.parse_args()

arch = args.arch
version = args.version
property_file = args.propertyfile
benchmark = args.benchmark
strategy = args.strategy

if version:
    print (os.popen(esbmc_path + "--version").read()[6:]),
    exit(0)

if property_file is None:
    print ("Please, specify a property file")
    exit(1)

if benchmark is None:
    print ("Please, specify a benchmark to verify")
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
elif "CHECK( init(main()), LTL(F end) )" in property_file_content:
    category_property = Property.termination
elif "COVER( init(main()), FQL(COVER EDGES(@CALL(reach_error))) )" in property_file_content:
    category_property = Property.reach
elif "COVER( init(main()), FQL(COVER EDGES(@DECISIONEDGE)) )" in property_file_content:
    category_property = Property.coverage
else:
    print ("Unsupported Property")
    exit(1)

result = verify(strategy, category_property, False)
print (get_result_string(result))
