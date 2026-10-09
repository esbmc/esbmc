#!/usr/bin/env python3

import os
import argparse
import shlex
import re
import subprocess
import time
import sys
import resource
from hashlib import sha256
import datetime

# Start time for this script
start_time = time.time()
SVCOMP_EXTRA_VERSION = " svcomp 0"

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
  fail_termination = 13
  fail_race = 14
  fail_python = 15

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
    if res == Result.fail_termination:
      return True
    if res == Result.fail_race:
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
  datarace = 6
  py_assertion = 7
  py_arithmetic = 8
  py_datalookup = 9
  py_dyntyping = 10

# sv-benchmarks' python/properties/*.prp each state that one family of runtime
# exceptions does not escape main(). ESBMC has no per-family scoping on the
# Python side -- one run checks them all -- so which family a run falsified is
# read off the violated-property comment instead of assumed from the flags.
PYTHON_EXCEPTION_FAMILIES = {
  Property.py_assertion: ("AssertionError",),
  Property.py_arithmetic: ("ZeroDivisionError", "FloatingPointError"),
  Property.py_datalookup: ("KeyError", "IndexError"),
  Property.py_dyntyping: ("TypeError", "AttributeError"),
}

# Parsed by the CLI entry point at the bottom; do_exec() is the only reader.
args = None

def do_exec(cmd_line):

  if args.dry_run:
    exit(0)

  the_args = shlex.split(cmd_line)

  p = subprocess.Popen(the_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  (stdout, stderr) = p.communicate()

  return stdout + stderr

# Function to run esbmc
def run(cmd_line):
  print("Verifying with ESBMC")
  print("Command: " + cmd_line)
  out = do_exec(cmd_line)
  print(out.decode())
  return out

# A property table row, e.g. "  FAILED       [main.assertion.1]  line 7  x != 0".
PROPERTY_ROW = re.compile(r"\s+(?:\033\[[0-9;]*m)?(FAILED|PASSED|NOT CHECKED|UNKNOWN)\b")

def violated_property_text(the_output):
  """The comments ESBMC attached to the properties it actually violated.

  Since esbmc/esbmc#7064 every run also prints a "** Results:" table naming
  each property it did not violate, so matching a comment against the whole
  output classifies the task from a property that passed or was never checked.
  """
  violated = []
  lines = the_output.splitlines()
  i = 0
  while i < len(lines):
    line = lines[i]
    i += 1
    if line.startswith("Violated property:"):
      while i < len(lines) and lines[i].startswith("  "):
        violated.append(lines[i])
        i += 1
    else:
      row = PROPERTY_ROW.match(line)
      if row and row.group(1) == "FAILED":
        violated.append(line)
  return "\n".join(violated)

MEMORY_LEAK = "dereference failure: forgotten memory"
UNREACHABILITY_INTRINSIC = "reachability: unreachable code reached"

# Ordered: the first comment found in the violated properties decides the
# category, so a task violating several keeps the answer the pre-#7250 chain
# gave. The trailing bare "dereference failure" is the catch-all for comments
# not spelled out above -- "Data object accessed with code type" and memcpy's
# write-side message, for two -- which an always-true operand in that chain
# used to catch by accident, at the price of making the free() checks dead.
MEMORY_VIOLATIONS = (
  (MEMORY_LEAK, Result.fail_memtrack),
  ("dereference failure: invalid pointer freed", Result.fail_free),
  ("dereference failure: invalidated dynamic object freed", Result.fail_free),
  ("dereference failure: accessed expired variable pointer", Result.fail_deref),
  ("dereference failure: invalid pointer", Result.fail_deref),
  ("dereference failure: NULL pointer", Result.fail_deref),
  ("dereference failure: free() of non-dynamic memory", Result.fail_free),
  ("dereference failure: Access to object out of bounds", Result.fail_deref),
  ("dereference failure: memset of memory segment of size", Result.fail_deref),
  ("dereference failure on memcpy: reading memory segment", Result.fail_deref),
  ("dereference failure: invalidated dynamic object", Result.fail_deref),
  ("array bounds violated", Result.fail_deref),
  ("Operand of free must have zero pointer offset", Result.fail_free),
  (" Verifier error called", Result.success),
  ("dereference failure", Result.fail_deref),
)


def classify_memory_violation(violated):
  """Map violated-property text to a memsafety sub-property, or None."""
  for comment, result in MEMORY_VIOLATIONS:
    if comment in violated:
      return result
  return None


UNCAUGHT_EXCEPTION = re.compile(r"uncaught exception: (\w+)")

# A plain `assert` lowers to a property whose comment reads "assertion <expr>",
# and that is the only AssertionError the Python frontend labels as such: a
# `raise AssertionError(msg)` reaches the table as the bare message, which
# nothing distinguishes from an operational model failing internally. Matching
# the bare form too would turn such a failure into a wrong answer on a correct
# program, so it is deliberately left inconclusive: 0 beats -16.
ASSERTION_ROW = re.compile(r"\]\s+line\s+\d+\s+assertion\b")


def classify_python_violation(violated, prop):
  """Does this violation belong to the exception family the property asks about?"""
  family = PYTHON_EXCEPTION_FAMILIES[prop]
  if any(name in family for name in UNCAUGHT_EXCEPTION.findall(violated)):
    return Result.fail_python

  if prop == Property.py_assertion and ASSERTION_ROW.search(violated):
    return Result.fail_python

  return Result.unknown


def add_python_module_path(benchmark):
  """Put sv-benchmarks' _sv_verifier module within the task's import reach.

  A Python task sits in python/<project>/ and imports _sv_verifier, which sits
  one level up in python/. CPython resolves an import against the script's own
  directory and fails the same way, which is why sv-benchmarks' check-syntax.py
  sets PYTHONPATH; ESBMC follows CPython here, so the search path is supplied
  rather than the resolution changed.
  """
  task_dir = os.path.dirname(os.path.abspath(benchmark))
  paths = [task_dir, os.path.dirname(task_dir)]
  if os.environ.get("PYTHONPATH"):
    paths.append(os.environ["PYTHONPATH"])
  os.environ["PYTHONPATH"] = os.pathsep.join(paths)


def python_property(property_file_content):
  """The Python track property this file states, or None for any other file."""
  names = set(re.findall(r"!\s*uncaught\(\s*(\w+)\s*\)", property_file_content))
  if not names:
    return None
  for prop, family in PYTHON_EXCEPTION_FAMILIES.items():
    if names <= set(family):
      return prop
  return None


def classify_violation(violated, prop):
  """What a violated property implies for the property under check, or None.

  None means this violation decides nothing here, which parse_result reads as
  "keep looking" -- a run that also printed SUCCESSFUL is then read as true.
  """
  if "unwinding assertion loop" in violated:
    return Result.err_unwinding_assertion

  if prop in PYTHON_EXCEPTION_FAMILIES:
    return classify_python_violation(violated, prop)

  if prop == Property.memcleanup:
    return Result.fail_memcleanup if MEMORY_LEAK in violated else None

  if prop == Property.termination:
    return Result.fail_termination

  if prop == Property.memory:
    return classify_memory_violation(violated)

  if prop == Property.overflow:
    return Result.fail_overflow

  if prop == Property.reach:
    return None if UNREACHABILITY_INTRINSIC in violated else Result.fail_reach

  if prop == Property.datarace:
    return Result.fail_race

  return None


def parse_result(the_output, prop):
  # ESBMC also prints a "  CWE: CWE-NNN" line after each violated-property
  # comment (see docs/cwe-mapping.md) and may emit a SARIF report under
  # --sarif-output. Both are purely informational; the SV-COMP category is
  # still derived from the unchanged freeform comment strings below, matched
  # against the violated properties rather than the whole output.

  # Parse output
  if "Timed out" in the_output:
    return Result.err_timeout

  if "Out of memory" in the_output:
    return Result.err_memout

  if "Chosen solver doesn\'t support floating-point numbers" in the_output:
    return Result.force_fp_mode

  if "VERIFICATION FAILED" in the_output:
    decided = classify_violation(violated_property_text(the_output), prop)
    if decided is not None:
      return decided

  if "VERIFICATION SUCCESSFUL" in the_output:
    return Result.success

  return Result.unknown

def get_result_string(the_result):
  if the_result == Result.fail_memcleanup:
    return "FALSE_MEMCLEANUP"

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

  if the_result == Result.fail_termination:
    return "FALSE_TERMINATION"

  if the_result == Result.fail_race:
    return "FALSE_DATARACE"

  if the_result == Result.fail_python:
    # BenchExec maps a bare FALSE to false(unreach-call), and a task
    # definition whose expected verdict carries no subproperty accepts any
    # result starting with "false" (benchexec result.get_result_category), so
    # the Python properties need no new result string.
    return "FALSE"

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
# --sv-comp enables SV-COMP mode (replaces the former ESBMC_SVCOMP build): it
# suppresses GCC-acceptable frontend diagnostics, treats __builtin_unreachable
# as a no-op, emits physical line numbers for witnesses, and avoids malloc/free
# in the fopen/fclose models.
esbmc_dargs = "--sv-comp --no-div-by-zero-check --force-malloc-success --force-realloc-success --state-hashing --add-symex-value-sets "
esbmc_dargs += "--no-align-check --k-step 2 --unlimited-k-steps "

# <https://github.com/esbmc/esbmc/pull/1190#issuecomment-1637047028>
esbmc_dargs += "--no-vla-size-check "


def check_if_benchmark_contains_pthread(benchmark):
  with open(benchmark, "r") as f:
    for line in f:
      if re.search("pthread_create", line.strip()):
        return True
  return False

def strategy_flags(strat):
  if strat in ("fixed", "kinduction"):
    return "--k-induction --max-inductive-step 3 "
  if strat == "falsi":
    return "--falsification "
  if strat == "incr":
    return "--incremental-bmc "
  print("Unknown strategy")
  exit(1)


def get_command_line(strat, prop, arch, benchmark, concurrency, dargs, esbmc_ci, validate=False):
  if prop in PYTHON_EXCEPTION_FAMILIES:
    # A Python task is a whole program whose only specification is that an
    # exception family does not escape, so none of the C defaults in dargs
    # apply. --no-div-by-zero-check in particular would switch off the very
    # check the arithmetic property asks about. The track has no witness
    # format yet, so no witness is written.
    #
    # --unlimited-k-steps removes the k=50 ceiling on the base case, so a
    # task whose violation or whose completeness proof needs a loop or
    # recursion depth past 50 is reached rather than answered UNKNOWN
    # (--unwind does not raise that bound). --k-step 2 advances k by two per
    # round, which halves the shallow solver rounds before a deep bound is
    # reached and is a measurable speedup on these tasks with no verdict
    # change. They go here, not in strategy_flags, because the C path
    # already carries --unlimited-k-steps in dargs.
    return (
      esbmc_path + "--sv-comp " + benchmark + " "
      + strategy_flags(strat) + "--k-step 2 --unlimited-k-steps ")

  command_line = esbmc_path + dargs

  # Add benchmark
  command_line += benchmark + " "

  # Add arch
  if arch == 32:
    command_line += "--32 "
  else:
    command_line += "--64 "

  concurrency = ((prop in (Property.reach, Property.datarace, Property.overflow, Property.memory)) and
                 check_if_benchmark_contains_pthread(benchmark))

  if concurrency:
    # --smt-symex-guard also turns on --smt-during-symex, which is what makes
    # sibling schedules share a solver context (issue #6831, #7043); do not add
    # it separately, and do not drop the guard without re-measuring.
    command_line += " --smt-symex-guard --bitwuzla --cswitch-skip-readonly-globals "
    # Unlike --max-context-bound, a truncated schedule space still reports
    # SUCCESSFUL, so this trades soundness of "true" verdicts for coverage.
    command_line += "--context-bound 3 "
    #command_line += "--no-slice " # TODO: Witness validation is only working without slicing

  # Add witness arg
  witness_name = os.path.basename(benchmark) if esbmc_ci else "witness"
  command_line += "--witness-output " + witness_name + " "

  # Special case for termination, it runs regardless of the strategy
  if prop == Property.termination:
    command_line += "--no-pointer-check --no-bounds-check --no-assertions "
    # --interval-analysis strengthens the inductive step: the post-havoc
    # bound pass pins each havoced loop variable (including ones modified
    # only through a callee) to its interval, so IS reasons from the
    # reachable state space instead of an arbitrary havoc. +52 correct-
    # false on the termination set with no new wrong results.
    command_line += "--termination --max-inductive-step 3 --interval-analysis "
    return command_line

  if prop == Property.overflow:
    # SV-COMP's no-overflow property is signed-integer arithmetic only: "Hence,
    # conversions to signed-integer types do not violate this property." A
    # float-to-int conversion out of range is real UB but a different property,
    # and parse_result() below reports any violation in this run as
    # FALSE_OVERFLOW, so the check has to be off here (esbmc #7572).
    command_line += "--no-pointer-check --no-bounds-check --overflow-check --no-assertions --no-fp-conversion-check "
  elif prop == Property.memory:
    command_line += "--memory-leak-check --no-reachable-memory-leak --no-assertions "
    # It seems SV-COMP doesn't want to check for memleaks on abort()
    # see also <https://github.com/esbmc/esbmc/issues/1259>
    command_line += "--no-abnormal-memory-leak "
    # many benchmarks assume malloc(0) == NULL and alloca(0) == NULL
    command_line += "--malloc-zero-is-null "
    strat = "incr"
  elif prop == Property.memcleanup:
    command_line += "--no-pointer-check --no-bounds-check --memory-leak-check --no-assertions "
    strat = "incr"
  elif prop == Property.reach:
    command_line += "--enable-unreachability-intrinsic "
    if concurrency:
      command_line += "--no-pointer-check --no-bounds-check "
    else:
      command_line += "--no-pointer-check --interval-analysis --no-bounds-check --error-label ERROR "
      if not validate:
        command_line += "--goto-unwind --unlimited-goto-unwind "
  elif prop == Property.datarace:
    # TODO: can we do better in case 'concurrency == False'?
    command_line += "--no-pointer-check --no-bounds-check --data-races-check-only --no-assertions "
  else:
    print("Unknown property")
    exit(1)

  # Add strategy
  if concurrency: # Concurrency only works with incremental
    # A violation needing few context switches can sit deep in unbounded DFS
    # order, where the task times out with no answer at all (issue #6831, #6947).
    # One bounded round first costs a median 0.02s and can only report a
    # violation -- it never claims a proof, so --incremental-bmc still owns
    # every other verdict.
    command_line += "--falsify-context-bound 1 --incremental-bmc "
  elif prop == Property.overflow: # Overflow only works with incremental
    command_line += "--incremental-bmc "
  else:
    command_line += strategy_flags(strat)

  return command_line

def verify(strat, prop, arch, benchmark, concurrency, dargs, esbmc_ci, witness_path, validate_mode):
  esbmc_command_line = get_command_line(strat, prop, arch, benchmark, concurrency, dargs, esbmc_ci, validate=bool(witness_path))

  if witness_path:
    esbmc_command_line += "--validate-" + validate_mode + "-witness "
    esbmc_command_line += "--witness " + witness_path + " "

  output = run(esbmc_command_line)
  res = parse_result(output.decode(), prop)
  return res

if __name__ == "__main__":
  # Options
  parser = argparse.ArgumentParser()
  parser.add_argument("-a", "--arch", help="Either 32 or 64 bits", type=int, choices=[32, 64], default=32)
  parser.add_argument("-v", "--version", help="Prints ESBMC's version", action='store_true')
  parser.add_argument("-p", "--propertyfile", help="Path to the property file")
  parser.add_argument("benchmark", nargs='?', help="Path to the benchmark")
  parser.add_argument("-s", "--strategy", help="ESBMC's strategy", choices=["kinduction", "falsi", "incr", "fixed"], default="fixed")
  parser.add_argument("-c", "--concurrency", help="Set concurrency flags", action='store_true')
  parser.add_argument("-n", "--dry-run", help="do not actually run ESBMC, just print the command", action='store_true')
  parser.add_argument("--ci", help="run this wrapper with special options for the CI (internal use)", action='store_true')
  parser.add_argument("--witness", help="Path to witness file; enables witness validation mode")
  parser.add_argument("--validate-violation-witness", dest="validate_violation", action='store_true',
                      help="Validate a violation witness (use with --witness)")
  parser.add_argument("--validate-correctness-witness", dest="validate_correctness", action='store_true',
                      help="Validate a correctness witness (use with --witness)")

  args = parser.parse_args()

  arch = args.arch
  version = args.version
  property_file = args.propertyfile
  benchmark = args.benchmark
  strategy = args.strategy
  concurrency = args.concurrency
  esbmc_ci = args.ci
  witness_path = args.witness

  if version:
    print(do_exec(esbmc_path + "--version").decode()[6:].strip()),
    exit(0)

  if property_file is None:
    print("Please, specify a property file")
    exit(1)

  if benchmark is None:
    print("Please, specify a benchmark to verify")
    exit(1)

  if witness_path and not args.validate_violation and not args.validate_correctness:
    print("Please specify --validate-violation-witness or --validate-correctness-witness when using --witness")
    exit(1)
  validate_mode = "violation" if args.validate_violation else "correctness" if args.validate_correctness else None

  # Parse property files
  f = open(property_file, 'r')
  property_file_content = f.read()

  category_property = 0
  if "CHECK( init(main()), LTL(G valid-free) )" in property_file_content:
    category_property = Property.memory
  elif "CHECK( init(main()), LTL(G ! overflow) )" in property_file_content:
    category_property = Property.overflow
  elif "CHECK( init(main()), LTL(G ! call(reach_error())) )" in property_file_content:
    category_property = Property.reach
  elif "CHECK( init(main()), LTL(F end) )" in property_file_content:
    category_property = Property.termination
  elif "CHECK( init(main()), LTL(G valid-memcleanup) )" in property_file_content:
    category_property = Property.memcleanup
  elif "CHECK( init(main()), LTL(G ! data-race) )" in property_file_content:
    category_property = Property.datarace
  else:
    category_property = python_property(property_file_content)

  if not category_property:
    print("Unsupported Property")
    exit(1)

  if category_property in PYTHON_EXCEPTION_FAMILIES:
    add_python_module_path(benchmark)

  result = verify(strategy, category_property, arch, benchmark, concurrency, esbmc_dargs, esbmc_ci, witness_path, validate_mode)

  print(get_result_string(result))
