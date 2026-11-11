#!/usr/bin/env python3

import os
import argparse
import glob
import json
import shlex
import re
import shutil
import subprocess
import tempfile
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
  err_crash = 10
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
    if res == Result.err_crash:
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
# exceptions does not escape main(). --python-check-exceptions scopes a run to
# that family, and which family a run falsified is still read off the
# violated-property comment instead of assumed from the flags.
PYTHON_EXCEPTION_FAMILIES = {
  Property.py_assertion: ("AssertionError",),
  Property.py_arithmetic: ("ZeroDivisionError", "FloatingPointError"),
  Property.py_datalookup: ("KeyError", "IndexError"),
  Property.py_dyntyping: ("TypeError", "AttributeError"),
}

# Parsed by the CLI entry point at the bottom; do_exec() is the only reader.
args = None

# ESBMC's exit status for an uncaught exception (sysexits.h EX_SOFTWARE).
ESBMC_INTERNAL_ERROR = 70


def abnormal_exit_line(returncode):
  if returncode < 0:
    return "ERROR: ESBMC exited abnormally (signal %d)\n" % -returncode
  if returncode == ESBMC_INTERNAL_ERROR:
    return "ERROR: ESBMC exited abnormally (exit code %d)\n" % returncode
  return ""

def do_exec(cmd_line, timeout=None, cwd=None):

  if args.dry_run:
    exit(0)

  the_args = shlex.split(cmd_line)

  p = subprocess.Popen(the_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
  try:
    (stdout, stderr) = p.communicate(timeout=timeout)
  except subprocess.TimeoutExpired:
    p.kill()
    p.communicate()
    return b"ERROR: Timed out\n"

  return stdout + stderr + abnormal_exit_line(p.returncode).encode()

# Function to run esbmc
def run(cmd_line, timeout=None):
  print("Verifying with ESBMC")
  print("Command: " + cmd_line)
  out = do_exec(cmd_line, timeout)
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

FIRST_VIOLATION = re.compile(r"^Violated property:\n((?:  .*\n?)+)", re.M)


def first_violation_text(the_output):
  """The property the counterexample violated first, or "" if none."""
  first = FIRST_VIOLATION.search(the_output)
  return first.group(1) if first else ""

VIOLATION_SITE = re.compile(r"Violated property:\n\s+file (\S+)(?: line (\d+))?")


def violation_site(the_output):
  """(file, line) of the first violated property, or None. The line is None
  for an uncaught exception, which ESBMC reports without one."""
  site = VIOLATION_SITE.search(the_output)
  if site is None:
    return None
  return os.path.basename(site.group(1)), site.group(2)


# A table header names a file, with ", function f" unless it heads the
# module-level properties. A row pads its line number to three columns.
ROW_FILE = re.compile(r"^(\S+)(?:, function .+)?$")
ROW_LINE = re.compile(r"\[[^\]]*\]\s+line\s+(\d+)\b")


def failed_sites(the_output):
  """{(file, line)} of every FAILED row of a --multi-property results table."""
  sites = set()
  current = None
  for line in the_output.splitlines():
    header = ROW_FILE.match(line)
    if header:
      current = os.path.basename(header.group(1))
      continue
    row = PROPERTY_ROW.match(line)
    if current and row and row.group(1) == "FAILED":
      at = ROW_LINE.search(line)
      sites.add((current, at.group(1) if at else None))
  return sites


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


# The scalar model raises a TypeError on a dynamically-typed operand as an
# assertion in scalar.c, so the comment is the bare message, not "uncaught
# exception: TypeError". A try/except TypeError around the operation still
# catches it. Only these operand checks count: list.c's "object of this type
# has no len()" also fails inside the model on correct programs.
MODEL_TYPEERROR = re.compile(
  r"TypeError: (?:unsupported operand type\(s\)|comparison not supported"
  r"|can only concatenate str)")


# An operational model raises an exception as an assertion whose comment
# starts with the exception's name, as list.c's "IndexError: out-of-bounds read
# in list" does.
MODEL_EXCEPTION = re.compile(r"\b(\w+Error): ")


class PythonFailure(int):
  """Result.fail_python that names the one exception of the property's family
  the violation reports, for an expected verdict such as false(IndexError)."""

  def __new__(cls, exception):
    failure = super().__new__(cls, Result.fail_python)
    failure.exception = exception
    return failure


def python_failure(violated, family):
  names = set(UNCAUGHT_EXCEPTION.findall(violated) + MODEL_EXCEPTION.findall(violated))
  names &= set(family)
  return PythonFailure(names.pop()) if len(names) == 1 else Result.fail_python


def classify_python_violation(violated, prop):
  """Does this violation belong to the exception family the property asks about?"""
  family = PYTHON_EXCEPTION_FAMILIES[prop]
  if any(name in family for name in UNCAUGHT_EXCEPTION.findall(violated)):
    return python_failure(violated, family)

  if "TypeError" in family and MODEL_TYPEERROR.search(violated):
    return python_failure(violated, family)

  if prop == Property.py_assertion and ASSERTION_ROW.search(violated):
    return Result.fail_python

  return Result.unknown


def add_python_module_path(benchmark):
  """Put sv-benchmarks' _sv_verifier module within the task's import reach.

  A Python task sits in python/<project>/, or a directory below it, and imports
  _sv_verifier, which sits in python/. CPython resolves an import against the
  script's own directory and fails the same way, which is why sv-benchmarks'
  check-syntax.py sets PYTHONPATH; ESBMC follows CPython here, so the search
  path is supplied rather than the resolution changed.
  """
  task_dir = os.path.dirname(os.path.abspath(benchmark))
  paths = [task_dir, os.path.dirname(task_dir)]
  # A task may sit deeper, as python/<project>/<task>/main.py does, so the
  # nearest ancestor above the parent holding _sv_verifier.py is added too.
  ancestor = paths[-1]
  while os.path.dirname(ancestor) != ancestor:
    ancestor = os.path.dirname(ancestor)
    if os.path.isfile(os.path.join(ancestor, "_sv_verifier.py")):
      paths.append(ancestor)
      break
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


COUNTEREXAMPLE = re.compile(r"^\[Counterexample\]$.*?(?=^Violated property:$)",
                            re.MULTILINE | re.DOTALL)


def parse_result(the_output, prop):
  # ESBMC also prints a "  CWE: CWE-NNN" line after each violated-property
  # comment (see docs/cwe-mapping.md) and may emit a SARIF report under
  # --sarif-output. Both are purely informational; the SV-COMP category is
  # still derived from the unchanged freeform comment strings below, matched
  # against the violated properties rather than the whole output.

  # A counterexample carries the program's own output, which may repeat any
  # of ESBMC's messages (esbmc/esbmc#9890).
  the_output = COUNTEREXAMPLE.sub("", the_output)

  if re.search(r"^ERROR: Timed out$", the_output, re.MULTILINE):
    return Result.err_timeout

  if "VERIFICATION FAILED" in the_output:
    # The model runs on past its first violation and fails later claims too,
    # such as a leak at exit after an invalid dereference. The program stopped
    # at the first one, so that one names the sub-property (#9895).
    if prop == Property.memory:
      violated = first_violation_text(the_output)
    else:
      violated = violated_property_text(the_output)
    decided = classify_violation(violated, prop)
    if decided is not None:
      return decided

  if "VERIFICATION SUCCESSFUL" in the_output:
    return Result.success

  # A k-induction step that runs out of memory is unknown and the next k is
  # tried, so only a run that reached no verdict ran out of memory.
  if re.search(r"^ERROR: Out of memory$", the_output, re.MULTILINE):
    return Result.err_memout

  # benchexec reports a run whose output has no verdict and no "Unknown" line
  # as ERROR, so a crash is not counted as unknown (#8097).
  if re.search(r"^ERROR: ESBMC exited abnormally", the_output, re.MULTILINE):
    return Result.err_crash

  return Result.unknown

def python_false_string(the_result):
  exception = getattr(the_result, "exception", None)
  return "FALSE(%s)" % exception if exception else "FALSE"


RESULT_STRINGS = {
  Result.fail_memcleanup: "FALSE_MEMCLEANUP",
  Result.fail_memtrack: "FALSE_MEMTRACK",
  Result.fail_free: "FALSE_FREE",
  Result.fail_deref: "FALSE_DEREF",
  Result.fail_overflow: "FALSE_OVERFLOW",
  Result.fail_reach: "FALSE_REACH",
  Result.fail_termination: "FALSE_TERMINATION",
  Result.fail_race: "FALSE_DATARACE",
  Result.success: "TRUE",
  Result.err_timeout: "Timed out",
  Result.err_unwinding_assertion: "Unknown",
  Result.err_memout: "Unknown",
  Result.err_crash: "ERROR",
  Result.unknown: "Unknown",
}


def get_result_string(the_result):
  if the_result == Result.fail_python:
    # A task definition whose expected verdict carries no subproperty accepts
    # any result starting with "false" (benchexec result.get_result_category).
    # One with a subproperty accepts only false(<subproperty>), so a FALSE
    # names its exception when the violation reports exactly one.
    return python_false_string(the_result)

  if the_result in RESULT_STRINGS:
    return RESULT_STRINGS[the_result]

  exit(0)

# strings
esbmc_path = "./esbmc "

# ESBMC default commands: this is the same for every submission
# --sv-comp enables SV-COMP mode (replaces the former ESBMC_SVCOMP build): it
# suppresses GCC-acceptable frontend diagnostics, treats __builtin_unreachable
# as a no-op, emits physical line numbers for witnesses, and avoids malloc/free
# in the fopen/fclose models.
# No --force-malloc-success / --force-realloc-success: SV-COMP lets an
# allocation fail (sv-benchmarks !1686).
esbmc_dargs = "--sv-comp --no-div-by-zero-check --state-hashing --add-symex-value-sets "
esbmc_dargs += "--no-align-check --k-step 2 --unlimited-k-steps "

# <https://github.com/esbmc/esbmc/pull/1190#issuecomment-1637047028>
esbmc_dargs += "--no-vla-size-check "


def check_if_benchmark_contains_pthread(benchmark):
  # A call, not a declaration: goblint-coreutils declare pthread_create
  # (`extern int (...) pthread_create)(...)`) and never start a thread.
  call = re.compile(r"\bpthread_create\s*\(")
  with open(benchmark, "r") as f:
    return any(call.search(line) and not line.lstrip().startswith("extern")
               for line in f)

def strategy_flags(strat):
  if strat in ("fixed", "kinduction"):
    return "--k-induction --max-inductive-step 3 "
  if strat == "falsi":
    return "--falsification "
  if strat == "incr":
    return "--incremental-bmc "
  print("Unknown strategy")
  exit(1)


def python_exception_flags(prop):
  """Report only the property's own exception family. Any other uncaught
  exception ends its path unreported, as CPython stops there, and the
  program's asserts become assumptions unless AssertionError is the family,
  so a failure outside the family no longer hides the one asked about."""
  return "--python-check-exceptions " + ",".join(PYTHON_EXCEPTION_FAMILIES[prop]) + " "


def get_command_line(strat, prop, arch, benchmark, dargs, esbmc_ci, keep_loops=False):
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
      + strategy_flags(strat) + "--k-step 2 --unlimited-k-steps "
      + python_exception_flags(prop))

  command_line = esbmc_path + dargs

  # Add benchmark
  command_line += benchmark + " "

  # Add arch
  if arch == 32:
    command_line += "--32 "
  else:
    command_line += "--64 "

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
    # FALSE_OVERFLOW, so the check has to be off here (esbmc #7572). The same
    # holds for floating-point arithmetic overflow (esbmc #10173).
    command_line += "--no-pointer-check --no-bounds-check --overflow-check --no-assertions --no-fp-conversion-check --no-fp-overflow-check "
  elif prop == Property.memory:
    command_line += "--memory-leak-check --no-reachable-memory-leak --no-assertions "
    # It seems SV-COMP doesn't want to check for memleaks on abort()
    # see also <https://github.com/esbmc/esbmc/issues/1259>
    command_line += "--no-abnormal-memory-leak "
    # many benchmarks assume malloc(0) == NULL and alloca(0) == NULL
    command_line += "--malloc-zero-is-null "
  elif prop == Property.memcleanup:
    command_line += "--no-pointer-check --no-bounds-check --memory-leak-check --no-assertions "
  elif prop == Property.reach:
    command_line += "--enable-unreachability-intrinsic "
    command_line += "--no-pointer-check --interval-analysis --no-bounds-check --error-label ERROR "
    if not keep_loops:
      command_line += "--goto-unwind --unlimited-goto-unwind "
  elif prop == Property.datarace:
    command_line += "--no-pointer-check --no-bounds-check --data-races-check-only --no-assertions "
  else:
    print("Unknown property")
    exit(1)

  # Add strategy
  command_line += strategy_flags(strat)

  return command_line

# Python ints are unbounded but ESBMC's default encoding wraps them at 64 bits
# (esbmc/esbmc#8892), so a violation only the wrapped encoding reaches is false.
# Bitwuzla does not support the integer encoding, hence --z3.
IR_RERUN_TIMEOUT = 120
# Keep both runs together under the competition's 900 s limit.
TOTAL_BUDGET = 870
MIN_RERUN_BUDGET = 30
# Wall-clock seconds and memory cap for the k-induction round of a no-overflow
# task.
KIND_OVERFLOW_TIMEOUT = 60
KIND_OVERFLOW_MEMLIMIT = "2500m"
# Bounds for the inductive step's havoced loop variables. Assuming the overflow
# checks keeps a counter's bound when widening sends the other one to infinity.
KIND_OVERFLOW_INTERVALS = ("--interval-analysis --interval-analysis-arithmetic "
                           "--interval-analysis-extrapolate "
                           "--interval-analysis-extrapolate-under-approximate "
                           "--interval-analysis-assume-asserts ")


def confirmed_by_integer_encoding(esbmc_command_line, the_output, prop, elapsed):
  """Does a --ir re-run violate the same property at the same file and line?

  A plain re-run confirms when it stops at the first run's site. Under
  k-induction a --multi-property re-run goes on raising k for every other
  claim after it has found that site, so it runs only when the plain re-run
  stops elsewhere: it checks every property, so it confirms the first run's
  site even when the plain re-run stopped at another violation first (#9803).

  Both re-runs share one budget, and one that would get too little of it is
  not made, so the violation is not confirmed.
  """
  budget = min(IR_RERUN_TIMEOUT, TOTAL_BUDGET - elapsed)
  if budget < MIN_RERUN_BUDGET:
    return False
  site = violation_site(the_output)
  if site is None:
    return False
  started = time.monotonic()
  plain = run(esbmc_command_line + "--ir --z3 ", budget).decode()
  plain_result = parse_result(plain, prop)
  if plain_result == Result.fail_python and violation_site(plain) == site:
    return True
  budget -= time.monotonic() - started
  if plain_result == Result.success or budget < MIN_RERUN_BUDGET:
    return False
  rerun = run(esbmc_command_line + "--ir --z3 --multi-property ", budget).decode()
  if parse_result(rerun, prop) != Result.fail_python:
    return False
  return violation_site(rerun) == site or site in failed_sites(rerun)


# CPython replay of ESBMC's counterexample. The wrapper asks ESBMC for the
# values of the nondet_* calls on the failing path, replays them in CPython
# through a stand-in _sv_verifier module and confirms the violation when the
# program ends with an exception of the property's family. One real execution
# whose every nondet call returns a value in its domain is a proof, whatever the
# counterexample looked like, so a replay can add a confirmation but never a
# verdict. Anything else leaves the verdict to the --ir re-run. The stand-in and
# its launcher are embedded at the end of this section, so the wrapper stays one
# file; they are written to a temporary directory for each replay.
REPLAY_BUDGET = 20
VALUES_RUN_TIMEOUT = 60
MIN_REPLAY_BUDGET = 2
REPLAY_MEMORY = 2 * 1024 ** 3
REPLAY_FILE_SIZE = 200 * 1024 ** 2
REPLAY_MAX_RUNS = 32
REPLAY_MAX_BITS = 4
REPLAY_MAX_COPY = 50 * 1024 ** 2
VALUES_FLAGS = "--generate-pytest-testcase --pytest-values-only --pytest-output-dir "
NO_VALUES = "witness = [\n    {},\n]\n"


def values_flags(values_dir):
  return VALUES_FLAGS + shlex.quote(values_dir) + " "


def task_provides_verifier(benchmark):
  """Does the task, or a directory above it, hold its own _sv_verifier.py?
  Its nondet_* calls then cannot be fed the counterexample's values."""
  directory = os.path.dirname(os.path.abspath(benchmark))
  while True:
    if os.path.isfile(os.path.join(directory, "_sv_verifier.py")):
      return True
    parent = os.path.dirname(directory)
    if parent == directory:
      return False
    directory = parent


def copy_task(benchmark, destination):
  """A copy of the task's directory, so the replay writes nowhere else."""
  source = os.path.dirname(os.path.abspath(benchmark))
  size = 0
  for root, _, files in os.walk(source):
    size += sum(os.path.getsize(os.path.join(root, f)) for f in files if os.path.isfile(os.path.join(root, f)))
    if size > REPLAY_MAX_COPY:
      return None
  copy = os.path.join(destination, "task")
  shutil.copytree(source, copy, ignore=shutil.ignore_patterns("__pycache__"))
  return os.path.join(copy, os.path.basename(benchmark))


def replay_limits():
  """Cap the replay's memory and file size; a limit the system refuses (macOS
  has no address-space limit) leaves that cap off, the timeout still applies."""
  for kind, size in ((resource.RLIMIT_AS, REPLAY_MEMORY), (resource.RLIMIT_FSIZE, REPLAY_FILE_SIZE)):
    try:
      resource.setrlimit(kind, (size, size))
    except (ValueError, OSError):
      pass
  os.nice(10)


def replay_once(work, replay_dir, program, witness, bits, fill, timeout):
  """One CPython run of the program; the launcher's account of its ending, or
  None when the run did not end by itself or left no account."""
  result = os.path.join(work, "result.json")
  if os.path.exists(result):
    os.remove(result)
  python = [sys.executable, "-s"] + (["-P"] if sys.version_info >= (3, 11) else [])
  env = {"PATH": "/usr/bin:/bin", "HOME": work, "LANG": "C.UTF-8", "PYTHONHASHSEED": "0",
         "ESBMC_REPLAY_WITNESS": witness, "ESBMC_REPLAY_BOOLS": bits, "ESBMC_REPLAY_BOOLS_FILL": fill}
  try:
    subprocess.run(python + [os.path.join(replay_dir, "launch.py"), result, replay_dir, program],
                   env=env, cwd=os.path.dirname(program), stdin=subprocess.DEVNULL,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=replay_limits,
                   timeout=timeout)
    with open(result, encoding="utf-8") as f:
      return json.load(f)
  except (subprocess.SubprocessError, OSError, ValueError):
    return None


def replay_confirms_family(ending, prop):
  """Did the run end with an exception of the property's family? An exception
  group, a failure inside the stand-in and a run that took a value outside the
  call's domain or ran out of recorded values do not count."""
  if not ending or ending.get("status") != "exception" or ending.get("group"):
    return False
  if ending.get("inconclusive") or ending.get("raised_in_verifier"):
    return False
  return any(name in PYTHON_EXCEPTION_FAMILIES[prop] for name in ending.get("builtin_mro", ()))


def recorded_values(values_dir):
  """The witness file ESBMC wrote, or an empty witness when it wrote none, as
  when the program makes no nondet call."""
  try:
    with open(sorted(glob.glob(os.path.join(values_dir, "test_*.py")))[0], encoding="utf-8") as f:
      return f.read()
  except (IndexError, OSError, UnicodeDecodeError):
    return NO_VALUES


def write_replay_files(directory):
  """Write the embedded stand-in and launcher into `directory`."""
  os.makedirs(directory)
  for name, text in (("_sv_verifier.py", REPLAY_VERIFIER), ("launch.py", REPLAY_LAUNCHER)):
    with open(os.path.join(directory, name), "w", encoding="utf-8") as f:
      f.write(text)
  return directory


def confirmed_by_replay(benchmark, prop, values_dir, elapsed):
  """Does replaying the counterexample's values in CPython end with the
  property's exception? ESBMC's witness leaves out its nondet_bool values, so
  a bool call with no recorded bool takes the next bit of a bit string, tried
  breadth first, with a constant tail for loops that ask for a bool each pass."""
  budget = min(REPLAY_BUDGET, TOTAL_BUDGET - elapsed)
  if budget < MIN_REPLAY_BUDGET or not replay_possible(benchmark):
    return False
  with tempfile.TemporaryDirectory(prefix="esbmc-replay-") as work:
    program = copy_task(benchmark, work)
    if program is None:
      return False
    replay_dir = write_replay_files(os.path.join(work, "replay"))
    witness = os.path.join(work, "witness.py")
    with open(witness, "w", encoding="utf-8") as f:
      f.write(recorded_values(values_dir))
    started = time.monotonic()
    queue, runs = [("", "")], 0
    while queue and runs < REPLAY_MAX_RUNS:
      left = budget - (time.monotonic() - started)
      if left < MIN_REPLAY_BUDGET:
        return False
      bits, fill = queue.pop(0)
      ending = replay_once(work, replay_dir, program, witness, bits, fill, left)
      runs += 1
      if replay_confirms_family(ending, prop):
        return True
      if ending and not fill and ending.get("inconclusive") == "bool bits exhausted":
        queue += [(bits, "0"), (bits, "1")]
        if len(bits) < REPLAY_MAX_BITS:
          queue += [(bits + "0", ""), (bits + "1", "")]
  return False


# Embedded replay sources, kept verbatim and apart from the code above. They run
# only in the replay's own Python process, never in the wrapper's.
REPLAY_VERIFIER = r'''# Replay stand-in for sv-benchmarks' python/_sv_verifier.py, used only by the
# wrapper's CPython replay of an ESBMC counterexample (see esbmc-wrapper.py).
# nondet_* return the values of the counterexample, read from the
# `witness = [...]` file that --generate-pytest-testcase --pytest-values-only
# writes (path in ESBMC_REPLAY_WITNESS). Scalars are taken in the order ESBMC
# recorded them, lists from list0, list1, ... and dicts from dict0, ...
# A value outside the call's domain, or a call with nothing left to return,
# makes the replay inconclusive: _replay_inconclusive is set and a
# BaseException outside every property family ends the run, so a replay can
# only confirm. check_type and _matches_type are the reference code verbatim.
import ast
import os
import re
import string
import collections.abc
import types
from typing import Any, Callable, Literal, Union, get_args, get_origin

_replay_inconclusive = None


class _ReplayExhausted(BaseException):
    pass


def _literal(node):
    """Evaluate the literal subset ESBMC writes, plus float('nan'/'inf'/'-inf')."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _literal(node.operand)
        if type(v) not in (int, float):
            raise ValueError("unary on non-number")
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.List):
        return [_literal(e) for e in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_literal(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return {_literal(k): _literal(v) for k, v in zip(node.keys, node.values)}
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "float"
            and len(node.args) == 1 and not node.keywords and isinstance(node.args[0], ast.Constant)
            and node.args[0].value in ("nan", "inf", "-inf")):
        return float(node.args[0].value)
    raise ValueError("unsupported witness node " + type(node).__name__)


def _load():
    path = os.environ.get("ESBMC_REPLAY_WITNESS")
    if not path:
        return None
    tree = ast.parse(open(path).read())
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and [t.id for t in stmt.targets if isinstance(t, ast.Name)] == ["witness"]:
            cases = _literal(stmt.value)
            return cases[0] if cases else {}
    return None


_scalars, _lists, _dicts = [], [], []
try:
    _w = _load()
    if _w is None:
        _replay_inconclusive = "no witness"
    else:
        for k, v in _w.items():
            if re.fullmatch(r"list\d+", k):
                _lists.append((int(k[4:]), v))
            elif re.fullmatch(r"dict\d+", k):
                _dicts.append((int(k[4:]), v))
            else:
                _scalars.append(v)
        _lists = [v for _, v in sorted(_lists)]
        _dicts = [v for _, v in sorted(_dicts)]
except Exception as e:
    _replay_inconclusive = "witness unreadable: " + repr(e)[:200]


def _replay_unused():
    return {"scalars": len(_scalars), "lists": len(_lists), "dicts": len(_dicts)}


def _give_up(reason):
    global _replay_inconclusive
    if _replay_inconclusive is None:
        _replay_inconclusive = reason
    raise _ReplayExhausted(reason)


def _as_int(v):
    if type(v) is int:
        return v
    if type(v) is bool:
        return int(v)
    _give_up(f"int expected, got {v!r}")


def _as_bool(v):
    if type(v) is bool:
        return v
    if type(v) is int and v in (0, 1):
        return bool(v)
    _give_up(f"bool expected, got {v!r}")


def _as_str(v):
    if type(v) is str and all(c in string.printable for c in v):
        return v
    _give_up(f"printable str expected, got {v!r}")


def _as_float(v):
    if type(v) in (float, int):
        return float(v)
    _give_up(f"float expected, got {v!r}")


def _pop_scalar():
    if _replay_inconclusive is not None:
        _give_up(_replay_inconclusive)
    if not _scalars:
        _give_up("witness exhausted (scalar)")
    return _scalars.pop(0)


# ESBMC's witness leaves out nondet_bool values, so a bool call whose next
# recorded value is not a bool takes the next bit of ESBMC_REPLAY_BOOLS instead. The
# driver tries bit strings in turn; every choice is in nondet_bool's domain.
_bits = list(os.environ.get("ESBMC_REPLAY_BOOLS", ""))
_fill = os.environ.get("ESBMC_REPLAY_BOOLS_FILL", "")  # "0"/"1": every later bool
_replay_bits_used = 0


def nondet_bool():
    global _replay_bits_used
    if _replay_inconclusive is not None:
        _give_up(_replay_inconclusive)
    if _scalars and type(_scalars[0]) is bool:
        return _scalars.pop(0)
    if not _bits:
        if _fill:
            _replay_bits_used += 1
            return _fill == "1"
        _give_up("bool bits exhausted")
    _replay_bits_used += 1
    return _bits.pop(0) == "1"


def _pop_typed(ok):
    """The next recorded scalar if it fits the call, else the first later one
    that does. ESBMC does not always record scalars in call order; any value
    that fits is in the call's domain, so the choice only affects coverage."""
    if _replay_inconclusive is not None:
        _give_up(_replay_inconclusive)
    for i, v in enumerate(_scalars):
        if ok(v):
            return _scalars.pop(i)
    _give_up("witness exhausted (no fitting scalar)")


def nondet_int() -> int:
    return _pop_typed(lambda v: type(v) is int)


def nondet_str():
    return _pop_typed(lambda v: type(v) is str and all(c in string.printable for c in v))


def nondet_float():
    return float(_pop_typed(lambda v: type(v) is float))


def nondet_float_range(a, b):
    v = float(_pop_typed(lambda v: type(v) in (float, int) and a <= v <= b))
    if not (a <= v <= b):
        _give_up(f"{v!r} outside [{a!r}, {b!r}]")
    return v


_ELEM = {}


def _convert_elem(f, v):
    conv = _ELEM.get(f)
    if conv is None:
        _give_up("element maker is not a nondet_* function")
    return conv(v)


def nondet_list(nondet_elem):
    if _replay_inconclusive is not None:
        _give_up(_replay_inconclusive)
    if not _lists:
        _give_up("witness exhausted (list)")
    v = _lists.pop(0)
    if type(v) is not list:
        _give_up(f"list expected, got {v!r}")
    return [_convert_elem(nondet_elem, e) for e in v]


def nondet_dict(nondet_key, nondet_value):
    if _replay_inconclusive is not None:
        _give_up(_replay_inconclusive)
    if not _dicts:
        _give_up("witness exhausted (dict)")
    v = _dicts.pop(0)
    if type(v) is not dict:
        _give_up(f"dict expected, got {v!r}")
    return {_convert_elem(nondet_key, k): _convert_elem(nondet_value, x) for k, x in v.items()}


_ELEM.update({nondet_int: _as_int, nondet_bool: _as_bool, nondet_str: _as_str, nondet_float: _as_float})

def _matches_type(value, hint) -> bool:
    if hint is Any:
        return True
    if hint is None or hint is type(None):
        return value is None
    origin = get_origin(hint)
    if origin is None:
        # Plain class, e.g. int or list
        return isinstance(value, hint)
    args = get_args(hint)
    if origin is Union or origin is types.UnionType:
        return any(_matches_type(value, a) for a in args)
    if origin is Literal:
        return any(value == a and type(value) is type(a) for a in args)
    if not isinstance(value, origin):
        return False
    if not args:
        return True
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return all(_matches_type(v, args[0]) for v in value)
        if args == ((),):
            return len(value) == 0
        return len(value) == len(args) and all(_matches_type(v, a) for v, a in zip(value, args))
    if origin is collections.abc.Callable:
        # Signatures are not checked, only callability
        return True
    if issubclass(origin, collections.abc.Mapping):
        key_hint, value_hint = args
        return all(_matches_type(k, key_hint) and _matches_type(v, value_hint) for k, v in value.items())
    if issubclass(origin, collections.abc.Iterable):
        (elem_hint,) = args
        return all(_matches_type(v, elem_hint) for v in value)
    raise NotImplementedError(f"check_type: unsupported type hint {hint!r}")

def check_type(value, hint):
    """Raise a TypeError if value is not of the type described by hint (e.g. list[list[int]])."""
    if not _matches_type(value, hint):
        raise TypeError(f"expected value of type {hint!r}, got {value!r}")
'''

REPLAY_LAUNCHER = r'''# Runs one benchmark program as __main__ under the replay _sv_verifier and
# records how it ended in a JSON file.
# Usage: python3 launch.py RESULT_JSON VERIFIER_DIR PROGRAM
import builtins
import json
import os
import runpy
import sys
import traceback

res_path, verifier_dir, prog = sys.argv[1], sys.argv[2], os.path.abspath(sys.argv[3])
out = {"status": "no-result"}

# The task's own directory comes first, so a task-local _sv_verifier.py is
# never shadowed by the replay one.
sys.path[0:1] = [os.path.dirname(prog), verifier_dir]
sys.argv = [prog]
try:
    runpy.run_path(prog, run_name="__main__")
    out = {"status": "normal"}
    sys.stdout.flush()
except SystemExit as e:
    out = {"status": "exit", "code": repr(e.code)}
except BaseException as e:
    tb = e.__traceback__.tb_next  # drop this launcher's frame
    frames = traceback.extract_tb(tb)
    inner = frames[-1] if frames else None
    verifier_file = os.path.join(verifier_dir, "_sv_verifier.py")
    out = {
        "status": "exception",
        "cls": type(e).__name__,
        "builtin_mro": [c.__name__ for c in type(e).__mro__ if c.__module__ == "builtins"],
        "group": isinstance(e, getattr(builtins, "BaseExceptionGroup", ())),
        # A failure inside the stand-in's own nondet_* code says nothing about
        # the program. check_type is part of the specification.
        "raised_in_verifier": bool(inner and os.path.abspath(inner.filename) == os.path.abspath(verifier_file)
                                   and inner.name not in ("check_type", "_matches_type")),
    }
module = sys.modules.get("_sv_verifier")
if module is None:
    out["inconclusive"] = None  # no nondet call was possible
elif getattr(module, "__file__", None) == os.path.join(verifier_dir, "_sv_verifier.py"):
    out["inconclusive"] = getattr(module, "_replay_inconclusive", None)
else:
    out["inconclusive"] = "the replay _sv_verifier was not the module imported"
fd = os.open(res_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
os.write(fd, json.dumps(out).encode())
os.close(fd)
os._exit(0)
'''


def has_invariants(witness_path):
  """Whether the YAML names an `invariant` in any style (block, flow, JSON).
  A comment naming one also counts, which only keeps loops rolled."""
  with open(witness_path) as f:
    return re.search(r"\binvariant\b", f.read()) is not None


def verify(strat, prop, arch, benchmark, dargs, esbmc_ci, witness_path, validate_mode):
  if witness_path and validate_mode == "violation":
    strat = "incr"

  # Unrolling a loop moves the waypoints placed on it, and an invariant on a
  # loop --goto-unwind removes is never checked. Without it a bounded loop is
  # proved one k-step at a time, so a correctness witness with no invariants,
  # such as the ones ESBMC writes, is validated with it.
  keep_loops = bool(witness_path) and (
    validate_mode == "violation" or has_invariants(witness_path))
  esbmc_command_line = get_command_line(strat, prop, arch, benchmark, dargs, esbmc_ci, keep_loops)

  if witness_path:
    esbmc_command_line += "--validate-" + validate_mode + "-witness "
    esbmc_command_line += "--witness " + witness_path + " "

  started = time.monotonic()
  if prop == Property.overflow and "--k-induction " in esbmc_command_line:
    # The inductive step proves most no-overflow loops in seconds, but on
    # nonlinear arithmetic its solve can stall, and on goto-heavy code it can
    # path-explode; those runs fall back to incremental BMC. The memory cap
    # stops this run before it reaches the competition's limit, which would
    # end the task instead.
    memlimit = "--memlimit " + KIND_OVERFLOW_MEMLIMIT + " "
    output = run(esbmc_command_line + memlimit + KIND_OVERFLOW_INTERVALS, KIND_OVERFLOW_TIMEOUT)
    res = parse_result(output.decode(), prop)
    if res in (Result.success, Result.fail_overflow):
      return res
    esbmc_command_line = esbmc_command_line.replace(strategy_flags(strat), "--incremental-bmc ")
  if prop in PYTHON_EXCEPTION_FAMILIES and not witness_path:
    return verify_python(esbmc_command_line, prop, benchmark, started)
  output = run(esbmc_command_line)
  elapsed = time.monotonic() - started
  res = parse_result(output.decode(), prop)
  if res == Result.fail_python and not confirmed_by_integer_encoding(esbmc_command_line, output.decode(), prop, elapsed):
    return Result.unknown
  return res


def counterexample_values(esbmc_command_line, prop, values_dir, elapsed):
  """Run ESBMC again on a violation it found, asking for the values of the
  nondet calls on the failing path. The flags change the front end (a walrus in
  a boolean operand is refused under them), so the first run never carries
  them. Returns whether the second run found the violation too."""
  budget = min(VALUES_RUN_TIMEOUT, 1.5 * elapsed + 10, TOTAL_BUDGET - elapsed - IR_RERUN_TIMEOUT)
  if budget < MIN_REPLAY_BUDGET:
    return False
  output = run(esbmc_command_line + values_flags(values_dir), budget)
  return parse_result(output.decode(), prop) == Result.fail_python


def replay_possible(benchmark):
  """Whether a CPython replay of this task could be tried at all."""
  return os.path.isfile(benchmark) and not task_provides_verifier(benchmark)


def confirmed_by_cpython(esbmc_command_line, prop, benchmark, elapsed, started):
  """Run ESBMC again for the counterexample's values and replay them."""
  values_dir = tempfile.mkdtemp(prefix="esbmc-values-")
  try:
    return counterexample_values(esbmc_command_line, prop, values_dir, elapsed) and \
        confirmed_by_replay(benchmark, prop, values_dir, time.monotonic() - started)
  finally:
    shutil.rmtree(values_dir, ignore_errors=True)


def verify_python(esbmc_command_line, prop, benchmark, started):
  """One run of a Python task. A FALSE stays FALSE when replaying ESBMC's
  counterexample in CPython ends with the property's exception, or else when a
  --ir re-run repeats it."""
  output = run(esbmc_command_line)
  elapsed = time.monotonic() - started
  res = parse_result(output.decode(), prop)
  if res != Result.fail_python:
    return res
  if replay_possible(benchmark) and confirmed_by_cpython(esbmc_command_line, prop, benchmark, elapsed, started):
    print("CPython replay of the counterexample confirms the violation")
    return res
  if replay_possible(benchmark):
    elapsed = time.monotonic() - started
  if confirmed_by_integer_encoding(esbmc_command_line, output.decode(), prop, elapsed):
    return res
  return Result.unknown


# Test-Comp. benchexec's esbmc tool-info runs esbmc-wrapper.py in both
# competitions, so this one wrapper serves both: a run with -o/--coverage (the
# fm-tools testcomp options) or a COVER property writes a test suite instead of
# a witness. Formerly scripts/competitions/testcomp/esbmc-wrapper-cov.py.

class TestGoal:
  coverage = 1    # coverage-branches, -conditions, -statements
  error_call = 2  # coverage-error-call

# ESBMC runs inside this directory and writes each test case there as soon as
# it finds it, so a run killed at the time limit still leaves its test suite.
TEST_DIR = "test-suite"

# Most characters of ESBMC output echoed to stdout: Test-Comp caps a tool's
# output at 2 MB and long counterexample traces exceed it.
MAX_ECHO = 1000000

# Unlike SV-COMP, Test-Comp runs assume allocation succeeds.
testcomp_dargs = "--sv-comp --no-div-by-zero-check --force-malloc-success --state-hashing --add-symex-value-sets "
testcomp_dargs += "--no-align-check --k-step 2 --floatbv --unlimited-k-steps --no-vla-size-check "


def is_testcomp_run(coverage, property_file_content):
  """SV-COMP properties are all CHECK( ... ); Test-Comp's are COVER( ... )."""
  return bool(coverage) or "COVER(" in property_file_content


def test_goal(property_file_content):
  """Test-Comp only measures branch coverage, so the condition and statement
  coverage properties are generated for as branch coverage."""
  if any(edge in property_file_content for edge in
         ("COVER EDGES(@DECISIONEDGE)", "COVER EDGES(@CONDITIONEDGE)", "COVER EDGES(@BASICBLOCKENTRY)")):
    return TestGoal.coverage
  if any(goal in property_file_content for goal in
         ("COVER EDGES(@CALL(reach_error))", "COVER EDGES(@CALL(__VERIFIER_error))",
          "LTL(G ! call(reach_error()))", "LTL(G ! call(__VERIFIER_error()))")):
    return TestGoal.error_call
  return None


def create_test_dir():
  """Create an empty TEST_DIR, clearing what an earlier run left there."""
  if os.path.exists(TEST_DIR):
    for file_name in os.listdir(TEST_DIR):
      file_path = os.path.join(TEST_DIR, file_name)
      if os.path.isfile(file_path):
        os.remove(file_path)
  else:
    os.makedirs(TEST_DIR)


def testcomp_command_line(strat, goal, arch, benchmark, coverage):
  # ESBMC runs inside TEST_DIR, so the binary and the benchmark are absolute
  esbmc = shlex.quote(os.path.join(os.path.dirname(os.path.realpath(__file__)), "esbmc"))
  command_line = esbmc + " " + testcomp_dargs
  command_line += shlex.quote(os.path.abspath(benchmark)) + " "
  command_line += "-Wno-incompatible-pointer-types -Wno-int-conversion "
  command_line += "--32 " if arch == 32 else "--64 "

  concurrency = check_if_benchmark_contains_pthread(benchmark)
  if concurrency:
    command_line += " --no-por --context-bound 2 "

  if goal == TestGoal.coverage:
    command_line += "--base-k-step 2 --no-bounds-check --no-pointer-check --quiet --no-standard-checks "
    command_line += "--generate-testcase "
    if coverage == "branch":
      command_line += "--branch-coverage "
    elif coverage == "condition":
      command_line += "--condition-coverage-rm --no-cov-asserts "
    elif coverage == "branch-function":
      command_line += "--branch-function-coverage "
  else:
    command_line += "--base-k-step 5 --enable-unreachability-intrinsic "
    command_line += "--generate-testcase "
    if concurrency:
      command_line += "--no-pointer-check --no-bounds-check "
    else:
      command_line += "--no-pointer-check --interval-analysis --no-bounds-check --error-label ERROR --goto-unwind --unlimited-goto-unwind "

  # Concurrency only works with incremental BMC. Test-Comp keeps
  # --max-inductive-step 8; strategy_flags() uses SV-COMP's 3.
  if concurrency or strat == "incr":
    command_line += "--incremental-bmc "
  elif strat in ("fixed", "kinduction"):
    command_line += "--k-induction --max-inductive-step 8 "
  elif strat == "falsi":
    command_line += "--falsification "
  return command_line


def testcomp_run(cmd_line):
  print("Verifying with ESBMC")
  print("Command: " + cmd_line)
  out = do_exec(cmd_line, cwd=TEST_DIR).decode(errors="replace")
  # The result is parsed from the full output; only the tail is echoed
  if len(out) > MAX_ECHO:
    print(f"[... {len(out) - MAX_ECHO} characters of ESBMC output omitted ...]")
    print(out[-MAX_ECHO:])
  else:
    print(out)
  return out


def testcomp_parse_result(the_output):
  if "Timed out" in the_output:
    return Result.err_timeout
  if "Out of memory" in the_output:
    return Result.err_memout
  if "VERIFICATION FAILED" in the_output:
    if "unwinding assertion loop" in violated_property_text(the_output):
      return Result.err_unwinding_assertion
    return Result.fail_reach
  if "VERIFICATION SUCCESSFUL" in the_output:
    return Result.success
  return Result.unknown


def testcomp(strat, property_file_content, arch, benchmark, coverage):
  goal = test_goal(property_file_content)
  if goal is None:
    print("Unsupported Property")
    exit(1)
  command_line = testcomp_command_line(strat, goal, arch, benchmark, coverage)
  if not args.dry_run:
    create_test_dir()
  return testcomp_parse_result(testcomp_run(command_line))


if __name__ == "__main__":
  # Options
  parser = argparse.ArgumentParser()
  parser.add_argument("-a", "--arch", help="Either 32 or 64 bits", type=int, choices=[32, 64], default=32)
  parser.add_argument("-v", "--version", help="Prints ESBMC's version", action='store_true')
  parser.add_argument("-p", "--propertyfile", help="Path to the property file")
  parser.add_argument("benchmark", nargs='?', help="Path to the benchmark")
  parser.add_argument("-s", "--strategy", help="ESBMC's strategy", choices=["kinduction", "falsi", "incr", "fixed"], default="fixed")
  parser.add_argument("-n", "--dry-run", help="do not actually run ESBMC, just print the command", action='store_true')
  parser.add_argument("--ci", help="run this wrapper with special options for the CI (internal use)", action='store_true')
  parser.add_argument("--witness", help="Path to witness file; enables witness validation mode")
  parser.add_argument("--validate-violation-witness", dest="validate_violation", action='store_true',
                      help="Validate a violation witness (use with --witness)")
  parser.add_argument("--validate-correctness-witness", dest="validate_correctness", action='store_true',
                      help="Validate a correctness witness (use with --witness)")
  parser.add_argument("-o", "--coverage", help="Test-Comp: coverage criterion for the generated tests",
                      choices=["branch", "branch-function", "condition"])

  args = parser.parse_args()

  arch = args.arch
  version = args.version
  property_file = args.propertyfile
  benchmark = args.benchmark
  strategy = args.strategy
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

  if is_testcomp_run(args.coverage, property_file_content):
    print(get_result_string(testcomp(strategy, property_file_content, arch, benchmark, args.coverage)))
    exit(0)

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

  result = verify(strategy, category_property, arch, benchmark, esbmc_dargs, esbmc_ci, witness_path, validate_mode)

  print(get_result_string(result))
