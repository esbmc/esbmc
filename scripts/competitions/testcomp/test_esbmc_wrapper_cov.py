#!/usr/bin/env python3
"""Self-test. Run: python3 scripts/competitions/testcomp/test_esbmc_wrapper_cov.py"""

import importlib.util
import os
import unittest

_spec = importlib.util.spec_from_file_location(
    "esbmc_wrapper_cov", os.path.join(os.path.dirname(__file__), "esbmc-wrapper-cov.py"))
wrapper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wrapper)

# Since esbmc/esbmc#7064 every run ends with a table naming every property,
# including the ones it never checked. Shaped like ESBMC 8.6 output on
# reducercommutativity/rangesum10.i in Test-Comp 2027.
REACH_WITH_UNCHECKED_UNWINDING = """
Violated property:
  file rangesum10.i line 61 column 6 function main
  reach_error
  0


** Results:
rangesum10.i, function main
  NOT CHECKED  [main.assertion.1]  line 54  unwinding assertion loop 3
  FAILED       [main.assertion.2]  line 61  reach_error

** 1 of 2 properties failed, 1 not checked

VERIFICATION FAILED

Bug found (k = 11)
"""

VIOLATED_UNWINDING = """
Violated property:
  file main.c line 42 column 3 function main
  unwinding assertion loop 3

** Results:
main.c, function main
  FAILED       [main.assertion.1]  line 42  unwinding assertion loop 3

** 1 of 1 properties failed

VERIFICATION FAILED
"""

# --quiet, which coverage mode passes, drops the counterexample and with it the
# "Violated property:" block, leaving only the table row.
QUIET_VIOLATED_UNWINDING = """
** Results:
main.c, function main
  FAILED       [main.assertion.1]  line 42  unwinding assertion loop 3

VERIFICATION FAILED
"""

SUCCESSFUL = """
** Results:
main.c, function main
  PASSED       [main.assertion.1]  line 42  unwinding assertion loop 3

VERIFICATION SUCCESSFUL
"""


def verdict(output):
    return wrapper.get_result_string(wrapper.parse_result(output, wrapper.Property.reach))


class ParseResult(unittest.TestCase):

    def test_unchecked_unwinding_row_does_not_mask_the_reach_violation(self):
        self.assertEqual(verdict(REACH_WITH_UNCHECKED_UNWINDING), "FALSE_REACH")

    def test_violated_unwinding_assertion_is_unknown(self):
        self.assertEqual(verdict(VIOLATED_UNWINDING), "Unknown")

    def test_quiet_violated_unwinding_assertion_is_unknown(self):
        self.assertEqual(verdict(QUIET_VIOLATED_UNWINDING), "Unknown")

    def test_successful_run_is_true(self):
        self.assertEqual(verdict(SUCCESSFUL), "TRUE")


if __name__ == "__main__":
    unittest.main()
