#!/usr/bin/env python3
"""Self-test. Run: python3 scripts/competitions/svcomp/test_esbmc_wrapper.py"""

import importlib.util
import os
import unittest

_spec = importlib.util.spec_from_file_location(
    "esbmc_wrapper", os.path.join(os.path.dirname(__file__), "esbmc-wrapper.py"))
wrapper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wrapper)

# Since esbmc/esbmc#7064 every run ends with a table naming every property,
# including the ones it never checked. Shaped like that output.
REACH_WITH_UNCHECKED_UNWINDING = """
Violated property:
  file main.c line 22 column 51 function __VERIFIER_assert
  error label
  0

** Results:
main.c, function __VERIFIER_assert
  FAILED       [__VERIFIER_assert.assertion.1]  line 22  error label
main.c, function main
  NOT CHECKED  [main.assertion.1]               line 42  unwinding assertion loop 3

** 1 of 2 properties failed, 1 not checked

VERIFICATION FAILED
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

FREE_WITH_UNCHECKED_LEAK = """
Violated property:
  file main.c line 884 column 3 function bad
  dereference failure: invalid pointer freed
  CWE: CWE-415, CWE-416

** Results:
main.c, function bad
  FAILED       [bad.invalid-pointer-freed.1]  line 884  dereference failure: invalid ptr freed
main.c, function main
  NOT CHECKED  [main.memory-leak.1]           line 914  dereference failure: forgotten memory: dyn_1

** 1 of 2 properties failed, 1 not checked

VERIFICATION FAILED
"""

FREE_NON_ZERO_OFFSET = """
Violated property:
  file main.c line 12 column 3 function main
  Operand of free must have zero pointer offset

VERIFICATION FAILED
"""

UNLISTED_DEREF_COMMENT = """
Violated property:
  file main.c line 6368 column 5 function attach
  dereference failure: Data object accessed with code type

VERIFICATION FAILED
"""


class ParseResultTest(unittest.TestCase):
    def verdict(self, output, prop):
        return wrapper.get_result_string(wrapper.parse_result(output, prop))

    def test_unchecked_unwinding_row_does_not_mask_a_violation(self):
        # Regression: matching the whole output read the NOT CHECKED row and
        # reported Unknown for ~2300 tasks of the 30s run.
        self.assertEqual(
            self.verdict(REACH_WITH_UNCHECKED_UNWINDING, wrapper.Property.reach),
            "FALSE_REACH")

    def test_violated_unwinding_assertion_is_still_inconclusive(self):
        self.assertEqual(
            self.verdict(VIOLATED_UNWINDING, wrapper.Property.reach), "Unknown")

    def test_unchecked_leak_row_does_not_win_over_the_violated_free(self):
        self.assertEqual(
            self.verdict(FREE_WITH_UNCHECKED_LEAK, wrapper.Property.memory),
            "FALSE_FREE")

    def test_free_offset_is_reachable(self):
        self.assertEqual(
            self.verdict(FREE_NON_ZERO_OFFSET, wrapper.Property.memory),
            "FALSE_FREE")

    def test_deref_comment_outside_the_list_still_falsifies(self):
        self.assertEqual(
            self.verdict(UNLISTED_DEREF_COMMENT, wrapper.Property.memory),
            "FALSE_DEREF")

    def test_successful_run_is_unaffected(self):
        self.assertEqual(
            self.verdict("** 0 of 3 properties failed, 3 passed\n"
                         "VERIFICATION SUCCESSFUL\n", wrapper.Property.reach),
            "TRUE")


if __name__ == "__main__":
    unittest.main()
