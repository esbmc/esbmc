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


class TransitionSystemStrategyTest(unittest.TestCase):
    def test_ric3_verdicts(self):
        self.assertEqual(wrapper.parse_ric3("UNSAT\n"), wrapper.Result.success)
        self.assertEqual(wrapper.parse_ric3("SAT\n"), wrapper.Result.fail_reach)
        self.assertEqual(wrapper.parse_ric3("UNKNOWN\n"), wrapper.Result.unknown)
        self.assertEqual(wrapper.parse_ric3(""), wrapper.Result.unknown)

    def test_ts_runs_never_unwind_loops(self):
        for strat in wrapper.TS_FLAGS:
            cmd = wrapper.ts_command_line(strat, 64, "task.c", "m.btor2")
            self.assertNotIn("--goto-unwind", cmd)
            self.assertIn("--64", cmd)
            self.assertIn(wrapper.TS_FLAGS[strat], cmd)

    def test_ts_engines_do_not_fall_back(self):
        for strat in ("ts-kind", "ts-pdr"):
            self.assertIn("--ts-no-fallback", wrapper.ts_command_line(strat, 32, "t.c", "m.btor2"))

    def test_hardware_strategies_export_to_the_given_model(self):
        for strat in ("ts-check", "ts-ric3"):
            self.assertIn("--ts-btor2 /tmp/x/m.btor2",
                          wrapper.ts_command_line(strat, 64, "t.c", "/tmp/x/m.btor2"))
        self.assertIn("/tmp/x/m.btor2", wrapper.ric3_command_line("/tmp/x/m.btor2"))

    def test_ric3_seed_is_pinned(self):
        # Unpinned, rIC3's search order decides the verdict: the same model has
        # solved in 2.9s at seed 0 and timed out past 120s at seeds 1-5, so a
        # run has to name its seed to be reproducible.
        self.assertIn("--rseed 0", wrapper.ric3_command_line("m.btor2"))
        self.assertIn("--rseed 7", wrapper.ric3_command_line("m.btor2", "7"))

    def test_ric3_seed_precedes_the_model(self):
        # rIC3 takes the model positionally, so an option after it is a parse
        # error rather than a setting.
        cmd = wrapper.ric3_command_line("m.btor2")
        self.assertLess(cmd.index("--rseed"), cmd.index("m.btor2"))

    def test_rejected_program_is_unknown(self):
        self.assertEqual(
            self.verdict("TS-CHECK rejected: main has no loop\nVERIFICATION UNKNOWN\n"),
            "Unknown")

    def verdict(self, output):
        return wrapper.get_result_string(wrapper.parse_result(output, wrapper.Property.reach))


if __name__ == "__main__":
    unittest.main()
