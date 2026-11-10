#!/usr/bin/env python3
"""Self-test. Run: python3 scripts/competitions/svcomp/test_esbmc_wrapper.py"""

import argparse
import shlex
import importlib.util
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

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

DEREF_THEN_LEAK = """
Violated property:
  file main.c line 9 column 3 function main
  dereference failure: invalidated dynamic object
  CWE: CWE-416

** Results:
main.c, function main
  FAILED       [main.invalidated-dynamic-object.1]  line  9  dereference failure: invalidated dynamic object
  FAILED       [main.memory-leak.1]                 line 11  dereference failure: forgotten memory: dynamic_2_value

** 2 of 2 properties failed

VERIFICATION FAILED
"""

DEREF_THEN_BAD_FREE = """
Violated property:
  file main.c line 9 column 3 function detach
  dereference failure: invalid pointer
  CWE: CWE-416, CWE-822, CWE-824, CWE-908

** Results:
main.c, function detach
  FAILED       [detach.invalid-pointer-dereference.1]  line  9  dereference failure: invalid pointer
  FAILED       [detach.invalid-pointer-freed.1]        line 12  dereference failure: invalid pointer freed

** 2 of 2 properties failed

VERIFICATION FAILED
"""

FREE_THEN_DEREF = """
Violated property:
  file main.c line 11 column 3 function main
  dereference failure: free() of non-dynamic memory
  CWE: CWE-590, CWE-761

** Results:
main.c, function main
  FAILED       [main.free-non-dynamic-memory.1]          line 11  dereference failure: free() of non-dynamic memory
  FAILED       [main.expired-pointer-dereference.1]      line 11  dereference failure: accessed expired variable pointer `buf'

** 2 of 2 properties failed

VERIFICATION FAILED
"""

UNLISTED_DEREF_COMMENT = """
Violated property:
  file main.c line 6368 column 5 function attach
  dereference failure: Data object accessed with code type

VERIFICATION FAILED
"""

# Python track outputs, captured from ESBMC 8.5.0 runs over programs built on
# sv-benchmarks' python/_sv_verifier.py nondet module.

PY_ZERODIVISION = """
Violated property:
  file p_zerodiv.py line 5 column 12 function main
  uncaught exception: ZeroDivisionError
  !(c:@__ESBMC_exc_thrown && c:@__ESBMC_exc_typeid == 16)

** Results:
p_zerodiv.py
  NOT CHECKED  [global.assertion.1]       line 0  uncaught exception
  NOT CHECKED  [global.assertion.2]       line 0  uncaught exception: IndexError
  NOT CHECKED  [global.assertion.3]       line 0  uncaught exception: ValueError
p_zerodiv.py, function main
  NOT CHECKED  [main.division-by-zero.1]  line 5  division by zero
  FAILED       [main.assertion.1]         line 5  uncaught exception: ZeroDivisionError

** 1 of 5 properties failed, 4 not checked

VERIFICATION FAILED
"""

PY_INDEXERROR = """
Violated property:
  file p_index.py line 6 column 12 function main
  uncaught exception: IndexError
  !(c:@__ESBMC_exc_thrown && c:@__ESBMC_exc_typeid == 14)

** Results:
p_index.py, function main
  FAILED       [main.assertion.1]  line 6  uncaught exception: IndexError

** 1 of 4 properties failed, 3 not checked

VERIFICATION FAILED
"""

PY_TYPEERROR = """
Violated property:
  file p_type.py line 5 column 8 function main
  uncaught exception: TypeError

** Results:
p_type.py, function main
  FAILED       [main.assertion.1]  line 5  uncaught exception: TypeError

** 1 of 5 properties failed, 4 not checked

VERIFICATION FAILED
"""

PY_ASSERTION = """
Violated property:
  file p_assert2.py line 5 column 4 function main
  assertion x != 3
  x != 3

** Results:
p_assert2.py, function main
  FAILED       [main.assertion.1]  line 5  assertion x != 3

** 1 of 1 properties failed

VERIFICATION FAILED
"""

# ValueError is in no property of the Python track, so no run that escapes one
# says anything about any of the four properties.
PY_VALUEERROR = """
Violated property:
  file p_raise_value.py line 6 column 8 function main
  uncaught exception: ValueError

** Results:
p_raise_value.py, function main
  FAILED       [main.assertion.1]  line 6  uncaught exception: ValueError

** 1 of 4 properties failed, 3 not checked

VERIFICATION FAILED
"""

# ESBMC's own list model failing internally on a correct program: the comment
# carries no exception type, exactly like `raise AssertionError(msg)` does.
# Reporting this as an assertion violation is the -16 answer of esbmc #7628's
# sibling on bm_fannkuch_det.py.
PY_MODEL_FAILURE = """
Violated property:
  file list.c line 108 column 3 function __ESBMC_list_size
  TypeError: object of this type has no len()

** Results:
list.c, function __ESBMC_list_size
  FAILED       [__ESBMC_list_size.assertion.1]  line 108  TypeError: object of this type has no len()

** 1 of 3 properties failed, 2 not checked

VERIFICATION FAILED
"""

# The trace prints the program's own output unindented, between states.
PROGRAM_PRINTS_BEFORE_LEAK = """
[Counterexample]


State 2 file main.c line 5 column 3 function main thread 0
----------------------------------------------------
  p = 0
{}


State 5 file main.c line 14 column 1 function main thread 0
----------------------------------------------------
Violated property:
  file main.c line 14 column 1 function main
  dereference failure: forgotten memory: dynamic_1_array
  CWE: CWE-401


** Results:
main.c, function main
  FAILED       [main.memory-leak.1]  line  14  dereference failure: forgotten memory: dynamic_1_array

** 1 of 1 properties failed

VERIFICATION FAILED
"""

ESBMC_OUT_OF_MEMORY = """
Starting Bounded Model Checking
ERROR: Out of memory

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

    # The model goes on past its first violation and fails later claims too
    # (#9895), shaped like the ldv-linux-3.4-simple and Juliet CWE590 logs.
    def test_a_leak_after_the_violated_deref_does_not_decide(self):
        self.assertEqual(
            self.verdict(DEREF_THEN_LEAK, wrapper.Property.memory),
            "FALSE_DEREF")

    def test_a_leak_violated_first_still_decides(self):
        output = DEREF_THEN_LEAK.replace(
            "line 9 column 3 function main\n  dereference failure: invalidated dynamic object",
            "line 11 column 1 function main\n  dereference failure: forgotten memory: dynamic_2_value")
        self.assertEqual(
            self.verdict(output, wrapper.Property.memory), "FALSE_MEMTRACK")

    def test_a_bad_free_after_the_violated_deref_does_not_decide(self):
        self.assertEqual(
            self.verdict(DEREF_THEN_BAD_FREE, wrapper.Property.memory),
            "FALSE_DEREF")

    def test_a_deref_failing_with_the_violated_free_does_not_decide(self):
        self.assertEqual(
            self.verdict(FREE_THEN_DEREF, wrapper.Property.memory),
            "FALSE_FREE")

    def test_program_output_in_the_counterexample_is_not_an_esbmc_error(self):
        # esbmc/esbmc#9890: sv-benchmarks memsafety-cve/cups.i prints this.
        for printed in ("Out of memory", "ERROR: Out of memory", "ERROR: Timed out"):
            with self.subTest(printed=printed):
                self.assertEqual(
                    self.verdict(PROGRAM_PRINTS_BEFORE_LEAK.format(printed),
                                 wrapper.Property.memory),
                    "FALSE_MEMTRACK")

    def test_program_printing_successful_does_not_prove_the_task(self):
        # A free violation decides nothing for memcleanup, so the only
        # SUCCESSFUL left in the output is the program's.
        output = "[Counterexample]\nVERIFICATION SUCCESSFUL\n" + FREE_WITH_UNCHECKED_LEAK
        self.assertEqual(self.verdict(output, wrapper.Property.memcleanup), "Unknown")

    def test_violated_comment_naming_an_esbmc_error_still_falsifies(self):
        output = REACH_WITH_UNCHECKED_UNWINDING.replace(
            "error label", 'assertion !"Out of memory" && !"Timed out"')
        self.assertEqual(self.verdict(output, wrapper.Property.reach), "FALSE_REACH")

    def test_esbmc_out_of_memory_is_unknown(self):
        self.assertEqual(
            self.verdict(ESBMC_OUT_OF_MEMORY, wrapper.Property.memory), "Unknown")

    # Under k-induction a step that runs out of memory is unknown and the next
    # k is tried, so a later step can still decide the task.
    OOM_AT_EARLIER_K = ("Checking base case, k = 1\n" + ESBMC_OUT_OF_MEMORY +
                        "Checking base case, k = 3\n")

    def test_a_proof_after_an_earlier_out_of_memory_is_true(self):
        self.assertEqual(
            self.verdict(self.OOM_AT_EARLIER_K + "VERIFICATION SUCCESSFUL\n",
                         wrapper.Property.py_assertion), "TRUE")

    def test_a_violation_after_an_earlier_out_of_memory_is_false(self):
        self.assertEqual(
            self.verdict(self.OOM_AT_EARLIER_K + PY_ASSERTION, wrapper.Property.py_assertion),
            "FALSE")

    def test_a_run_out_of_memory_without_a_verdict_is_a_memory_out(self):
        self.assertEqual(
            wrapper.parse_result(self.OOM_AT_EARLIER_K, wrapper.Property.py_assertion),
            wrapper.Result.err_memout)

    def test_a_timeout_after_an_out_of_memory_is_a_timeout(self):
        self.assertEqual(
            self.verdict(self.OOM_AT_EARLIER_K + "ERROR: Timed out\n", wrapper.Property.py_assertion),
            "Timed out")

    def test_esbmc_timeout_after_a_counterexample_is_a_timeout(self):
        self.assertEqual(
            self.verdict(PROGRAM_PRINTS_BEFORE_LEAK.format("") + "ERROR: Timed out\n",
                         wrapper.Property.memory),
            "Timed out")

    def test_successful_run_is_unaffected(self):
        self.assertEqual(
            self.verdict("** 0 of 3 properties failed, 3 passed\n"
                         "VERIFICATION SUCCESSFUL\n", wrapper.Property.reach),
            "TRUE")


# The scalar model's own TypeError on `r + 1`, where `r` is an int on one path
# and a str on the other.
PY_MODEL_TYPEERROR = """
Violated property:
  file scalar.c line 218 column 3 function __python_scalar_add_num
  TypeError: unsupported operand type(s) for +
  (_Bool)tagged && (_Bool)type_matches


** Results:
scalar.c, function __python_scalar_add_num
  FAILED       [__python_scalar_add_num.assertion.1]  line 218  TypeError: unsupported operand type(s) for +

** 1 of 12 properties failed, 6 passed, 5 not checked

VERIFICATION FAILED
"""

class PythonPropertyTest(unittest.TestCase):
    """The Python track scores one exception family per property file.

    One ESBMC run checks every family at once, so the family a run falsified
    has to be read off the violated-property comment. Crediting a violation to
    the wrong family is a wrong answer, worth -16.
    """

    def verdict(self, output, prop):
        return wrapper.get_result_string(wrapper.parse_result(output, prop))

    def test_zerodivision_falsifies_arithmetic(self):
        self.assertEqual(
            self.verdict(PY_ZERODIVISION, wrapper.Property.py_arithmetic), "FALSE")

    def test_zerodivision_says_nothing_about_data_lookup(self):
        self.assertEqual(
            self.verdict(PY_ZERODIVISION, wrapper.Property.py_datalookup), "Unknown")

    def test_zerodivision_says_nothing_about_assertion_safety(self):
        self.assertEqual(
            self.verdict(PY_ZERODIVISION, wrapper.Property.py_assertion), "Unknown")

    def test_indexerror_falsifies_data_lookup(self):
        self.assertEqual(
            self.verdict(PY_INDEXERROR, wrapper.Property.py_datalookup), "FALSE")

    def test_typeerror_falsifies_dynamic_typing(self):
        self.assertEqual(
            self.verdict(PY_TYPEERROR, wrapper.Property.py_dyntyping), "FALSE")

    def test_typeerror_says_nothing_about_arithmetic(self):
        self.assertEqual(
            self.verdict(PY_TYPEERROR, wrapper.Property.py_arithmetic), "Unknown")

    def test_model_typeerror_falsifies_dynamic_typing(self):
        self.assertEqual(
            self.verdict(PY_MODEL_TYPEERROR, wrapper.Property.py_dyntyping), "FALSE")

    def test_model_typeerror_says_nothing_about_arithmetic(self):
        self.assertEqual(
            self.verdict(PY_MODEL_TYPEERROR, wrapper.Property.py_arithmetic), "Unknown")

    def test_internal_len_failure_is_not_a_typeerror(self):
        self.assertEqual(
            self.verdict(PY_MODEL_FAILURE, wrapper.Property.py_dyntyping), "Unknown")

    def test_assert_falsifies_assertion_safety(self):
        self.assertEqual(
            self.verdict(PY_ASSERTION, wrapper.Property.py_assertion), "FALSE")

    def test_assert_says_nothing_about_arithmetic(self):
        self.assertEqual(
            self.verdict(PY_ASSERTION, wrapper.Property.py_arithmetic), "Unknown")

    def test_valueerror_falsifies_no_python_property(self):
        for prop in (wrapper.Property.py_assertion, wrapper.Property.py_arithmetic,
                     wrapper.Property.py_datalookup, wrapper.Property.py_dyntyping):
            self.assertEqual(self.verdict(PY_VALUEERROR, prop), "Unknown")

    def test_internal_model_failure_is_not_an_assertion_violation(self):
        self.assertEqual(
            self.verdict(PY_MODEL_FAILURE, wrapper.Property.py_assertion), "Unknown")

    def test_successful_python_run_is_true(self):
        self.assertEqual(
            self.verdict("** 0 of 7 properties failed, 7 passed\n"
                         "VERIFICATION SUCCESSFUL\n",
                         wrapper.Property.py_arithmetic), "TRUE")


# The --multi-property re-run of two SV-COMP 2027 Python tasks, trimmed. ESBMC
# pads a row's line number to three columns, and lists a module-level
# property under a bare path line rather than a ", function" header.
MULTI_FUNCTION_ROWS = """
Violated property:
  file ../../sv-benchmarks/python/vllm/vllm_check_stop_buggy.py line 74 column 4 function main
  assertion num_tokens < max_model_len
  num_tokens < max_model_len

Violated property:
  file ../../sv-benchmarks/python/vllm/vllm_check_stop_buggy.py line 75 column 4 function main
  assertion num_output_tokens < max_tokens
  num_output_tokens < max_tokens

** Results:
../../sv-benchmarks/python/vllm/vllm_check_stop_buggy.py
  PASSED       [global.assertion.1]                                line   0  uncaught exception
../../sv-benchmarks/python/vllm/vllm_check_stop_buggy.py, function main
  PASSED       [main.assertion.1]                                  line  58  assertion num_output_tokens >= 1
  FAILED       [main.assertion.2]                                  line  74  assertion num_tokens < max_model_len
  FAILED       [main.assertion.3]                                  line  75  assertion num_output_tokens < max_tokens
/esbmc-vfs/libc/library/string.c, function memcpy
  PASSED       [memcpy.null-pointer-dereference.1]                 line 343  dereference failure: NULL pointer on DST

** 2 of 67 properties failed, 65 passed

VERIFICATION FAILED
"""

MULTI_MODULE_ROWS = """
Violated property:
  file ../../sv-benchmarks/python/type-eval-py/returns_return_lambda_unsafe/main.py line 27 column 0
  assertion a >= 0
  a >= 0

Violated property:
  file ../../sv-benchmarks/python/type-eval-py/returns_return_lambda_unsafe/main.py line 28 column 0
  assertion a == n * n
  a == n * n

** Results:
../../sv-benchmarks/python/type-eval-py/returns_return_lambda_unsafe/main.py
  PASSED       [global.assertion.1]                                line   0  uncaught exception
  PASSED       [global.assertion.2]                                line  24  dereference failure: Program code accessed with non-zero offset
  FAILED       [global.assertion.3]                                line  27  assertion a >= 0
  FAILED       [global.assertion.4]                                line  28  assertion a == n * n
  PASSED       [global.assertion.5]                                line  30  assertion b >= 0.25
/esbmc-vfs/libc/library/string.c, function memcpy
  PASSED       [memcpy.null-pointer-dereference.1]                 line 343  dereference failure: NULL pointer on DST

** 2 of 68 properties failed, 66 passed

VERIFICATION FAILED
"""


def single_run(path, line, comment, function=None):
    """A run that stops at its first violation, shaped like ESBMC's output."""
    where = " column 4 function " + function if function else " column 0"
    header = path + (", function " + function if function else "")
    claim = (function or "global") + ".assertion.1"
    return ("\nViolated property:\n  file %s line %s%s\n  %s\n\n** Results:\n%s\n"
            "  FAILED       [%s]  line %3s  %s\n\nVERIFICATION FAILED\n"
            % (path, line, where, comment, header, claim, line, comment))


class FailedSitesTest(unittest.TestCase):
    """failed_sites reads every FAILED row of a --multi-property results table."""

    def test_padded_rows_under_a_function_header(self):
        self.assertEqual(wrapper.failed_sites(MULTI_FUNCTION_ROWS),
                         {("vllm_check_stop_buggy.py", "74"), ("vllm_check_stop_buggy.py", "75")})

    def test_padded_rows_under_a_module_header(self):
        self.assertEqual(wrapper.failed_sites(MULTI_MODULE_ROWS), {("main.py", "27"), ("main.py", "28")})

    def test_an_unpadded_three_digit_line(self):
        failed = MULTI_FUNCTION_ROWS.replace("  PASSED       [memcpy.", "  FAILED       [memcpy.")
        self.assertIn(("string.c", "343"), wrapper.failed_sites(failed))

    def test_passed_rows_are_not_sites(self):
        sites = wrapper.failed_sites(MULTI_FUNCTION_ROWS) | wrapper.failed_sites(MULTI_MODULE_ROWS)
        for passed in ("0", "24", "30", "58", "343"):
            self.assertNotIn(passed, {line for _, line in sites})


class IntegerEncodingConfirmationTest(unittest.TestCase):
    """A Python violation stays FALSE only if a --ir re-run repeats it in place."""

    PASSED = "** 0 of 1 properties failed, 1 passed\n\nVERIFICATION SUCCESSFUL\n"
    TIMED_OUT = b"ERROR: Timed out\n"
    NO_LINE = PY_ASSERTION.replace(" line 5 column 4", "")

    def verify(self, prop, first, *reruns, benchmark="main.py", elapsed=0, plain_seconds=0):
        outputs = iter([first, *reruns])
        calls = []

        def fake_run(cmd_line, timeout=None):
            calls.append((cmd_line, timeout))
            out = next(outputs)
            return out if isinstance(out, bytes) else out.encode()

        with mock.patch.object(wrapper, "run", fake_run), \
                mock.patch.object(wrapper.time, "monotonic", side_effect=[0, elapsed, 0, plain_seconds]):
            result = wrapper.verify("kinduction", prop, 64, benchmark, "", False, None, None)
        return wrapper.get_result_string(result), calls

    def test_rerun_at_the_same_line_keeps_false(self):
        verdict, calls = self.verify(wrapper.Property.py_assertion, PY_ASSERTION, PY_ASSERTION)
        self.assertEqual(verdict, "FALSE")
        self.assertTrue(calls[1][0].endswith("--ir --z3 "))
        self.assertEqual(calls[1][1], wrapper.IR_RERUN_TIMEOUT)
        self.assertIsNone(calls[0][1])
        self.assertEqual(len(calls), 2)

    def test_a_first_run_leaving_the_full_budget_is_rerun(self):
        verdict, calls = self.verify(wrapper.Property.py_assertion, PY_ASSERTION, self.PASSED, elapsed=700)
        self.assertEqual(verdict, "Unknown")
        self.assertEqual(calls[1][1], wrapper.IR_RERUN_TIMEOUT)

    def test_a_first_run_past_the_threshold_is_unknown_without_a_rerun(self):
        elapsed = wrapper.TOTAL_BUDGET - wrapper.MIN_RERUN_BUDGET + 1
        verdict, calls = self.verify(wrapper.Property.py_assertion, PY_ASSERTION, self.PASSED, elapsed=elapsed)
        self.assertEqual(verdict, "Unknown")
        self.assertEqual(len(calls), 1)

    def test_a_first_run_at_the_threshold_is_still_rerun(self):
        elapsed = wrapper.TOTAL_BUDGET - wrapper.MIN_RERUN_BUDGET
        verdict, calls = self.verify(wrapper.Property.py_assertion, PY_ASSERTION, self.PASSED, elapsed=elapsed)
        self.assertEqual(verdict, "Unknown")
        self.assertEqual(calls[1][1], wrapper.MIN_RERUN_BUDGET)

    def test_rerun_that_passes_is_unknown_without_a_fallback(self):
        verdict, calls = self.verify(wrapper.Property.py_assertion, PY_ASSERTION, self.PASSED)
        self.assertEqual(verdict, "Unknown")
        self.assertEqual(len(calls), 2)

    OTHER_LINE = PY_ASSERTION.replace("line 5", "line 9")
    BOTH_LINES = PY_ASSERTION.replace("line 5 column 4", "line 9 column 4").replace(
        "  FAILED       [main.assertion.1]  line 5  assertion x != 3\n",
        "  FAILED       [main.assertion.1]  line 5  assertion x != 3\n"
        "  FAILED       [main.assertion.2]  line 9  assertion y != 4\n")

    def test_rerun_failing_at_another_line_is_unknown(self):
        verdict, calls = self.verify(wrapper.Property.py_assertion, PY_ASSERTION, self.OTHER_LINE, self.OTHER_LINE)
        self.assertEqual(verdict, "Unknown")
        self.assertTrue(calls[2][0].endswith("--ir --z3 --multi-property "))

    def test_a_fallback_failing_both_lines_keeps_the_first_runs_site(self):
        verdict, calls = self.verify(wrapper.Property.py_assertion, PY_ASSERTION, self.OTHER_LINE, self.BOTH_LINES,
                                     plain_seconds=10)
        self.assertEqual(verdict, "FALSE")
        self.assertTrue(calls[1][0].endswith("--ir --z3 "))
        self.assertTrue(calls[2][0].endswith("--ir --z3 --multi-property "))
        self.assertEqual(calls[2][1], wrapper.IR_RERUN_TIMEOUT - 10)

    VLLM = "../../sv-benchmarks/python/vllm/vllm_check_stop_buggy.py"
    LAMBDA = "../../sv-benchmarks/python/type-eval-py/returns_return_lambda_unsafe/main.py"

    def test_a_fallback_confirms_a_padded_function_row(self):
        first = single_run(self.VLLM, "75", "assertion num_output_tokens < max_tokens", "main")
        plain = single_run(self.VLLM, "74", "assertion num_tokens < max_model_len", "main")
        verdict, _ = self.verify(wrapper.Property.py_assertion, first, plain, MULTI_FUNCTION_ROWS)
        self.assertEqual(verdict, "FALSE")

    def test_a_fallback_confirms_a_padded_module_row(self):
        first = single_run(self.LAMBDA, "28", "assertion a == n * n")
        plain = single_run(self.LAMBDA, "27", "assertion a >= 0")
        verdict, _ = self.verify(wrapper.Property.py_assertion, first, plain, MULTI_MODULE_ROWS)
        self.assertEqual(verdict, "FALSE")

    def test_a_fallback_without_the_first_runs_site_failed_is_unknown(self):
        first = single_run(self.VLLM, "58", "assertion num_output_tokens >= 1", "main")
        plain = single_run(self.VLLM, "74", "assertion num_tokens < max_model_len", "main")
        verdict, _ = self.verify(wrapper.Property.py_assertion, first, plain, MULTI_FUNCTION_ROWS)
        self.assertEqual(verdict, "Unknown")

    def test_no_budget_left_for_the_fallback_is_unknown(self):
        plain_seconds = wrapper.IR_RERUN_TIMEOUT - wrapper.MIN_RERUN_BUDGET + 1
        verdict, calls = self.verify(wrapper.Property.py_assertion, PY_ASSERTION, self.OTHER_LINE, self.BOTH_LINES,
                                     plain_seconds=plain_seconds)
        self.assertEqual(verdict, "Unknown")
        self.assertEqual(len(calls), 2)

    def test_a_fallback_at_the_threshold_is_still_made(self):
        plain_seconds = wrapper.IR_RERUN_TIMEOUT - wrapper.MIN_RERUN_BUDGET
        verdict, calls = self.verify(wrapper.Property.py_assertion, PY_ASSERTION, self.OTHER_LINE, self.BOTH_LINES,
                                     plain_seconds=plain_seconds)
        self.assertEqual(verdict, "FALSE")
        self.assertEqual(calls[2][1], wrapper.MIN_RERUN_BUDGET)

    def test_rerun_timeout_is_unknown(self):
        verdict, calls = self.verify(wrapper.Property.py_assertion, PY_ASSERTION, self.TIMED_OUT,
                                     plain_seconds=wrapper.IR_RERUN_TIMEOUT)
        self.assertEqual(verdict, "Unknown")
        self.assertEqual(len(calls), 2)

    def test_rerun_failing_another_property_is_unknown(self):
        same_site_valueerror = PY_VALUEERROR.replace("p_raise_value.py", "p_assert2.py").replace("line 6 column 8", "line 5 column 4")
        verdict, _ = self.verify(wrapper.Property.py_assertion, PY_ASSERTION, same_site_valueerror, same_site_valueerror)
        self.assertEqual(verdict, "Unknown")

    def test_no_line_on_either_side_matches_on_the_file(self):
        verdict, _ = self.verify(wrapper.Property.py_assertion, self.NO_LINE, self.NO_LINE)
        self.assertEqual(verdict, "FALSE")

    def test_a_line_on_one_side_only_is_unknown(self):
        verdict, _ = self.verify(wrapper.Property.py_assertion, self.NO_LINE, PY_ASSERTION, PY_ASSERTION)
        self.assertEqual(verdict, "Unknown")

    def test_a_violation_without_a_site_is_unknown(self):
        table_only = PY_ASSERTION.split("** Results:")[1]
        table_only = "** Results:" + table_only
        verdict, _ = self.verify(wrapper.Property.py_assertion, table_only, table_only)
        self.assertEqual(verdict, "Unknown")

    def test_a_true_python_verdict_is_not_rerun(self):
        verdict, calls = self.verify(wrapper.Property.py_assertion, self.PASSED, PY_ASSERTION)
        self.assertEqual(verdict, "TRUE")
        self.assertEqual(len(calls), 1)

    def test_a_c_violation_is_never_rerun(self):
        with tempfile.NamedTemporaryFile("w", suffix=".c") as c_file:
            verdict, calls = self.verify(wrapper.Property.reach, REACH_WITH_UNCHECKED_UNWINDING, self.PASSED, benchmark=c_file.name)
        self.assertEqual(verdict, "FALSE_REACH")
        self.assertEqual(len(calls), 1)


def write_task(directory, source, files=None):
    """Write main.py, and any other files, into a task directory."""
    for name, text in {"main.py": source, **(files or {})}.items():
        with open(os.path.join(directory, name), "w", encoding="utf-8") as f:
            f.write(text)


class FakeEsbmc:
    """Stands in for wrapper.run(): answers the first run, the values run and
    the --ir re-run, and records each command line and its timeout."""

    def __init__(self, first, ir, witness, values_run):
        self.first, self.ir, self.witness, self.values_run = first, ir, witness, values_run
        self.calls = []
        self.timeouts = []

    def __call__(self, cmd_line, timeout=None):
        self.calls.append(cmd_line)
        self.timeouts.append(timeout)
        words = shlex.split(cmd_line)
        if "--pytest-output-dir" not in words:
            return (self.ir if "--ir" in cmd_line else self.first).encode()
        if self.witness is not None:
            values_dir = words[words.index("--pytest-output-dir") + 1]
            with open(os.path.join(values_dir, "test_main.py"), "w", encoding="utf-8") as f:
                f.write(self.witness)
        return (self.values_run or self.first).encode()


class CPythonReplayTest(unittest.TestCase):
    """A FALSE stays FALSE when CPython, fed the counterexample's values, ends
    with the property's exception; anything else leaves it to the --ir re-run."""

    PASSED = IntegerEncodingConfirmationTest.PASSED
    HEADER = "import _sv_verifier\n"

    def verify(self, prop, first, source, witness, **options):
        """Run verify() on a task whose first ESBMC run says FALSE with `witness`
        as its recorded values; returns the verdict and the ESBMC command lines.
        Options: ir, files, verifier, values_run."""
        esbmc = FakeEsbmc(first, self.PASSED if options.get("ir") is None else options["ir"],
                          witness, options.get("values_run"))
        with tempfile.TemporaryDirectory() as task:
            write_task(task, source, options.get("files"))
            with mock.patch.object(wrapper, "run", esbmc), \
                    mock.patch.object(wrapper, "REPLAY_VERIFIER", options.get("verifier") or wrapper.REPLAY_VERIFIER):
                result = wrapper.verify("kinduction", prop, 64, os.path.join(task, "main.py"),
                                        "", False, None, None)
        return wrapper.get_result_string(result), esbmc.calls

    def ir_calls(self, calls):
        return [c for c in calls if c.endswith("--ir --z3 ")]

    def test_a_replay_ending_in_the_family_confirms_without_the_ir_rerun(self):
        verdict, calls = self.verify(
            wrapper.Property.py_datalookup, PY_INDEXERROR,
            self.HEADER + "xs = _sv_verifier.nondet_list(_sv_verifier.nondet_int)\nxs[3]\n",
            "witness = [\n    {'list0': [1]},\n]\n")
        self.assertEqual(verdict, "FALSE")
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(self.ir_calls(calls)), 0)

    def test_a_replay_ending_normally_falls_back_to_the_ir_rerun(self):
        verdict, calls = self.verify(
            wrapper.Property.py_datalookup, PY_INDEXERROR,
            self.HEADER + "xs = _sv_verifier.nondet_list(_sv_verifier.nondet_int)\nxs[0]\n",
            "witness = [\n    {'list0': [1]},\n]\n")
        self.assertEqual(verdict, "Unknown")
        self.assertEqual(len(self.ir_calls(calls)), 1)

    def test_a_fallback_that_confirms_keeps_false(self):
        verdict, calls = self.verify(
            wrapper.Property.py_datalookup, PY_INDEXERROR,
            self.HEADER + "xs = _sv_verifier.nondet_list(_sv_verifier.nondet_int)\nxs[0]\n",
            "witness = [\n    {'list0': [1]},\n]\n", ir=PY_INDEXERROR)
        self.assertEqual(verdict, "FALSE")
        self.assertEqual(len(self.ir_calls(calls)), 1)

    def test_an_exception_of_another_family_does_not_confirm(self):
        verdict, calls = self.verify(
            wrapper.Property.py_datalookup, PY_INDEXERROR,
            self.HEADER + "x = _sv_verifier.nondet_int()\n1 + 'a'\n",
            "witness = [\n    {'arg0': 1},\n]\n")
        self.assertEqual(verdict, "Unknown")
        self.assertEqual(len(self.ir_calls(calls)), 1)

    def test_the_same_replay_confirms_only_its_own_property(self):
        source = self.HEADER + "x = _sv_verifier.nondet_int()\n[][x]\n"
        witness = "witness = [\n    {'arg0': 1},\n]\n"
        verdict, calls = self.verify(wrapper.Property.py_arithmetic, PY_ZERODIVISION, source, witness)
        self.assertEqual(verdict, "Unknown")
        self.assertEqual(len(self.ir_calls(calls)), 1)

    def test_an_overflow_error_from_a_size_limit_does_not_confirm(self):
        verdict, calls = self.verify(
            wrapper.Property.py_datalookup, PY_INDEXERROR,
            self.HEADER + "n = _sv_verifier.nondet_int()\n[0] * n\n",
            "witness = [\n    {'arg0': 9223372036854775807},\n]\n")
        self.assertEqual(verdict, "Unknown")
        self.assertEqual(len(self.ir_calls(calls)), 1)

    def test_a_witness_with_too_few_values_is_inconclusive_even_if_caught(self):
        source = self.HEADER + (
            "try:\n    a = _sv_verifier.nondet_int()\n    b = _sv_verifier.nondet_int()\n"
            "except BaseException:\n    pass\n[][0]\n")
        verdict, calls = self.verify(
            wrapper.Property.py_datalookup, PY_INDEXERROR, source, "witness = [\n    {'arg0': 1},\n]\n")
        self.assertEqual(verdict, "Unknown")
        self.assertEqual(len(self.ir_calls(calls)), 1)

    def test_a_missing_bool_is_completed_from_a_bit_string(self):
        source = self.HEADER + (
            "s = _sv_verifier.nondet_str()\nif _sv_verifier.nondet_bool():\n    d = {}\n    d[s]\n")
        verdict, calls = self.verify(
            wrapper.Property.py_datalookup, PY_INDEXERROR, source, "witness = [\n    {'arg0': 'k'},\n]\n")
        self.assertEqual(verdict, "FALSE")
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(self.ir_calls(calls)), 0)

    def test_a_constant_bool_tail_covers_a_loop_asking_for_a_bool_each_pass(self):
        source = self.HEADER + (
            "n = 0\nwhile _sv_verifier.nondet_bool():\n    n += 1\n    if n > 5:\n        [][0]\n")
        verdict, calls = self.verify(
            wrapper.Property.py_datalookup, PY_INDEXERROR, source, "witness = [\n    {},\n]\n")
        self.assertEqual(verdict, "FALSE")
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(self.ir_calls(calls)), 0)

    def test_a_scalar_is_taken_by_type_when_the_witness_order_differs(self):
        source = self.HEADER + (
            "x = _sv_verifier.nondet_int()\nf = _sv_verifier.nondet_float()\n1 / (x - 7)\n")
        verdict, calls = self.verify(
            wrapper.Property.py_arithmetic, PY_ZERODIVISION, source,
            "witness = [\n    {'arg0': 1.5, 'arg1': 7},\n]\n")
        self.assertEqual(verdict, "FALSE")
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(self.ir_calls(calls)), 0)

    def test_a_missing_witness_is_an_empty_one(self):
        verdict, calls = self.verify(
            wrapper.Property.py_datalookup, PY_INDEXERROR, "[][0]\n", None)
        self.assertEqual(verdict, "FALSE")
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(self.ir_calls(calls)), 0)

    def test_a_program_that_asks_for_a_value_of_an_empty_witness_is_inconclusive(self):
        verdict, calls = self.verify(
            wrapper.Property.py_datalookup, PY_INDEXERROR,
            self.HEADER + "x = _sv_verifier.nondet_int()\n[][0]\n", None)
        self.assertEqual(verdict, "Unknown")
        self.assertEqual(len(self.ir_calls(calls)), 1)

    def test_a_non_printable_string_is_inconclusive(self):
        verdict, calls = self.verify(
            wrapper.Property.py_datalookup, PY_INDEXERROR,
            self.HEADER + "s = _sv_verifier.nondet_str()\n[][0]\n",
            "witness = [\n    {'arg0': '\\x01'},\n]\n")
        self.assertEqual(verdict, "Unknown")
        self.assertEqual(len(self.ir_calls(calls)), 1)

    def test_a_float_outside_its_range_is_inconclusive(self):
        verdict, calls = self.verify(
            wrapper.Property.py_datalookup, PY_INDEXERROR,
            self.HEADER + "f = _sv_verifier.nondet_float_range(0.0, 1.0)\n[][0]\n",
            "witness = [\n    {'arg0': 5.0},\n]\n")
        self.assertEqual(verdict, "Unknown")
        self.assertEqual(len(self.ir_calls(calls)), 1)

    def test_a_task_with_its_own_verifier_module_is_not_replayed(self):
        with mock.patch.object(wrapper, "replay_once", side_effect=AssertionError("replayed")):
            verdict, calls = self.verify(
                wrapper.Property.py_datalookup, PY_INDEXERROR, "[][0]\n", None,
                files={"_sv_verifier.py": "def nondet_int():\n    return 0\n"})
        self.assertEqual(verdict, "Unknown")
        self.assertEqual(len(self.ir_calls(calls)), 1)

    def test_a_replay_that_does_not_end_falls_back_within_its_budget(self):
        started = time.monotonic()
        with mock.patch.object(wrapper, "REPLAY_BUDGET", 3):
            verdict, calls = self.verify(
                wrapper.Property.py_datalookup, PY_INDEXERROR, "while True:\n    pass\n", None)
        self.assertEqual(verdict, "Unknown")
        self.assertEqual(len(self.ir_calls(calls)), 1)
        self.assertLess(time.monotonic() - started, 15)

    @unittest.skipUnless(hasattr(__builtins__ if isinstance(__builtins__, dict) is False else None, "ExceptionGroup")
                         or "ExceptionGroup" in dir(__import__("builtins")), "needs Python 3.11")
    def test_an_exception_group_is_not_read_as_its_members(self):
        verdict, calls = self.verify(
            wrapper.Property.py_datalookup, PY_INDEXERROR,
            "class G(ExceptionGroup, IndexError):\n    pass\n\n\nraise G('g', [IndexError('x')])\n", None)
        self.assertEqual(verdict, "Unknown")
        self.assertEqual(len(self.ir_calls(calls)), 1)

    def test_a_missing_python_falls_back(self):
        with mock.patch.object(wrapper.sys, "executable", "/nonexistent/python3"):
            verdict, calls = self.verify(wrapper.Property.py_datalookup, PY_INDEXERROR, "[][0]\n", None)
        self.assertEqual(verdict, "Unknown")
        self.assertEqual(len(self.ir_calls(calls)), 1)

    def test_no_budget_left_skips_the_replay(self):
        with mock.patch.object(wrapper, "replay_once", side_effect=AssertionError("replayed")), \
                mock.patch.object(wrapper, "TOTAL_BUDGET", 0):
            verdict, calls = self.verify(wrapper.Property.py_datalookup, PY_INDEXERROR, "[][0]\n", None)
        self.assertEqual(verdict, "Unknown")
        self.assertEqual(len(self.ir_calls(calls)), 0)

    def test_the_wrapper_is_one_file_holding_the_replay_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            written = wrapper.write_replay_files(os.path.join(directory, "replay"))
            self.assertEqual(sorted(os.listdir(written)), ["_sv_verifier.py", "launch.py"])
            for name in os.listdir(written):
                with open(os.path.join(written, name), encoding="utf-8") as f:
                    compile(f.read(), name, "exec")

    def test_the_first_run_never_carries_the_pytest_flags(self):
        _, calls = self.verify(wrapper.Property.py_datalookup, PY_INDEXERROR, "[][0]\n", None)
        self.assertEqual(sum(1 for c in calls if "--pytest" not in c and "--ir" not in c), 1)
        self.assertNotIn("--pytest", calls[0])

    def test_a_values_run_that_does_not_find_the_violation_falls_back_to_ir(self):
        verdict, calls = self.verify(
            wrapper.Property.py_datalookup, PY_INDEXERROR, "[][0]\n", None, values_run=self.PASSED)
        self.assertEqual(verdict, "Unknown")
        self.assertEqual(len(self.ir_calls(calls)), 1)

    def test_a_values_run_is_skipped_when_it_would_leave_no_budget_for_the_ir_rerun(self):
        esbmc = FakeEsbmc(PY_INDEXERROR, self.PASSED, None, None)
        with tempfile.TemporaryDirectory() as task, mock.patch.object(wrapper, "run", esbmc):
            write_task(task, "[][0]\n")
            wrapper.verify_python("esbmc ", wrapper.Property.py_datalookup, os.path.join(task, "main.py"),
                                  time.monotonic() - (wrapper.TOTAL_BUDGET - wrapper.IR_RERUN_TIMEOUT - 1))
        self.assertEqual(len(esbmc.calls), 2)
        self.assertNotIn("--pytest", "".join(esbmc.calls))
        self.assertTrue(esbmc.calls[1].endswith("--ir --z3 "))
        self.assertEqual(esbmc.timeouts[1], wrapper.IR_RERUN_TIMEOUT)

    def test_only_python_runs_ask_for_the_values(self):
        _, calls = self.verify(wrapper.Property.py_datalookup, PY_INDEXERROR, "[][0]\n", None)
        self.assertNotIn("--pytest", calls[0])
        self.assertIn("--generate-pytest-testcase --pytest-values-only --pytest-output-dir", calls[1])
        self.assertNotIn("--pytest", "".join(self.ir_calls(calls)))
        with tempfile.NamedTemporaryFile("w", suffix=".c") as c_file, \
                mock.patch.object(wrapper, "confirmed_by_replay", side_effect=AssertionError("replayed")):
            esbmc = FakeEsbmc(REACH_WITH_UNCHECKED_UNWINDING, self.PASSED, None, None)
            with mock.patch.object(wrapper, "run", esbmc):
                result = wrapper.verify("kinduction", wrapper.Property.reach, 64, c_file.name,
                                        "", False, None, None)
        self.assertEqual(wrapper.get_result_string(result), "FALSE_REACH")
        self.assertNotIn("--pytest", esbmc.calls[0])

    def test_a_witness_validation_run_does_not_replay(self):
        with tempfile.NamedTemporaryFile("w", suffix=".py") as py_file, \
                mock.patch.object(wrapper, "confirmed_by_replay", side_effect=AssertionError("replayed")):
            esbmc = FakeEsbmc(PY_INDEXERROR, self.PASSED, None, None)
            with mock.patch.object(wrapper, "run", esbmc):
                wrapper.verify("incr", wrapper.Property.py_datalookup, 64, py_file.name, "",
                               False, "w.yml", "violation")
        self.assertNotIn("--pytest", esbmc.calls[0])

    # Negative controls: a stand-in that ignores the witness, or a replay judged
    # under another property, must not confirm what the tests above confirm.
    def mutated_stand_in(self, old, new):
        self.assertIn(old, wrapper.REPLAY_VERIFIER)
        return wrapper.REPLAY_VERIFIER.replace(old, new, 1)

    def test_control_a_stand_in_returning_zero_does_not_confirm_the_scalar_case(self):
        stand_in = self.mutated_stand_in(
            "    return _pop_typed(lambda v: type(v) is int)", "    return 0")
        source = self.HEADER + "x = _sv_verifier.nondet_int()\nf = _sv_verifier.nondet_float()\n1 / (x - 7)\n"
        verdict, _ = self.verify(
            wrapper.Property.py_arithmetic, PY_ZERODIVISION, source,
            "witness = [\n    {'arg0': 1.5, 'arg1': 7},\n]\n", verifier=stand_in)
        self.assertEqual(verdict, "Unknown")

    def test_control_a_stand_in_without_the_bool_search_does_not_confirm_the_bool_case(self):
        stand_in = self.mutated_stand_in(
            "    if not _bits:\n        if _fill:", "    if not _bits:\n        if False:")
        source = self.HEADER + (
            "n = 0\nwhile _sv_verifier.nondet_bool():\n    n += 1\n    if n > 5:\n        [][0]\n")
        verdict, _ = self.verify(
            wrapper.Property.py_datalookup, PY_INDEXERROR, source, "witness = [\n    {},\n]\n",
            verifier=stand_in)
        self.assertEqual(verdict, "Unknown")


class PythonCommandLineTest(unittest.TestCase):
    """Each run reports only its property's exception family."""

    def command(self, prop):
        return wrapper.get_command_line("kinduction", prop, None, "main.py", "", False)

    def test_each_property_checks_its_own_family(self):
        cases = (
            (wrapper.Property.py_assertion, "AssertionError"),
            (wrapper.Property.py_arithmetic, "ZeroDivisionError,FloatingPointError"),
            (wrapper.Property.py_datalookup, "KeyError,IndexError"),
            (wrapper.Property.py_dyntyping, "TypeError,AttributeError"),
        )
        for prop, family in cases:
            self.assertIn("--python-check-exceptions " + family + " ", self.command(prop))

    def test_the_multi_property_rerun_gets_sequential_k_induction(self):
        cmd = self.command(wrapper.Property.py_assertion)
        self.assertIn("--k-induction ", cmd)
        self.assertNotIn("--k-induction-parallel", cmd)


class PythonPropertyFileTest(unittest.TestCase):
    """python/properties/*.prp in sv-benchmarks, verbatim."""

    def test_each_property_file_is_recognised(self):
        cases = (
            ("CHECK( init(main()), ! uncaught(AssertionError))\n",
             wrapper.Property.py_assertion),
            ("CHECK( init(main()), ! uncaught(ZeroDivisionError))\n"
             "CHECK( init(main()), ! uncaught(FloatingPointError))\n",
             wrapper.Property.py_arithmetic),
            ("CHECK( init(main()), ! uncaught(KeyError))\n"
             "CHECK( init(main()), ! uncaught(IndexError))\n",
             wrapper.Property.py_datalookup),
            ("CHECK( init(main()), ! uncaught(TypeError))\n"
             "CHECK( init(main()), ! uncaught(AttributeError))\n",
             wrapper.Property.py_dyntyping),
        )
        for content, expected in cases:
            self.assertEqual(wrapper.python_property(content), expected)

    def test_a_c_property_file_is_not_a_python_property(self):
        self.assertIsNone(
            wrapper.python_property("CHECK( init(main()), LTL(G ! overflow) )\n"))

    def test_an_unknown_exception_family_is_not_matched(self):
        self.assertIsNone(
            wrapper.python_property("CHECK( init(main()), ! uncaught(OSError))\n"))


class PythonModulePathTest(unittest.TestCase):
    """A task in python/<project>/ imports _sv_verifier from python/.

    CPython resolves an import against the script's own directory, so the task
    fails there too unless PYTHONPATH carries the parent; sv-benchmarks' own
    check-syntax.py sets it for that reason. ESBMC follows CPython, so the
    wrapper supplies the search path.
    """

    def setUp(self):
        self.saved = os.environ.get("PYTHONPATH")

    def tearDown(self):
        if self.saved is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = self.saved

    def test_task_directory_and_its_parent_are_both_reachable(self):
        os.environ.pop("PYTHONPATH", None)
        wrapper.add_python_module_path("/sv-benchmarks/python/vllm/vllm_cdiv.py")
        self.assertEqual(
            os.environ["PYTHONPATH"].split(os.pathsep),
            ["/sv-benchmarks/python/vllm", "/sv-benchmarks/python"])

    def test_an_existing_pythonpath_is_kept(self):
        os.environ["PYTHONPATH"] = "/already/here"
        wrapper.add_python_module_path("/sv-benchmarks/python/vllm/vllm_cdiv.py")
        self.assertEqual(
            os.environ["PYTHONPATH"].split(os.pathsep)[-1], "/already/here")

    def test_a_task_nested_deeper_reaches_the_verifier_module(self):
        os.environ.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory() as root:
            python = os.path.join(root, "python")
            task = os.path.join(python, "type-eval-py", "args_assigned_call")
            os.makedirs(task)
            with open(os.path.join(python, "_sv_verifier.py"), "w", encoding="utf-8"):
                pass
            wrapper.add_python_module_path(os.path.join(task, "main.py"))
            self.assertIn(python, os.environ["PYTHONPATH"].split(os.pathsep))

    def test_a_relative_benchmark_path_is_made_absolute(self):
        os.environ.pop("PYTHONPATH", None)
        wrapper.add_python_module_path("boto3_all_not_none.py")
        for entry in os.environ["PYTHONPATH"].split(os.pathsep):
            self.assertTrue(os.path.isabs(entry), entry)


class FallThroughTest(unittest.TestCase):
    """What a violation that decides nothing for the property under check does.

    These pin behaviour the chain in parse_result had only implicitly: a
    violated property that says nothing about the property being checked falls
    through to the SUCCESSFUL test, so an output carrying both verdicts is
    read as TRUE. Characterisation tests, so a refactoring cannot move it.
    """

    def verdict(self, output, prop):
        return wrapper.get_result_string(wrapper.parse_result(output, prop))

    LEAKLESS_FAILURE = """
Violated property:
  file main.c line 7 column 3 function main
  dereference failure: invalid pointer

** Results:
main.c, function main
  FAILED       [main.deref.1]  line 7  dereference failure: invalid pointer

VERIFICATION FAILED
"""

    def test_memcleanup_without_a_leak_is_inconclusive(self):
        self.assertEqual(
            self.verdict(self.LEAKLESS_FAILURE, wrapper.Property.memcleanup),
            "Unknown")

    def test_a_run_carrying_both_verdicts_falls_through_to_true(self):
        self.assertEqual(
            self.verdict(self.LEAKLESS_FAILURE + "VERIFICATION SUCCESSFUL\n",
                         wrapper.Property.memcleanup),
            "TRUE")

    def test_reach_on_the_unreachability_intrinsic_is_inconclusive(self):
        output = """
Violated property:
  file main.c line 7 column 3 function main
  reachability: unreachable code reached

** Results:
main.c, function main
  FAILED       [main.assertion.1]  line 7  reachability: unreachable code reached

VERIFICATION FAILED
"""
        self.assertEqual(self.verdict(output, wrapper.Property.reach), "Unknown")

    def test_memory_on_a_comment_no_rule_matches_is_inconclusive(self):
        output = """
Violated property:
  file main.c line 7 column 3 function main
  unwinding assertion loop 3 is not what this tests

** Results:
main.c, function main
  FAILED       [main.assertion.1]  line 7  some comment no rule names

VERIFICATION FAILED
"""
        self.assertEqual(self.verdict(output, wrapper.Property.memory), "Unknown")


# Test-Comp. Shaped like ESBMC 8.6 output on reducercommutativity/rangesum10.i
# in Test-Comp 2027.
TESTCOMP_REACH_WITH_UNCHECKED_UNWINDING = """
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

# --quiet, which coverage mode passes, drops the counterexample and with it the
# "Violated property:" block, leaving only the table row.
TESTCOMP_QUIET_VIOLATED_UNWINDING = """
** Results:
main.c, function main
  FAILED       [main.assertion.1]  line 42  unwinding assertion loop 3

VERIFICATION FAILED
"""

TESTCOMP_SUCCESSFUL = """
** Results:
main.c, function main
  PASSED       [main.assertion.1]  line 42  unwinding assertion loop 3

VERIFICATION SUCCESSFUL
"""

COVERAGE_BRANCHES = "COVER( init(main()), FQL(COVER EDGES(@DECISIONEDGE)) )\n"
COVERAGE_CONDITIONS = "COVER( init(main()), FQL(COVER EDGES(@CONDITIONEDGE)) )\n"
COVERAGE_STATEMENTS = "COVER( init(main()), FQL(COVER EDGES(@BASICBLOCKENTRY)) )\n"
COVERAGE_ERROR_CALL = "COVER( init(main()), FQL(COVER EDGES(@CALL(reach_error))) )\n"
UNREACH_CALL = "CHECK( init(main()), LTL(G ! call(reach_error())) )\n"


class TestCompParseResultTest(unittest.TestCase):

    def verdict(self, output):
        return wrapper.get_result_string(wrapper.testcomp_parse_result(output))

    def test_unchecked_unwinding_row_does_not_mask_the_reach_violation(self):
        self.assertEqual(self.verdict(TESTCOMP_REACH_WITH_UNCHECKED_UNWINDING), "FALSE_REACH")

    def test_violated_unwinding_assertion_is_unknown(self):
        self.assertEqual(self.verdict(VIOLATED_UNWINDING), "Unknown")

    def test_quiet_violated_unwinding_assertion_is_unknown(self):
        self.assertEqual(self.verdict(TESTCOMP_QUIET_VIOLATED_UNWINDING), "Unknown")

    def test_successful_run_is_true(self):
        self.assertEqual(self.verdict(TESTCOMP_SUCCESSFUL), "TRUE")

    def test_timeout(self):
        self.assertEqual(self.verdict("ERROR: Timed out\n"), "Timed out")


def dry_run(prp_content, *options):
    # -n prints the ESBMC command and stops before running it
    with tempfile.TemporaryDirectory() as d:
        prp = os.path.join(d, "goal.prp")
        task = os.path.join(d, "task.c")
        with open(prp, "w") as f:
            f.write(prp_content)
        with open(task, "w") as f:
            f.write("int main() { return 0; }\n")
        return subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "esbmc-wrapper.py"),
             "-n", "-p", prp, "-s", "kinduction", *options, task],
            capture_output=True, text=True, check=True, cwd=d).stdout


class TestCompRunTest(unittest.TestCase):
    """benchexec's tool-info runs esbmc-wrapper.py for SV-COMP and Test-Comp alike."""

    def test_coverage_option_or_cover_property_is_testcomp(self):
        self.assertTrue(wrapper.is_testcomp_run("branch-function", UNREACH_CALL))
        for content in (COVERAGE_BRANCHES, COVERAGE_ERROR_CALL):
            self.assertTrue(wrapper.is_testcomp_run(None, content))

    def test_svcomp_properties_are_not_testcomp(self):
        self.assertFalse(wrapper.is_testcomp_run(None, UNREACH_CALL))
        self.assertFalse(wrapper.is_testcomp_run(None, "CHECK( init(main()), LTL(G valid-free) )\n"))

    def test_goals(self):
        for content in (COVERAGE_BRANCHES, COVERAGE_CONDITIONS, COVERAGE_STATEMENTS):
            self.assertEqual(wrapper.test_goal(content), wrapper.TestGoal.coverage)
        for content in (COVERAGE_ERROR_CALL, UNREACH_CALL):
            self.assertEqual(wrapper.test_goal(content), wrapper.TestGoal.error_call)
        self.assertIsNone(wrapper.test_goal("CHECK( init(main()), LTL(F end) )\n"))

    def test_branch_coverage_command(self):
        out = dry_run(COVERAGE_BRANCHES, "-o", "branch-function")
        for flag in ("--force-malloc-success", "--floatbv", "--generate-testcase",
                     "--branch-function-coverage", "--max-inductive-step 8"):
            self.assertIn(flag, out)
        self.assertNotIn("--witness-output", out)

    def test_error_call_command(self):
        out = dry_run(COVERAGE_ERROR_CALL, "-o", "branch-function")
        for flag in ("--generate-testcase", "--enable-unreachability-intrinsic", "--error-label ERROR"):
            self.assertIn(flag, out)
        self.assertNotIn("--branch-function-coverage", out)

    def test_svcomp_command_unchanged(self):
        out = dry_run(UNREACH_CALL)
        self.assertNotIn("--generate-testcase", out)
        self.assertNotIn("--force-malloc-success", out)


class ReachValidationUnwindTest(unittest.TestCase):
    """Correctness validation unrolls loops only when the witness has no
    invariants, such as ESBMC's own; violation validation never does."""

    UNROLL = "--goto-unwind --unlimited-goto-unwind"
    ESBMC = '- entry_type: "invariant_set"\n  content:\n    []\n'
    BLOCK = ('- entry_type: "invariant_set"\n  content:\n'
             '    - invariant:\n        type: "loop_invariant"\n        value: "i <= 5"\n')
    FLOW = '[{entry_type: invariant_set, content: [{invariant: {type: loop_invariant, value: i <= 5}}]}]\n'
    JSON = '[{"entry_type": "invariant_set", "content": [{"invariant": {"type": "loop_invariant"}}]}]\n'

    def command(self, witness, mode):
        with tempfile.NamedTemporaryFile("w", suffix=".yml") as w:
            w.write(witness)
            w.flush()
            return dry_run(UNREACH_CALL, "--witness", w.name, "--validate-" + mode + "-witness")

    def test_verification_unrolls(self):
        self.assertIn(self.UNROLL, dry_run(UNREACH_CALL))

    def test_invariant_free_correctness_witness_unrolls(self):
        self.assertIn(self.UNROLL, self.command(self.ESBMC, "correctness"))

    def test_invariants_keep_loops_in_every_yaml_style(self):
        for witness in (self.BLOCK, self.FLOW, self.JSON):
            self.assertNotIn("--goto-unwind", self.command(witness, "correctness"), witness)

    def test_violation_validation_keeps_loops(self):
        self.assertNotIn("--goto-unwind", self.command(self.ESBMC, "violation"))


class PthreadDetectionTest(unittest.TestCase):
    """Only a call to pthread_create makes a task concurrent (#9971)."""

    def detects(self, source):
        with tempfile.NamedTemporaryFile("w", suffix=".i") as f:
            f.write(source)
            f.flush()
            return wrapper.check_if_benchmark_contains_pthread(f.name)

    def test_a_call_is_concurrent(self):
        self.assertTrue(self.detects(
            "extern int pthread_create (pthread_t *__restrict __newthread,\n"
            "  status = pthread_create (&t, ((void *)0), f, ((void *)0));\n"))
        self.assertTrue(self.detects("  pthread_create(&t1, 0, P0, 0);\n"))

    def test_a_declaration_alone_is_not(self):
        self.assertFalse(self.detects(
            "extern int ( __attribute__((__nonnull__(1,3))) pthread_create)"
            "(pthread_t * __restrict __newthread ,\n"))
        self.assertFalse(self.detects(
            "extern int pthread_create (pthread_t *__restrict __newthread,\n"))


class ThreadedTaskCommandLineTest(unittest.TestCase):
    """ESBMC, not the wrapper, configures a program that starts a thread
    (#10189)."""

    def test_a_pthread_create_call_keeps_the_common_command_line(self):
        with tempfile.NamedTemporaryFile("w", suffix=".i") as f:
            f.write("int main() { pthread_create(&t, 0, f, 0); }\n")
            f.flush()
            cmd = wrapper.get_command_line(
                "kinduction", wrapper.Property.reach, 64, f.name, "", False)
        self.assertIn("--k-induction-parallel ", cmd)
        self.assertIn("--interval-analysis", cmd)
        self.assertNotIn("--context-bound", cmd)


class TimeoutTest(unittest.TestCase):
    """A timeout kills the processes --k-induction-parallel forks, which
    otherwise hold the output pipes open until they finish."""

    def test_a_timeout_kills_the_forked_children(self):
        started = time.monotonic()
        with mock.patch.object(wrapper, "args", argparse.Namespace(dry_run=False)):
            out = wrapper.do_exec("sh -c 'sleep 60 & sleep 60'", timeout=1)
        self.assertEqual(out, b"ERROR: Timed out\n")
        self.assertLess(time.monotonic() - started, 30)


class OverflowFallbackTest(unittest.TestCase):
    """A no-overflow task runs k-induction first, then incremental BMC."""

    OVERFLOW = """
Violated property:
  file task.c line 1 column 20 function main
  arithmetic overflow on add

VERIFICATION FAILED
"""
    SUCCESSFUL = "VERIFICATION SUCCESSFUL\n"
    UNKNOWN = "VERIFICATION UNKNOWN\n"

    def run_wrapper(self, kind_output, incr_output):
        """Run the wrapper on a stand-in esbmc that answers by strategy."""
        with tempfile.TemporaryDirectory() as d:
            for name, text in (("kind.out", kind_output), ("incr.out", incr_output),
                               ("no-overflow.prp", "CHECK( init(main()), LTL(G ! overflow) )\n"),
                               ("task.c", "int main() { return 0; }\n")):
                with open(os.path.join(d, name), "w", encoding="utf-8") as f:
                    f.write(text)
            esbmc = os.path.join(d, "esbmc")
            with open(esbmc, "w", encoding="utf-8") as f:
                f.write('#!/bin/sh\necho "$@" >> calls.log\n'
                        'case "$*" in *--k-induction*) cat kind.out ;; *) cat incr.out ;; esac\n')
            os.chmod(esbmc, 0o755)
            script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "esbmc-wrapper.py")
            out = subprocess.run(
                [sys.executable, script, "-p", "no-overflow.prp", "-s", "kinduction", "task.c"],
                capture_output=True, text=True, check=True, cwd=d).stdout
            with open(os.path.join(d, "calls.log"), encoding="utf-8") as f:
                calls = f.read().splitlines()
        return out.splitlines()[-1], calls

    def test_a_proof_by_k_induction_is_kept(self):
        verdict, calls = self.run_wrapper(self.SUCCESSFUL, self.UNKNOWN)
        self.assertEqual(verdict, "TRUE")
        self.assertEqual(len(calls), 1)
        self.assertIn("--memlimit " + wrapper.KIND_OVERFLOW_MEMLIMIT, calls[0])
        self.assertIn(wrapper.KIND_OVERFLOW_INTERVALS, calls[0] + " ")

    def test_an_overflow_found_by_k_induction_is_kept(self):
        verdict, calls = self.run_wrapper(self.OVERFLOW, self.SUCCESSFUL)
        self.assertEqual(verdict, "FALSE_OVERFLOW")
        self.assertEqual(len(calls), 1)

    def test_a_timed_out_k_induction_falls_back_to_incremental_bmc(self):
        verdict, calls = self.run_wrapper("ERROR: Timed out\n", self.SUCCESSFUL)
        self.assertEqual(verdict, "TRUE")
        self.assertEqual(len(calls), 2)

    def test_an_inconclusive_k_induction_falls_back_to_incremental_bmc(self):
        verdict, calls = self.run_wrapper(self.UNKNOWN, self.OVERFLOW)
        self.assertEqual(verdict, "FALSE_OVERFLOW")
        self.assertEqual(len(calls), 2)
        self.assertIn("--incremental-bmc", calls[1])
        self.assertNotIn("--k-induction", calls[1])
        self.assertNotIn("--memlimit", calls[1])
        self.assertNotIn("--interval-analysis", calls[1])


if __name__ == "__main__":
    unittest.main()
