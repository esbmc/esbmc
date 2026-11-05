#!/usr/bin/env python3
"""Self-test. Run: python3 scripts/competitions/svcomp/test_esbmc_wrapper.py"""

import importlib.util
import os
import subprocess
import sys
import tempfile
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


class IntegerEncodingConfirmationTest(unittest.TestCase):
    """A Python violation stays FALSE only if a --ir re-run repeats it in place."""

    PASSED = "** 0 of 1 properties failed, 1 passed\n\nVERIFICATION SUCCESSFUL\n"
    TIMED_OUT = b"ERROR: Timed out\n"
    NO_LINE = PY_ASSERTION.replace(" line 5 column 4", "")

    def verify(self, prop, first, rerun, benchmark="main.py", elapsed=0):
        outputs = iter([first, rerun])
        calls = []

        def fake_run(cmd_line, timeout=None):
            calls.append((cmd_line, timeout))
            out = next(outputs)
            return out if isinstance(out, bytes) else out.encode()

        with mock.patch.object(wrapper, "run", fake_run), \
                mock.patch.object(wrapper.time, "monotonic", side_effect=[0, elapsed]):
            result = wrapper.verify("kinduction", prop, 64, benchmark, False, "", False, None, None)
        return wrapper.get_result_string(result), calls

    def test_rerun_at_the_same_line_keeps_false(self):
        verdict, calls = self.verify(wrapper.Property.py_assertion, PY_ASSERTION, PY_ASSERTION)
        self.assertEqual(verdict, "FALSE")
        self.assertTrue(calls[1][0].endswith("--ir --z3 --multi-property "))
        self.assertEqual(calls[1][1], wrapper.IR_RERUN_TIMEOUT)
        self.assertIsNone(calls[0][1])

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

    def test_rerun_that_passes_is_unknown(self):
        verdict, _ = self.verify(wrapper.Property.py_assertion, PY_ASSERTION, self.PASSED)
        self.assertEqual(verdict, "Unknown")

    def test_rerun_failing_at_another_line_is_unknown(self):
        other = PY_ASSERTION.replace("line 5", "line 9")
        verdict, _ = self.verify(wrapper.Property.py_assertion, PY_ASSERTION, other)
        self.assertEqual(verdict, "Unknown")

    def test_rerun_failing_both_lines_keeps_the_first_runs_site(self):
        both = PY_ASSERTION.replace("line 5 column 4", "line 9 column 4").replace(
            "  FAILED       [main.assertion.1]  line 5  assertion x != 3\n",
            "  FAILED       [main.assertion.1]  line 5  assertion x != 3\n"
            "  FAILED       [main.assertion.2]  line 9  assertion y != 4\n")
        verdict, _ = self.verify(wrapper.Property.py_assertion, PY_ASSERTION, both)
        self.assertEqual(verdict, "FALSE")

    def test_rerun_timeout_is_unknown(self):
        verdict, _ = self.verify(wrapper.Property.py_assertion, PY_ASSERTION, self.TIMED_OUT)
        self.assertEqual(verdict, "Unknown")

    def test_rerun_failing_another_property_is_unknown(self):
        same_site_valueerror = PY_VALUEERROR.replace("p_raise_value.py", "p_assert2.py").replace("line 6 column 8", "line 5 column 4")
        verdict, _ = self.verify(wrapper.Property.py_assertion, PY_ASSERTION, same_site_valueerror)
        self.assertEqual(verdict, "Unknown")

    def test_no_line_on_either_side_matches_on_the_file(self):
        verdict, _ = self.verify(wrapper.Property.py_assertion, self.NO_LINE, self.NO_LINE)
        self.assertEqual(verdict, "FALSE")

    def test_a_line_on_one_side_only_is_unknown(self):
        verdict, _ = self.verify(wrapper.Property.py_assertion, self.NO_LINE, PY_ASSERTION)
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
            verdict, calls = self.verify(wrapper.Property.reach, REACH_WITH_UNCHECKED_UNWINDING, self.PASSED, c_file.name)
        self.assertEqual(verdict, "FALSE_REACH")
        self.assertEqual(len(calls), 1)


class PythonCommandLineTest(unittest.TestCase):
    """Each run reports only its property's exception family."""

    def command(self, prop):
        return wrapper.get_command_line("kinduction", prop, None, "main.py", False, "", False)

    def test_each_property_checks_its_own_family(self):
        cases = (
            (wrapper.Property.py_assertion, "AssertionError"),
            (wrapper.Property.py_arithmetic, "ZeroDivisionError,FloatingPointError"),
            (wrapper.Property.py_datalookup, "KeyError,IndexError"),
            (wrapper.Property.py_dyntyping, "TypeError,AttributeError"),
        )
        for prop, family in cases:
            self.assertIn("--python-check-exceptions " + family + " ", self.command(prop))


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

    def dry_run(self, prp_content, *options):
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

    def test_branch_coverage_command(self):
        out = self.dry_run(COVERAGE_BRANCHES, "-o", "branch-function")
        for flag in ("--force-malloc-success", "--floatbv", "--generate-testcase",
                     "--branch-function-coverage", "--max-inductive-step 8"):
            self.assertIn(flag, out)
        self.assertNotIn("--witness-output", out)

    def test_error_call_command(self):
        out = self.dry_run(COVERAGE_ERROR_CALL, "-o", "branch-function")
        for flag in ("--generate-testcase", "--enable-unreachability-intrinsic", "--error-label ERROR"):
            self.assertIn(flag, out)
        self.assertNotIn("--branch-function-coverage", out)

    def test_svcomp_command_unchanged(self):
        out = self.dry_run(UNREACH_CALL)
        self.assertNotIn("--generate-testcase", out)
        self.assertNotIn("--force-malloc-success", out)


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

    def test_an_overflow_found_by_k_induction_is_kept(self):
        verdict, calls = self.run_wrapper(self.OVERFLOW, self.SUCCESSFUL)
        self.assertEqual(verdict, "FALSE_OVERFLOW")
        self.assertEqual(len(calls), 1)

    def test_an_inconclusive_k_induction_falls_back_to_incremental_bmc(self):
        verdict, calls = self.run_wrapper(self.UNKNOWN, self.OVERFLOW)
        self.assertEqual(verdict, "FALSE_OVERFLOW")
        self.assertEqual(len(calls), 2)
        self.assertIn("--incremental-bmc", calls[1])
        self.assertNotIn("--k-induction", calls[1])
        self.assertNotIn("--memlimit", calls[1])


if __name__ == "__main__":
    unittest.main()
