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

    def test_a_relative_benchmark_path_is_made_absolute(self):
        os.environ.pop("PYTHONPATH", None)
        wrapper.add_python_module_path("boto3_all_not_none.py")
        for entry in os.environ["PYTHONPATH"].split(os.pathsep):
            self.assertTrue(os.path.isabs(entry), entry)


if __name__ == "__main__":
    unittest.main()
