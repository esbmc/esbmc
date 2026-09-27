#!/usr/bin/env python3
"""Self-test. Run: python3 scripts/irep2/test_bars.py

Each case is a shape that made the script over-count a site the migration had
already converted, which is the failure mode that matters: an unfalsifiable bar.
"""

import importlib.util
import os
import unittest

_spec = importlib.util.spec_from_file_location("bars",
                                               os.path.join(os.path.dirname(__file__), "bars.py"))
bars = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bars)


def count(source):
    """(raw, refined) B-2 writes in one translation unit's text."""
    return bars.count_b2("t.cpp", bars.strip_noise(source))


def write_lines(source):
    """The source lines --list would report for each write."""
    clean = bars.strip_noise(source)
    return [
        bars.line_of(lines, m.start()) for lines, line in bars.statements(clean)
        for m in bars.WRITE.finditer(line)
    ]


class TestRefinement(unittest.TestCase):

    def test_legacy_write_counts(self):
        self.assertEqual(count("typet t; sym.set_type(t);"), (1, 1))

    def test_migrate_call_does_not_count(self):
        self.assertEqual(count("sym.set_type(migrate_type(t));"), (1, 0))

    def test_constructor_does_not_count(self):
        self.assertEqual(count("sym.set_type(array_type2tc(s, n, false));"), (1, 0))

    def test_declared_name_does_not_count(self):
        self.assertEqual(count("type2tc t2 = f(); sym.set_type(t2);"), (1, 0))

    def test_element_of_a_declared_vector_does_not_count(self):
        self.assertEqual(
            count("const std::vector<type2tc> &args = c.arguments;\n"
                  "sym.set_type(args[0]);"), (1, 0))

    def test_field_through_a_container_does_not_count(self):
        self.assertEqual(
            count("const expr2tc &callee = *call->callee;\n"
                  "sym.set_type(callee->type);"), (1, 0))

    def test_field_of_a_node_reference_does_not_count(self):
        self.assertEqual(
            count("const code_type2t &ct = to_code_type(t);\n"
                  "sym.set_type(ct.return_type);"), (1, 0))

    def test_an_irep2_only_helper_does_not_count(self):
        self.assertEqual(count("sym.set_value(gen_false_expr());"), (1, 0))

    def test_a_helper_with_a_legacy_namesake_still_counts(self):
        self.assertEqual(count("sym.set_value(gen_zero(t, true));"), (1, 1))

    def test_a_helper_with_a_legacy_namesake_over_an_irep2_type_does_not_count(self):
        self.assertEqual(
            count("void f(const type2tc &t, int n) { sym.set_value(gen_zero(t, true)); }"), (1, 0))

    def test_a_constant_bit_string_does_not_count(self):
        self.assertEqual(
            count("constant_exprt c(size_type());\n"
                  "c.set_value(integer2binary(h, 64));"), (1, 0))

    def test_a_symbol_value_still_counts(self):
        self.assertEqual(count("constant_exprt c(t); sym.set_value(c);"), (1, 1))

    def test_write_split_over_lines_sees_its_argument(self):
        self.assertEqual(count("sym.set_type(\n  array_type2tc(s, n, false));"), (1, 0))


class TestNoise(unittest.TestCase):

    def test_a_write_in_a_comment_does_not_count(self):
        self.assertEqual(count("// sym.set_type(t);\ntypet t;"), (0, 0))

    def test_block_comment_keeps_the_line_numbering(self):
        clean = bars.strip_noise("a\n/* two\n   lines */\nb\n")
        self.assertEqual(len(clean.split("\n")), 5)


class TestReportedLine(unittest.TestCase):

    def test_line_is_the_write_after_a_block_comment(self):
        self.assertEqual(write_lines("/* c\n   c */\nvoid f()\n{\n  sym.set_type(t);\n}\n"), [5])

    def test_line_is_the_write_not_the_brace_less_if(self):
        self.assertEqual(write_lines("if (ok)\n  sym.set_value(v);\n"), [2])

    def test_line_is_the_first_line_of_a_split_write(self):
        self.assertEqual(write_lines("sym.set_type(\n  t);\n"), [1])

    def test_a_parenthesis_in_a_character_literal_does_not_shift_it(self):
        self.assertEqual(write_lines("if (c == '(')\n  n++;\nsym.set_type(t);\n"), [3])


if __name__ == "__main__":
    unittest.main()
