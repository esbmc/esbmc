#!/usr/bin/env python3
"""Self-test for ab_interleave.py. Run: python3 scripts/perf/test_ab_interleave.py

The regexes are the only thing between the bisect and a silently unparsed run,
so they are checked against real ESBMC output rather than a hand-written line.
"""

import unittest

import ab_interleave

# Verbatim from: esbmc scripts/perf/oracles/loop10k.c --unwind 10000
#                      --overflow-check --quiet   (ESBMC 8.4.0)
ESBMC_OUTPUT = """\
ESBMC version 8.4.0 64-bit x86_64 linux
Target: 64-bit little-endian x86_64-unknown-linux with esbmclibc
Parsing scripts/perf/oracles/loop10k.c
Converting
GOTO program creation time: 0.311s
GOTO program processing time: 0.000s
Starting Bounded Model Checking
Symex completed in: 0.800s (120003 assignments)
Caching time: 0.197s (removed 1831 assertions)
Slicing time: 0.703s (removed 40010 assignments)
Generated 79992 VCC(s), 78161 remaining after simplification (78162 assignments)
No solver specified; defaulting to bitwuzla
Encoding remaining VCC(s) using bit-vector/floating-point arithmetic
Encoding to solver time: 4.261s
Solving with solver Bitwuzla 0.9.0
Runtime decision procedure: 2.100s
BMC program time: 8.090s

VERIFICATION SUCCESSFUL
"""


class TestParse(unittest.TestCase):
    def test_every_phase_and_count_parses(self):
        sample = ab_interleave.parse(ESBMC_OUTPUT)
        self.assertEqual(sample["goto"], 0.311)
        self.assertEqual(sample["goto-proc"], 0.0)
        self.assertEqual(sample["symex"], 0.800)
        self.assertEqual(sample["caching"], 0.197)
        self.assertEqual(sample["slicing"], 0.703)
        self.assertEqual(sample["encoding"], 4.261)
        self.assertEqual(sample["solve"], 2.100)
        self.assertEqual(sample["bmc"], 8.090)
        self.assertEqual(sample["assignments"], 120003)
        self.assertEqual(sample["vccs"], 79992)
        self.assertEqual(sample["remaining"], 78161)

    def test_truncated_output_yields_none_not_zero(self):
        sample = ab_interleave.parse("GOTO program creation time: 0.311s\n")
        self.assertEqual(sample["goto"], 0.311)
        self.assertIsNone(sample["symex"])
        self.assertIsNone(sample["vccs"])


class TestRatios(unittest.TestCase):
    def test_ratio_is_per_pair(self):
        # Drift doubles both arms in the second pair: a paired ratio sees
        # 1.10 twice, a ratio of medians would not.
        a_samples = [{"wall": 10.0}, {"wall": 20.0}]
        b_samples = [{"wall": 11.0}, {"wall": 22.0}]
        self.assertEqual(ab_interleave.ratios(a_samples, b_samples, "wall"), [1.1, 1.1])

    def test_unparsed_side_drops_the_pair(self):
        a_samples = [{"wall": 10.0}, {"wall": None}]
        b_samples = [{"wall": 11.0}, {"wall": 22.0}]
        self.assertEqual(ab_interleave.ratios(a_samples, b_samples, "wall"), [1.1])


if __name__ == "__main__":
    unittest.main()
