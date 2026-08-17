#!/usr/bin/env python3
"""Self-test. Run: python3 scripts/competitions/svcomp/test_marginal_timeouts.py"""

import unittest
from xml.etree import ElementTree

import marginal_timeouts

# Shaped like BenchExec 3.x output: one <run> per task, times as '12.3s'.
RESULT_XML = """<?xml version="1.0"?>
<result benchmarkname="esbmc" cpuTimelimit="100.0s" walltimelimit="120.0s">
  <run name="ok_fast.c" files="[ok_fast.c]">
    <column title="status" value="true"/>
    <column title="cputime" value="3.140000000s"/>
  </run>
  <run name="ok_marginal.c" files="[ok_marginal.c]">
    <column title="status" value="true"/>
    <column title="cputime" value="99.100000000s"/>
  </run>
  <run name="ok_just_inside.c" files="[ok_just_inside.c]">
    <column title="status" value="false(unreach-call)"/>
    <column title="cputime" value="95.000000000s"/>
  </run>
  <run name="lost.c" files="[lost.c]">
    <column title="status" value="TIMEOUT"/>
    <column title="cputime" value="100.200000000s"/>
  </run>
</result>
"""


class TestClassify(unittest.TestCase):
    def setUp(self):
        self.result = ElementTree.fromstring(RESULT_XML)

    def test_marginal_band_is_inclusive_at_the_threshold(self):
        wins, losses, safe = marginal_timeouts.classify(self.result, 100.0, 5.0)
        self.assertEqual([name for name, _ in wins], ["ok_marginal.c", "ok_just_inside.c"])
        self.assertEqual([name for name, _ in losses], ["lost.c"])
        self.assertEqual(safe, 1)

    def test_a_narrower_margin_keeps_fewer_tasks(self):
        wins, _, safe = marginal_timeouts.classify(self.result, 100.0, 1.0)
        self.assertEqual([name for name, _ in wins], ["ok_marginal.c"])
        self.assertEqual(safe, 2)

    def test_limit_comes_from_the_cpu_budget(self):
        self.assertEqual(marginal_timeouts.limit_of(self.result, None), 100.0)
        self.assertEqual(marginal_timeouts.limit_of(self.result, 30.0), 30.0)


class TestParseSeconds(unittest.TestCase):
    def test_strips_the_unit(self):
        self.assertEqual(marginal_timeouts.parse_seconds("12.5s"), 12.5)

    def test_missing_or_unparseable_is_none(self):
        self.assertIsNone(marginal_timeouts.parse_seconds(None))
        self.assertIsNone(marginal_timeouts.parse_seconds(""))
        self.assertIsNone(marginal_timeouts.parse_seconds("-"))


if __name__ == "__main__":
    unittest.main()
