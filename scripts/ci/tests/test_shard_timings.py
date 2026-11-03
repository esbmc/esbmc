# ci_fixture puts scripts/ci on sys.path, so it comes first.
import ci_fixture  # pylint: disable=wrong-import-order,unused-import
import unittest

import shard_timings

LOG = """\
2026-11-02T16:54:11Z     1/5 Test     #2: csmith-kind ..........   Passed   30.02 sec
2026-11-02T16:54:12Z     2/5 Test     #1: regression/python/a ..   Passed    4.00 sec
2026-11-02T16:54:13Z     3/5 Test     #3: regression/numpy/b ...   Passed    6.00 sec
2026-11-02T16:54:14Z     4/5 Test     #4: regression/esbmc/c ...***Failed    1.50 sec
2026-11-02T16:54:15Z     5/7 Test     #5: regression/python-intensive/d   Passed   9.00 sec
2026-11-02T16:54:16Z     6/7 Test     #6:  starts with a space ...   Passed    0.01 sec
2026-11-02T16:54:17Z     7/7 Test     #7: regression/esbmc/skip .....***Skipped   0.07 sec
            Start  6: regression/esbmc/e
"""


class ShardTimingsTest(unittest.TestCase):

    def test_durations_are_read_in_index_order_whatever_their_status(self):
        tests = shard_timings.read_durations(LOG.splitlines())
        self.assertEqual([t[0] for t in tests], [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(tests[3], (4, "regression/esbmc/c", 1.5))
        self.assertEqual(tests[5], (6, " starts with a space", 0.01))
        self.assertEqual(tests[6], (7, "regression/esbmc/skip", 0.07))

    def test_families_follow_the_selection_labels(self):
        self.assertEqual(shard_timings.family("csmith-kind"), "producer")
        self.assertEqual(shard_timings.family("regression/numpy/b"), "python")
        self.assertEqual(shard_timings.family("regression/esbmc/c"), "native")
        self.assertEqual(shard_timings.family("regression/python-intensive/d"), "excluded")

    def test_slots_take_the_next_test_when_they_free_up(self):
        self.assertEqual(shard_timings.makespan([4, 1, 1, 1], 2), 4)
        self.assertEqual(shard_timings.makespan([1, 1, 1], 1), 3)

    def test_shards_stripe_their_family(self):
        tests = shard_timings.read_durations(LOG.splitlines())
        seconds = shard_timings.shard_seconds(tests, (("python", 2), ("native", 1)), 1)
        self.assertEqual(seconds, {"python-1": 4.0, "python-2": 6.0, "native-1": 1.57})


if __name__ == "__main__":
    unittest.main()
