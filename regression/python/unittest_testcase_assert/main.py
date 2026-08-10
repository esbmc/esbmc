# Every CPython regression test is a unittest.TestCase subclass, so the module
# has an operational model. The assertX methods lower to plain asserts; there
# is no runner, so a method runs only when something calls it.
import unittest


class T(unittest.TestCase):
    def test_add(self) -> None:
        self.assertEqual(1 + 1, 2)
        self.assertTrue(2 > 1)
        self.assertIn(2, [1, 2, 3])
        self.assertIsNone(None)
