# A driver calls the test method, so the assertion is really checked rather
# than merely converted. 1 + 1 is not 3.
import unittest


class T(unittest.TestCase):
    def test_add(self) -> None:
        self.assertEqual(1 + 1, 3)


t = T()
t.test_add()
