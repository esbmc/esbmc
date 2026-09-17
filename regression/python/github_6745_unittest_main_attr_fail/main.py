import unittest


class T(unittest.TestCase):
    expected = 3

    def test_value(self) -> None:
        self.assertEqual(1 + 1, self.expected)


unittest.main()
