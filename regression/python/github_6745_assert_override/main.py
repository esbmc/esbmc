import unittest


class T(unittest.TestCase):

    def assertEqual(self, a: int, b: int) -> None:
        assert a != b

    def test_a(self) -> None:
        self.assertEqual(1, 2)


unittest.main()
