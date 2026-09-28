import unittest


class T(unittest.TestCase):

    def setUp(self) -> None:
        self.base = 2

    def test_add(self) -> None:
        self.assertEqual(self.base + 1, 3)

    def test_compare(self) -> None:
        self.assertTrue(self.base > 1)


unittest.main()
