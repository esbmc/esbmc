import unittest


class T(unittest.TestCase):

    def setUp(self) -> None:
        self.items = [1, 2]

    def test_order(self) -> None:
        self.assertEqual(self.items, [2, 1])


unittest.main()
