import unittest


class T(unittest.TestCase):

    def test_type(self) -> None:
        self.assertListEqual([1, 2], (1, 2))


unittest.main()
