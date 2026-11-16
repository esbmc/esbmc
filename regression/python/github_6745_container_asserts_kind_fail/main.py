import unittest


class T(unittest.TestCase):

    def test_kind(self) -> None:
        self.assertEqual([1, 2], (1, 2))


unittest.main()
