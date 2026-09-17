import unittest


class T(unittest.TestCase):

    def test_add(self) -> None:
        self.assertEqual(1 + 1, 3)


unittest.main()
