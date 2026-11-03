import unittest

from helper import Mixin


class T(Mixin, unittest.TestCase):

    def test_a(self) -> None:
        self.assertEqual(1, 1)


unittest.main()
