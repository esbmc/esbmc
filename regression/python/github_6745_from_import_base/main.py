from unittest import TestCase


class T(TestCase):

    def test_ok(self) -> None:
        self.assertEqual(2 + 2, 4)


t = T()
t.test_ok()
