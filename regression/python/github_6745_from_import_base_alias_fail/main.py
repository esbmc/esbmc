from unittest import TestCase as TC


class T(TC):

    def test_broken(self) -> None:
        self.assertEqual(1, 2)


t = T()
t.test_broken()
