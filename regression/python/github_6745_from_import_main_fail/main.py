from unittest import TestCase, main


class T(TestCase):

    def test_broken(self) -> None:
        self.assertTrue(1 > 2)


main()
