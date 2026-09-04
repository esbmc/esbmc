import unittest as ut


class T(ut.TestCase):

    def test_broken(self) -> None:
        self.assertLess(3, 2)


if __name__ == "__main__":
    ut.main()
