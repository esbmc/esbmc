import unittest

calls = [0]


def bump() -> int:
    calls[0] += 1
    return calls[0]


class T(unittest.TestCase):

    def test_order(self) -> None:
        self.assertLess(b=bump(), a=bump())


unittest.main()
