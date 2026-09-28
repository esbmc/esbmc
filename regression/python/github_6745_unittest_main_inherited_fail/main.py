import unittest


class Base(unittest.TestCase):

    def setUp(self) -> None:
        self.value = 1

    def test_value(self) -> None:
        self.assertEqual(self.value, 1)


class Derived(Base):

    def setUp(self) -> None:
        self.value = 2


unittest.main()
