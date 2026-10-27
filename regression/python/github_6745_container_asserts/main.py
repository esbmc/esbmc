import unittest

calls = [0]


def bump() -> list[int]:
    calls[0] += 1
    return [calls[0]]


class T(unittest.TestCase):

    def setUp(self) -> None:
        self.items = [1, 2]
        self.n = 3

    def test_containers(self) -> None:
        self.assertEqual(self.items, [1, 2])
        self.assertEqual({1: 2}, {1: 2})
        self.assertEqual((1, 2), (1, 2))
        self.assertEqual({1, 2}, {2, 1})
        self.assertNotEqual(self.items, [2, 1])
        self.assertIn(2, self.items)

    def test_order(self) -> None:
        self.assertGreater(self.n, 2)
        self.assertLess(self.n, 4, "msg")
        for x in self.items:
            self.assertGreaterEqual(x, 1)

    def test_typed(self) -> None:
        self.assertIsInstance(self.items, list)
        self.assertNotIsInstance(self.items, tuple)
        self.assertListEqual(list1=self.items, list2=[1, 2], msg=f"n={self.n}")
        self.assertTupleEqual((1, 2), (1, 2))
        self.assertDictEqual({1: 2}, {1: 2})
        self.assertMultiLineEqual("a", "a")
        self.assertListEqual(bump(), [1])
        self.assertEqual(calls[0], 1)


unittest.main()
