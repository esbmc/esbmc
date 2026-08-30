import unittest


class T(unittest.TestCase):

    @unittest.skip("would fail if the runner did not skip it")
    def test_skipped(self) -> None:
        self.fail()

    def helper(self) -> None:
        self.fail()

    def test_run(self) -> None:
        self.assertEqual(1 + 1, 2)


unittest.main()
