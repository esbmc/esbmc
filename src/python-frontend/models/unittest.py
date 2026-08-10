# Operational model for the unittest module.
# pylint: disable=unused-argument
# Every CPython regression test is a unittest.TestCase subclass, so without
# this module none of them convert at all (#6745). The model covers the
# assertion vocabulary those tests actually use and maps each assertX onto a
# plain `assert`, which the frontend already turns into a claim.
#
# NOT modelled: test discovery and the runner. main() is a no-op, so a test
# method runs only when something calls it -- the model makes a TestCase file
# convertible, it does not execute the suite. setUp/tearDown are no-ops here so
# a subclass can override them; nothing calls them automatically either.
# msg arguments are accepted and ignored: the claim carries the location.
from typing import Any


class TestCase:
    """unittest.TestCase -- assertions lowered to ESBMC claims."""

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def assertEqual(self, first: Any, second: Any, msg: Any = None) -> None:
        assert first == second

    def assertNotEqual(self, first: Any, second: Any, msg: Any = None) -> None:
        assert first != second

    def assertTrue(self, expr: Any, msg: Any = None) -> None:
        assert expr

    def assertFalse(self, expr: Any, msg: Any = None) -> None:
        assert not expr

    def assertIs(self, first: Any, second: Any, msg: Any = None) -> None:
        assert first is second

    def assertIsNot(self, first: Any, second: Any, msg: Any = None) -> None:
        assert first is not second

    def assertIsNone(self, obj: Any, msg: Any = None) -> None:
        assert obj is None

    def assertIsNotNone(self, obj: Any, msg: Any = None) -> None:
        assert obj is not None

    def assertIn(self, member: Any, container: Any, msg: Any = None) -> None:
        assert member in container

    def assertNotIn(self, member: Any, container: Any, msg: Any = None) -> None:
        assert member not in container

    def assertLess(self, a: Any, b: Any, msg: Any = None) -> None:
        assert a < b

    def assertLessEqual(self, a: Any, b: Any, msg: Any = None) -> None:
        assert a <= b

    def assertGreater(self, a: Any, b: Any, msg: Any = None) -> None:
        assert a > b

    def assertGreaterEqual(self, a: Any, b: Any, msg: Any = None) -> None:
        assert a >= b

    def fail(self, msg: Any = None) -> None:
        assert False


def main(*args: Any, **kwargs: Any) -> None:
    """unittest.main() -- no runner, so nothing to do."""
