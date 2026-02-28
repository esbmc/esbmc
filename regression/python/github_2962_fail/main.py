# Test for Issue #2962: Method is properly executed and can fail
# Verifies that _init_members() called from __init__ actually runs
class Foo:

    def __init__(self) -> None:
        self._init_members()

    def _init_members(self) -> None:
        assert False  # Should fail here, proving method was executed


f = Foo()
