# github.com/esbmc/esbmc/issues/6242 (review finding #2)
# Characterization test for a missing capability: `self.attr = self.method()`
# is typed by the enclosing class's return type and ignores subclass overrides
# (no virtual dispatch), so this valid polymorphic program is reported as a
# false VERIFICATION FAILED. CPython: Derived().go() == 2 holds, so the correct
# result is SUCCESSFUL. Pinned as CORE-FAILED (not KNOWNBUG/FUTURE) so a crash
# regression is caught and so the test breaks loudly if virtual dispatch lands.
class P1:
    def val(self) -> int:
        return 1


class P2:
    def val(self) -> int:
        return 2


class Base:
    def make(self):
        return P1()

    def __init__(self):
        self.o = self.make()

    def go(self) -> int:
        return self.o.val()


class Derived(Base):
    def make(self):
        return P2()


assert Derived().go() == 2
