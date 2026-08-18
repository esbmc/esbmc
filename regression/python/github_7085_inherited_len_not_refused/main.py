# A base class supplies __len__, so the no-__len__ refusal must not fire here:
# the check walks the ancestry before refusing. What this pins is the failure
# *reason* -- an inherited __len__ is still not dispatched (pre-existing), so
# len() falls through to strlen and mismatches. Revisit this test when
# base-class dispatch is fixed -- it should then verify successfully.

class Base:
    def __init__(self) -> None:
        self.n: int = 4

    def __len__(self) -> int:
        return self.n


class Derived(Base):
    pass


def main():
    d = Derived()
    assert len(d) == 4


main()
