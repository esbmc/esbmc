# No __len__ anywhere in the ancestry, so the refusal still applies after the
# base-class walk.

class Base:
    def __init__(self) -> None:
        self.i: int = 0


class Derived(Base):
    pass


def main():
    d = Derived()
    assert len(d) == 0


main()
