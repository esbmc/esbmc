from typing import Optional


class Base:
    def __init__(self) -> None:
        self.n: int = 1


class T(Base):
    def __init__(self, k: int) -> None:
        super().__init__()
        self.k: int = k


def main() -> None:
    t = T(nondet_int())
    d = 41
    assert getattr(t, "n") + getattr(t, "total", d) == 42
    missing: Optional[int] = getattr(t, "count", None)
    assert missing is None


main()
