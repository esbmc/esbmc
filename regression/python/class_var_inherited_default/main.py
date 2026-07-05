class Base:
    x: int = 3


class Derived(Base):
    y: int = 7


def read_y(d: Derived) -> int:
    return d.y


def main() -> None:
    d = Derived()
    # Inherited/derived class-variable defaults read through a Class* parameter.
    assert read_y(d) == 7
    assert d.x == 3


main()
