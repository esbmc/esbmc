# example_3_dataclasses.py


class CoordinateInit:

    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y


class CoordinateData:

    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def equals(self, other: "CoordinateData | None") -> bool:
        if other is None:
            return False
        x_equal: bool = (self.x == other.x)
        y_equal: bool = (self.y == other.y)
        return x_equal and y_equal


class CoordinateFrost:

    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def equals(self, other: "CoordinateFrost | None") -> bool:
        if other is None:
            return False
        x_equal = (self.x == other.x)
        y_equal = (self.y == other.y)
        return x_equal and y_equal

    def get_hash(self) -> int:
        return self.x * 31 + self.y


def demonstrate_classes(dummy: object) -> None:
    # Regular coordinates
    i1 = CoordinateInit(1, 2)
    i2 = CoordinateInit(1, 2)
    print("Regular coordinates:")
    assert i1.x == 1
    assert i1.y == 2
    assert i2.x == 1
    assert i2.y == 2

    # Data coordinates
    d1 = CoordinateData(1, 2)
    d2 = CoordinateData(1, 2)
    print("Equal coordinates?")
    assert d1.equals(d2) == True

    # Frost coordinates
    f1 = CoordinateFrost(1, 2)
    f2 = CoordinateFrost(1, 2)
    print("Equal frost coordinates?")
    assert f1.equals(f2) == True
    print("Hash values:")
    assert f1.get_hash() == 33
    assert f2.get_hash() == 33


if __name__ == "__main__":
    demonstrate_classes(None)
