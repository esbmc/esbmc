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
        x_equal:bool = (self.x == other.x)
        y_equal:bool = (self.y == other.y)
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
    print(i1.x)
    print(i1.y)
    print(i2.x)
    print(i2.y)

    # Data coordinates
    d1 = CoordinateData(1, 2)
    d2 = CoordinateData(1, 2)
    print("Equal coordinates?")
    print(d1.equals(d2))

    # Frost coordinates
    f1 = CoordinateFrost(1, 2)
    f2 = CoordinateFrost(1, 2)
    print("Equal frost coordinates?")
    print(f1.equals(f2))
    print("Hash values:")
    print(f1.get_hash())
    print(f2.get_hash())


if __name__ == "__main__":
    demonstrate_classes(None)


