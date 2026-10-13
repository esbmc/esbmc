class V:
    def __init__(self, x: int):
        self.x = x

    def scale(self, k: int) -> int:
        return self.x * k


def main() -> None:
    v = V(2)
    assert getattr(v, "scale")(3) == 6
    assert getattr(V(5), "scale")(k=2) == 10


main()
