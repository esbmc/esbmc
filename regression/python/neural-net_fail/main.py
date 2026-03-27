def relu(x: float) -> float:
    if x > 0:
        return x
    else:
        return 0.0


def main() -> None:
    x: float = 0.749
    y: float = 0.498
    nodeA: float = 2 * x - 3 * y
    nodeB: float = x + 4 * y
    f: float = relu(nodeA) + relu(nodeB)
    assert f >= 2.745


main()
