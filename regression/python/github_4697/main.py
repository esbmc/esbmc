def matvec(W: list[list[float]], x: list[float]) -> list[float]:
    y: list[float] = [0.0, 0.0]
    for i in range(2):
        s: float = 0.0
        for j in range(2):
            s = s + W[i][j] * x[j]
        y[i] = s
    return y


def main() -> None:
    x: list[float] = [1.0, 2.0]
    W: list[list[float]] = [[1.0, 0.0], [0.0, 1.0]]
    y = matvec(W, x)
    assert y[0] == 1.0
    assert y[1] == 2.0


if __name__ == "__main__":
    main()
