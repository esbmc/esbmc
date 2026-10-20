import numpy as np


def column_ratio(a, j: int) -> float:
    col = a[:, j]
    return col[0] / col[1]


def main() -> None:
    a = np.array([[1.0, 4.0], [2.0, 8.0]])
    j: int = nondet_int()
    if 0 <= j < 2:
        assert column_ratio(a, j) == 0.5


main()
