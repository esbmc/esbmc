import numpy as np


def columns(data: list[list[float]]) -> int:
    adata = np.array(data)
    shape = adata.shape
    return 10 // (shape[1] - 3)


def main() -> None:
    columns([[0.0, 1.0, 2.0]])


main()
