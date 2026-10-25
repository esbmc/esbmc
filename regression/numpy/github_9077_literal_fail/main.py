import numpy as np


def columns() -> int:
    adata = np.array([[0.0, 1.0, 2.0]])
    shape = adata.shape
    return 10 // (shape[1] - 3)


def main() -> None:
    columns()


main()
