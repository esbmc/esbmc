from typing import List


def f(numbers: List[float]) -> int:
    n: int = len(numbers)
    return n


if __name__ == "__main__":
    assert f([1.0, 2.0, 3.0]) == 3
