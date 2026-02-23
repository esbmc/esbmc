from typing import List


def f(numbers: List[float]) -> float:
    total: float = 0.0
    i: int = 0
    n: int = len(numbers)
    while i < n:
        total = total + numbers[i]
        i = i + 1
    return total


if __name__ == "__main__":
    assert abs(f([1.0, 2.0, 3.0]) - 6.0) < 1e-6
