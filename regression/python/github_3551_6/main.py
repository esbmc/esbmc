from typing import List


def f(numbers: List[float]) -> float:
    i: int = 0
    mean: float = 2.0
    diff: float = numbers[i] - mean
    if diff < 0.0:
        diff = -diff
    return diff


if __name__ == "__main__":
    assert abs(f([1.0]) - 1.0) < 1e-6
