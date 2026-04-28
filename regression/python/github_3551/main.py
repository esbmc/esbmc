from typing import List


def mean_absolute_deviation(numbers: List[float]) -> float:
    total: float = 0.0
    i: int = 0
    length: int = len(numbers)
    while i < length:
        total = total + numbers[i]
        i = i + 1
    mean: float = total / length

    mad: float = 0.0
    i = 0
    while i < length:
        diff: float = numbers[i] - mean
        if diff < 0.0:
            diff = -diff
        mad = mad + diff
        i = i + 1
    return mad / length


if __name__ == "__main__":
    assert abs(mean_absolute_deviation([1.0, 2.0, 3.0]) - 2.0/3.0) < 1e-6
