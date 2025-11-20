# HumanEval/4
# Entry Point: mean_absolute_deviation
# ESBMC-compatible format with direct assertions

from typing import List


def mean_absolute_deviation(numbers: List[float]) -> float:
    """ For a given list of input numbers, calculate Mean Absolute Deviation
    around the mean of this dataset.
    Mean Absolute Deviation is the average absolute difference between each
    element and a centerpoint (mean in this case):
    MAD = average | x - x_mean |
    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])
    1.0
    """
    mean = sum(numbers) / len(numbers)
    return sum(abs(x - mean) for x in numbers) / len(numbers)


# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert abs(mean_absolute_deviation([1.0, 2.0, 3.0]) - 2.0/3.0) < 1e-6
    assert abs(mean_absolute_deviation([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-6
    assert abs(mean_absolute_deviation([1.0, 2.0, 3.0, 4.0, 5.0]) - 6.0/5.0) < 1e-6
    print("✅ HumanEval/4 - All assertions completed!")

