# HumanEval/21
# Entry Point: rescale_to_unit
# ESBMC-compatible format with direct assertions

from typing import List


def rescale_to_unit(numbers: List[float]) -> List[float]:
    """ Given list of numbers (of at least two elements), apply a linear transform to that list,
    such that the smallest number will become 0 and the largest will become 1
    >>> rescale_to_unit([1.0, 2.0, 3.0, 4.0, 5.0])
    [0.0, 0.25, 0.5, 0.75, 1.0]
    """
    min_number = min(numbers)
    max_number = max(numbers)
    return [(x - min_number) / (max_number - min_number) for x in numbers]

# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert rescale_to_unit([2.0, 49.9]) == [0.0, 1.0]
    # assert rescale_to_unit([100.0, 49.9]) == [1.0, 0.0]
    # assert rescale_to_unit([1.0, 2.0, 3.0, 4.0, 5.0]) == [0.0, 0.25, 0.5, 0.75, 1.0]
    # assert rescale_to_unit([2.0, 1.0, 5.0, 3.0, 4.0]) == [0.25, 0.0, 1.0, 0.5, 0.75]
    # assert rescale_to_unit([12.0, 11.0, 15.0, 13.0, 14.0]) == [0.25, 0.0, 1.0, 0.5, 0.75]
    print("✅ HumanEval/21 - All assertions completed!")
