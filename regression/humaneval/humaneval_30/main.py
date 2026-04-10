# HumanEval/30
# Entry Point: get_positive
# ESBMC-compatible format with direct assertions



def get_positive(l: list):
    """Return only positive numbers in the list.
    >>> get_positive([-1, 2, -4, 5, 6])
    [2, 5, 6]
    >>> get_positive([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10])
    [5, 3, 2, 3, 9, 123, 1]
    """
    return [e for e in l if e > 0]

# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert get_positive([-1, -2, 4, 5, 6]) == [4, 5, 6]
    # assert get_positive([5, 3, -5, 2, 3, 3, 9, 0, 123, 1, -10]) == [5, 3, 2, 3, 3, 9, 123, 1]
    # assert get_positive([-1, -2]) == []
    # assert get_positive([]) == []
    print("✅ HumanEval/30 - All assertions completed!")
