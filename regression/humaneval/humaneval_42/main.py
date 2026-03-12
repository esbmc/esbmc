# HumanEval/42
# Entry Point: incr_list
# ESBMC-compatible format with direct assertions



def incr_list(l: list):
    """Return list with elements incremented by 1.
    >>> incr_list([1, 2, 3])
    [2, 3, 4]
    >>> incr_list([5, 3, 5, 2, 3, 3, 9, 0, 123])
    [6, 4, 6, 3, 4, 4, 10, 1, 124]
    """
    return [(e + 1) for e in l]

# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert incr_list([]) == []
    # assert incr_list([3, 2, 1]) == [4, 3, 2]
    # assert incr_list([5, 2, 5, 2, 3, 3, 9, 0, 123]) == [6, 3, 6, 3, 4, 4, 10, 1, 124]
    print("✅ HumanEval/42 - All assertions completed!")
