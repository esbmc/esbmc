# HumanEval/100
# Entry Point: make_a_pile
# ESBMC-compatible format with direct assertions


def make_a_pile(n):
    """
    Given a positive integer n, you have to make a pile of n levels of stones.
    The first level has n stones.
    The number of stones in the next level is:
        - the next odd number if n is odd.
        - the next even number if n is even.
    Return the number of stones in each level in a list, where element at index
    i represents the number of stones in the level (i+1).

    Examples:
    >>> make_a_pile(3)
    [3, 5, 7]
    """
    return [n + 2*i for i in range(n)]

# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert make_a_pile(3) == [3, 5, 7], "Test 3"
    # assert make_a_pile(4) == [4,6,8,10], "Test 4"
    # assert make_a_pile(5) == [5, 7, 9, 11, 13]
    # assert make_a_pile(6) == [6, 8, 10, 12, 14, 16]
    # assert make_a_pile(8) == [8, 10, 12, 14, 16, 18, 20, 22]
    # assert True, "This prints if this assert fails 2 (also good for debugging!)"
    print("✅ HumanEval/100 - All assertions completed!")
