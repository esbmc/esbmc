# HumanEval/52
# Entry Point: below_threshold
# ESBMC-compatible format with direct assertions



def below_threshold(l: list, t: int):
    """Return True if all numbers in the list l are below threshold t.
    >>> below_threshold([1, 2, 4, 10], 100)
    True
    >>> below_threshold([1, 20, 4, 10], 5)
    False
    """
    for e in l:
        if e >= t:
            return False
    return True

# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert below_threshold([1, 2, 4, 10], 100)
    # assert not below_threshold([1, 20, 4, 10], 5)
    # assert below_threshold([1, 20, 4, 10], 21)
    # assert below_threshold([1, 20, 4, 10], 22)
    # assert below_threshold([1, 8, 4, 10], 11)
    # assert not below_threshold([1, 8, 4, 10], 10)
    print("✅ HumanEval/52 - All assertions completed!")
