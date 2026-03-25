# HumanEval/35
# Entry Point: max_element
# ESBMC-compatible format with direct assertions



def max_element(l: list):
    """Return maximum element in the list.
    >>> max_element([1, 2, 3])
    3
    >>> max_element([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10])
    123
    """
    m = l[0]
    for e in l:
        if e > m:
            m = e
    return m

# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert max_element([1, 2, 3]) == 3
    # assert max_element([5, 3, -5, 2, -3, 3, 9, 0, 124, 1, -10]) == 124
    print("✅ HumanEval/35 - All assertions completed!")
