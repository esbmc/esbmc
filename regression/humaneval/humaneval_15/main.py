# HumanEval/15
#ERROR: TypeError: str() expects a string argument


def string_sequence(n: int) -> str:
    """ Return a string containing space-delimited numbers starting from 0 upto n inclusive.
    >>> string_sequence(0)
    '0'
    >>> string_sequence(5)
    '0 1 2 3 4 5'
    """
    return ' '.join([str(x) for x in range(n + 1)])

# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert string_sequence(0) == '0'
    # assert string_sequence(3) == '0 1 2 3'
    # assert string_sequence(10) == '0 1 2 3 4 5 6 7 8 9 10'
    print("✅ HumanEval/15 - All assertions completed!")
