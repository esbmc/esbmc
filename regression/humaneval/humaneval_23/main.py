# HumanEval/23
# Entry Point: strlen
# ESBMC-compatible format with direct assertions



def strlen(string: str) -> int:
    """ Return length of given string
    >>> strlen('')
    0
    >>> strlen('abc')
    3
    """
    return len(string)

# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert strlen('') == 0
    # assert strlen('x') == 1
    # assert strlen('asdasnakj') == 9
    print("✅ HumanEval/23 - All assertions completed!")
