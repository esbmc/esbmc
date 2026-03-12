# HumanEval/34
# Entry Point: unique
# ESBMC-compatible format with direct assertions



def unique(l: list):
    """Return sorted unique elements in a list
    >>> unique([5, 3, 5, 2, 3, 3, 9, 0, 123])
    [0, 2, 3, 5, 9, 123]
    """
    return sorted(list(set(l)))

# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert unique([5, 3, 5, 2, 3, 3, 9, 0, 123]) == [0, 2, 3, 5, 9, 123]
    print("✅ HumanEval/34 - All assertions completed!")
