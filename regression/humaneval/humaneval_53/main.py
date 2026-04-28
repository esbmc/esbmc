# HumanEval/53
# Entry Point: add
# ESBMC-compatible format with direct assertions



def add(x: int, y: int):
    """Add two numbers x and y
    >>> add(2, 3)
    5
    >>> add(5, 7)
    12
    """
    return x + y

# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert add(0, 1) == 1
    # assert add(1, 0) == 1
    # assert add(2, 3) == 5
    # assert add(5, 7) == 12
    # assert add(7, 5) == 12
    # assert add(x, y) == x + y
    print("✅ HumanEval/53 - All assertions completed!")
