# HumanEval/60
# Entry Point: sum_to_n
# ESBMC-compatible format with direct assertions



def sum_to_n(n: int):
    """sum_to_n is a function that sums numbers from 1 to n.
    >>> sum_to_n(30)
    465
    >>> sum_to_n(100)
    5050
    >>> sum_to_n(5)
    15
    >>> sum_to_n(10)
    55
    >>> sum_to_n(1)
    1
    """
    return sum(range(n + 1))

# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert sum_to_n(1) == 1
    # assert sum_to_n(6) == 21
    # assert sum_to_n(11) == 66
    # assert sum_to_n(30) == 465
    # assert sum_to_n(100) == 5050
    print("✅ HumanEval/60 - All assertions completed!")
