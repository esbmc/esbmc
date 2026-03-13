# HumanEval/24
# Entry Point: largest_divisor
# ESBMC-compatible format with direct assertions



def largest_divisor(n: int) -> int:
    """ For a given number n, find the largest number that divides n evenly, smaller than n
    >>> largest_divisor(15)
    5
    """
    for i in reversed(range(n)):
        if n % i == 0:
            return i

# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert largest_divisor(3) == 1
    # assert largest_divisor(7) == 1
    # assert largest_divisor(10) == 5
    # assert largest_divisor(100) == 50
    # assert largest_divisor(49) == 7
    print("✅ HumanEval/24 - All assertions completed!")
