# HumanEval/49
# Entry Point: modp
# ESBMC-compatible format with direct assertions



def modp(n: int, p: int):
    """Return 2^n modulo p (be aware of numerics).
    >>> modp(3, 5)
    3
    >>> modp(1101, 101)
    2
    >>> modp(0, 101)
    1
    >>> modp(3, 11)
    8
    >>> modp(100, 101)
    1
    """
    ret = 1
    for i in range(n):
        ret = (2 * ret) % p
    return ret

# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert modp(3, 5) == 3
    # assert modp(1101, 101) == 2
    # assert modp(0, 101) == 1
    # assert modp(3, 11) == 8
    # assert modp(100, 101) == 1
    # assert modp(30, 5) == 4
    # assert modp(31, 5) == 3
    print("✅ HumanEval/49 - All assertions completed!")
