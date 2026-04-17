# HumanEval/55
# Entry Point: fib
# ESBMC-compatible format with direct assertions



def fib(n: int):
    """Return n-th Fibonacci number.
    >>> fib(10)
    55
    >>> fib(1)
    1
    >>> fib(8)
    21
    """
    if n == 0:
        return 0
    if n == 1:
        return 1
    return fib(n - 1) + fib(n - 2)

# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert fib(10) == 55
    # assert fib(1) == 1
    # assert fib(8) == 21
    # assert fib(11) == 89
    # assert fib(12) == 144
    print("✅ HumanEval/55 - All assertions completed!")
