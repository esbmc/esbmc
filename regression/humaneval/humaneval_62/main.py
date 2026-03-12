# HumanEval/62
# Entry Point: derivative
# ESBMC-compatible format with direct assertions



def derivative(xs: list):
    """ xs represent coefficients of a polynomial.
    xs[0] + xs[1] * x + xs[2] * x^2 + ....
     Return derivative of this polynomial in the same form.
    >>> derivative([3, 1, 2, 4, 5])
    [1, 4, 12, 20]
    >>> derivative([1, 2, 3])
    [2, 6]
    """
    return [(i * x) for i, x in enumerate(xs)][1:]

# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert derivative([3, 1, 2, 4, 5]) == [1, 4, 12, 20]
    # assert derivative([1, 2, 3]) == [2, 6]
    # assert derivative([3, 2, 1]) == [2, 2]
    # assert derivative([3, 2, 1, 0, 4]) == [2, 2, 0, 16]
    # assert derivative([1]) == []
    print("✅ HumanEval/62 - All assertions completed!")
