# HumanEval/106
# Entry Point: f
# ESBMC-compatible format with direct assertions


def f(n):
    """ Implement the function f that takes n as a parameter,
    and returns a list of size n, such that the value of the element at index i is the factorial of i if i is even
    or the sum of numbers from 1 to i otherwise.
    i starts from 1.
    the factorial of i is the multiplication of the numbers from 1 to i (1 * 2 * ... * i).
    Example:
    f(5) == [1, 2, 6, 24, 15]
    """
    ret = []
    for i in range(1,n+1):
        if i%2 == 0:
            x = 1
            for j in range(1,i+1): x *= j
            ret += [x]
        else:
            x = 0
            for j in range(1,i+1): x += j
            ret += [x]
    return ret

# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert f(5) == [1, 2, 6, 24, 15]
    # assert f(7) == [1, 2, 6, 24, 15, 720, 28]
    # assert f(1) == [1]
    # assert f(3) == [1, 2, 6]
    print("✅ HumanEval/106 - All assertions completed!")
