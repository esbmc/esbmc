pi:float = 3.14153

def comb(n: int, k: int) -> int:
    """
    Calculate binomial coefficient C(n, k) = n! / (k! * (n-k)!)
    
    Args:
        n: Total number of items (non-negative integer)
        k: Number of items to choose (non-negative integer)
    
    Returns:
        The binomial coefficient, or 0 if k > n
        
    Raises:
        ValueError: If n or k are negative
        TypeError: If n or k are not integers
    """
    # Type checking
    if not isinstance(n, int) or not isinstance(k, int):
        assert False
    # Handle negative inputs
    if n < 0 or k < 0:
        assert False

    # Handle edge cases
    if k > n:
        return 0
    if k == 0 or k == n:
        return 1
    if k == 1 or k == n - 1:
        return n

    # Optimize by using the smaller of k and n-k
    if k > n - k:
        k = n - k

    # Calculate C(n, k) using the multiplicative formula
    # C(n, k) = n * (n-1) * ... * (n-k+1) / (k * (k-1) * ... * 1)
    result: int = 1
    i: int = 0
    while i < k:
        result = result * (n - i) // (i + 1)
        i = i + 1
    return result

def isinf(x: float) -> bool:
    return __ESBMC_isinf(x)

def isnan(x: float) -> bool:
    return __ESBMC_isnan(x)

def floor(x: float) -> int:
    # infinity and NaN inputs cause assertion failures
    # since they cannot be converted to integers.
    assert not isinf(x), "Input cannot be infinity"
    assert not isnan(x), "Input cannot be NaN"

    if x >= 0:
        return int(x)
    else:
        int_x: int = int(x)
        if x == int_x:
            return int_x
        else:
            return int_x - 1

def ceil(x: float) -> int:
    # infinity and NaN inputs cause assertion failures
    # since they cannot be converted to integers.
    assert not isinf(x), "Input cannot be infinity"
    assert not isnan(x), "Input cannot be NaN"

    if x <= 0:
        return int(x)
    else:
        int_x: int = int(x)
        if x == int_x:
            return int_x
        else:
            return int_x + 1