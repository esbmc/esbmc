pi: float = 3.14153
e: float = 2.71828
inf: float = float('inf')


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
        raise TypeError("Both n and k must be integers")
    # Handle negative inputs
    if n < 0 or k < 0:
        raise ValueError("Both n and k must be non-negative integers")

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


def sin(x: float) -> float:
    """
    Calculate sine of x (in radians)

    Args:
        x: Input angle in radians

    Returns:
        Sine of x
    """
    return __ESBMC_sin(x)


def cos(x: float) -> float:
    """
    Calculate cosine of x (in radians)

    Args:
        x: Input angle in radians

    Returns:
        Cosine of x
    """
    return __ESBMC_cos(x)


def tan(x: float) -> float:
    """
    Calculate tangent of x (in radians)
    """
    return __ESBMC_tan(x)


def sqrt(x: float) -> float:
    """
    Calculate square root of x

    Args:
        x: Non-negative number

    Returns:
        Square root of x

    Raises:
        ValueError: If x is negative (math domain error)
    """
    if x < 0:
        raise ValueError("math domain error")

    return __ESBMC_sqrt(x)


def exp(x: float) -> float:
    """
    Calculate e raised to the power of x

    Args:
        x: Input value

    Returns:
        e^x
    """
    return __ESBMC_exp(x)


def log(x: float) -> float:
    """
    Calculate natural logarithm of x

    Args:
        x: Positive number

    Returns:
        Natural logarithm of x

    Raises:
        ValueError: If x <= 0 (math domain error)
    """
    return __ESBMC_log(x)


def asin(x: float) -> float:
    """
    Calculate arcsine of x (in radians)

    Raises:
        ValueError: If |x| > 1 (math domain error)
    """
    if x < -1.0 or x > 1.0:
        raise ValueError("math domain error")
    return __ESBMC_asin(x)


def acos(x: float) -> float:
    """
    Calculate arccosine of x (in radians)
    """
    return __ESBMC_acos(x)


def atan(x: float) -> float:
    """
    Calculate arctangent of x (in radians)
    """
    return __ESBMC_atan(x)


def atan2(y: float, x: float) -> float:
    """
    Calculate two-argument arctangent (in radians)
    """
    return __ESBMC_atan2(y, x)


def log2(x: float) -> float:
    """
    Calculate base-2 logarithm of x

    Raises:
        ValueError: If x <= 0 (math domain error)
    """
    if x <= 0:
        raise ValueError("math domain error")
    return __ESBMC_log2(x)


def log10(x: float) -> float:
    """
    Calculate base-10 logarithm of x

    Raises:
        ValueError: If x <= 0 (math domain error)
    """
    if x <= 0:
        raise ValueError("math domain error")
    return __ESBMC_log10(x)


def pow(x: float, y: float) -> float:
    """
    Calculate x raised to the power of y
    """
    return __ESBMC_pow(x, y)


def fabs(x: float) -> float:
    """
    Calculate absolute value of x
    """
    return __ESBMC_fabs(x)


def trunc(x: float) -> int:
    """
    Truncate x toward zero and return int
    """
    return int(x)


def fmod(x: float, y: float) -> float:
    """
    Floating-point remainder of x / y
    """
    return __ESBMC_fmod(x, y)


def copysign(x: float, y: float) -> float:
    """
    Return x with the sign of y
    """
    return __ESBMC_copysign(x, y)


def sinh(x: float) -> float:
    """
    Calculate hyperbolic sine of x
    """
    return __ESBMC_sinh(x)


def cosh(x: float) -> float:
    """
    Calculate hyperbolic cosine of x
    """
    return __ESBMC_cosh(x)


def tanh(x: float) -> float:
    """
    Calculate hyperbolic tangent of x
    """
    return __ESBMC_tanh(x)


def isfinite(x: float) -> bool:
    return (not isinf(x)) and (not isnan(x))


def degrees(x: float) -> float:
    return x * (180.0 / pi)


def radians(x: float) -> float:
    return x * (pi / 180.0)


def modf(x: float) -> tuple[float, float]:
    """
    Split x into fractional and integer parts
    """
    int_part: float = float(int(x))
    frac_part: float = x - int_part
    return (frac_part, int_part)
