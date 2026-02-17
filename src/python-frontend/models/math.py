def __ESBMC_expm1(x: float) -> float:
    ...


def __ESBMC_log1p(x: float) -> float:
    ...


def __ESBMC_exp2(x: float) -> float:
    ...


def __ESBMC_asinh(x: float) -> float:
    ...


def __ESBMC_acosh(x: float) -> float:
    ...


def __ESBMC_atanh(x: float) -> float:
    ...


def __ESBMC_hypot(x: float, y: float) -> float:
    ...


pi: float = 3.14153
e: float = 2.71828
inf: float = float('inf')
tau: float = 6.28306
nan: float = float('nan')


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
    if x <= 0:
        raise ValueError("math domain error")
    return __ESBMC_log(x)


def factorial(n: int) -> int:
    if not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n < 0:
        raise ValueError("n must be >= 0")
    result: int = 1
    i: int = 2
    while i <= n:
        result = result * i
        i = i + 1
    return result


def gcd(a: int, b: int) -> int:
    if not isinstance(a, int) or not isinstance(b, int):
        raise TypeError("gcd() arguments must be integers")
    if a < 0:
        a = 0 - a
    if b < 0:
        b = 0 - b
    while b != 0:
        tmp = a % b
        a = b
        b = tmp
    return a


def lcm(a: int, b: int) -> int:
    if not isinstance(a, int) or not isinstance(b, int):
        raise TypeError("lcm() arguments must be integers")
    if a == 0 or b == 0:
        return 0
    g = gcd(a, b)
    if a < 0:
        a = 0 - a
    if b < 0:
        b = 0 - b
    return (a // g) * b


def isqrt(n: int) -> int:
    if not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n < 0:
        raise ValueError("n must be >= 0")
    if n == 0:
        return 0
    x: int = n
    y: int = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x


def perm(n: int, k: int = -1) -> int:
    if not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n < 0:
        raise ValueError("n must be >= 0")
    if not isinstance(k, int):
        raise TypeError("k must be an integer")
    if k < 0:
        return factorial(n)
    if k > n:
        return 0
    result: int = 1
    i: int = 0
    while i < k:
        result = result * (n - i)
        i = i + 1
    return result


def prod(values: list[int], start: int = 1) -> int:
    result = start
    i = 0
    while i < len(values):
        result = result * values[i]
        i = i + 1
    return result


def isclose(a: float, b: float, rel_tol: float = 1e-09, abs_tol: float = 0.0) -> bool:
    if rel_tol < 0.0 or abs_tol < 0.0:
        raise ValueError("tolerances must be non-negative")
    diff = a - b
    if diff < 0.0:
        diff = 0.0 - diff
    abs_a = a
    if abs_a < 0.0:
        abs_a = 0.0 - abs_a
    abs_b = b
    if abs_b < 0.0:
        abs_b = 0.0 - abs_b
    max_ab = abs_a
    if abs_b > max_ab:
        max_ab = abs_b
    limit = rel_tol * max_ab
    if abs_tol > limit:
        limit = abs_tol
    return diff <= limit


def expm1(x: float) -> float:
    return __ESBMC_expm1(x)


def log1p(x: float) -> float:
    if x <= -1.0:
        raise ValueError("math domain error")
    return __ESBMC_log1p(x)


def exp2(x: float) -> float:
    return __ESBMC_exp2(x)


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

    Raises:
        ValueError: If |x| > 1 (math domain error)
    """
    if x < -1.0 or x > 1.0:
        raise ValueError("math domain error")
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


def asinh(x: float) -> float:
    return __ESBMC_asinh(x)


def acosh(x: float) -> float:
    if x < 1.0:
        raise ValueError("math domain error")
    return __ESBMC_acosh(x)


def atanh(x: float) -> float:
    if x <= -1.0 or x >= 1.0:
        raise ValueError("math domain error")
    return __ESBMC_atanh(x)


def hypot(x: float, y: float) -> float:
    return __ESBMC_hypot(x, y)


def dist(p: list[float], q: list[float]) -> float:
    if len(p) != len(q):
        raise ValueError("points must have the same dimension")
    total = 0.0
    i = 0
    while i < len(p):
        d = p[i] - q[i]
        total = total + d * d
        i = i + 1
    return sqrt(total)


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
