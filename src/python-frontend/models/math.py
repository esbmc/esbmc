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


def cbrt(x: float) -> float:
    """
    Calculate the cube root of x
    """
    if x == 0.0:
        return 0.0
    if x < 0.0:
        return 0.0 - pow(0.0 - x, 1.0 / 3.0)
    return pow(x, 1.0 / 3.0)


def erf(x: float) -> float:
    """
    Calculate the error function of x
    """
    # Abramowitz and Stegun 7.1.26 approximation
    p: float = 0.3275911
    a1: float = 0.254829592
    a2: float = -0.284496736
    a3: float = 1.421413741
    a4: float = -1.453152027
    a5: float = 1.061405429

    sign: float = 1.0
    abs_x: float = x
    if abs_x < 0.0:
        sign = 0.0 - 1.0
        abs_x = 0.0 - abs_x

    t: float = 1.0 / (1.0 + p * abs_x)
    y: float = 1.0 - (
        (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t
    ) * exp(0.0 - abs_x * abs_x)
    return sign * y


def erfc(x: float) -> float:
    """
    Calculate the complementary error function of x
    """
    return 1.0 - erf(x)


def frexp(x: float) -> tuple[float, int]:
    """
    Split x into mantissa and exponent (x = m * 2**e),
    with m in [0.5, 1.0) (or 0.0 if x == 0)
    """
    if x == 0.0:
        return (0.0, 0)

    e: int = 0
    ax: float = x
    if ax < 0.0:
        ax = 0.0 - ax

    max_iter: int = 2048
    iter_count: int = 0
    while ax < 0.5 and iter_count < max_iter:
        ax = ax * 2.0
        e = e - 1
        iter_count = iter_count + 1
    if iter_count >= max_iter:
        raise ValueError("frexp loop bound exceeded")

    iter_count = 0
    while ax >= 1.0 and iter_count < max_iter:
        ax = ax / 2.0
        e = e + 1
        iter_count = iter_count + 1
    if iter_count >= max_iter:
        raise ValueError("frexp loop bound exceeded")

    if x < 0.0:
        ax = 0.0 - ax
    return (ax, e)


def fsum(values: list[float]) -> float:
    """
    Return an accurate floating point sum of values
    """
    total: float = 0.0
    c: float = 0.0
    i: int = 0
    n: int = len(values)
    while i < n:
        y: float = values[i] - c
        t: float = total + y
        c = (t - total) - y
        total = t
        i = i + 1
    return total


def _lanczos_sum(z: float) -> float:
    """
    Lanczos approximation sum for gamma/lgamma
    """
    a: float = 0.99999999999980993
    a = a + 676.5203681218851 / (z + 1.0)
    a = a + -1259.1392167224028 / (z + 2.0)
    a = a + 771.3234287776531 / (z + 3.0)
    a = a + -176.6150291621406 / (z + 4.0)
    a = a + 12.507343278686905 / (z + 5.0)
    a = a + -0.13857109526572012 / (z + 6.0)
    a = a + 0.000009984369578019572 / (z + 7.0)
    a = a + 0.00000015056327351493116 / (z + 8.0)
    return a


def gamma(x: float) -> float:
    """
    Calculate the gamma function of x.
    For positive integers x, this function satisfies gamma(x) = (x - 1)!.
    """
    if x == int(x):
        xi: int = int(x)
        if xi <= 0:
            raise ValueError("math domain error")
        # For positive integer xi, Gamma(xi) = (xi - 1)!, which we compute explicitly.
        factorial_x_minus_1: int = 1
        i: int = 2
        while i <= xi - 1:
            factorial_x_minus_1 = factorial_x_minus_1 * i
            i = i + 1
        return float(factorial_x_minus_1)

    if x <= 0.0:
        raise ValueError("math domain error")

    # Lanczos approximation (no reflection, valid for x > 0)
    z: float = x - 1.0
    a: float = _lanczos_sum(z)

    t: float = z + 7.0 + 0.5
    return sqrt(2.0 * pi) * pow(t, z + 0.5) * exp(0.0 - t) * a


def ldexp(x: float, i: int) -> float:
    """
    Return x * (2**i)
    """
    if not isinstance(i, int):
        raise TypeError("i must be an integer")
    return x * pow(2.0, float(i))


def lgamma(x: float) -> float:
    """
    Calculate the natural logarithm of the absolute value of the gamma function
    """
    if x == int(x):
        xi: int = int(x)
        if xi <= 0:
            raise ValueError("math domain error")
        result: float = 0.0
        i: int = 2
        while i <= xi - 1:
            result = result + log(i * 1.0)
            i = i + 1
        return result

    if x <= 0.0:
        raise ValueError("math domain error")

    z: float = x - 1.0
    a: float = _lanczos_sum(z)

    t: float = z + 7.0 + 0.5
    return 0.5 * log(2.0 * pi) + (z + 0.5) * log(t) - t + log(a)


def remainder(x: float, y: float) -> float:
    """
    Return IEEE 754-style remainder of x with respect to y
    """
    if y == 0.0:
        raise ValueError("math domain error")

    q: float = x / y
    n_int: int = trunc(q)
    n: float = n_int * 1.0
    frac: float = q - n

    if frac > 0.5:
        n = n + 1.0
    elif frac < -0.5:
        n = n - 1.0
    elif frac == 0.5 or frac == -0.5:
        if fmod(n, 2.0) != 0.0:
            if q > 0.0:
                n = n + 1.0
            else:
                n = n - 1.0

    return x - n * y


def sumprod(a: list[float], b: list[float]) -> float:
    """
    Return the sum of products of two sequences
    """
    if len(a) != len(b):
        raise ValueError("sumprod() requires sequences of the same length")

    total: float = 0.0
    i: int = 0
    n: int = len(a)
    while i < n:
        total = total + a[i] * b[i]
        i = i + 1
    return total


def ulp(x: float) -> float:
    """
    Return the value of the least significant bit of the float x
    """
    if x == 0.0:
        return pow(2.0, -1074.0)

    ax: float = fabs(x)
    _, e = frexp(ax)
    return ldexp(1.0, e - 53)


def nextafter(x: float, y: float) -> float:
    """
    Return the next floating-point value after x towards y
    """
    if isnan(x) or isnan(y):
        return float('nan')
    if x == y:
        return x
    if x == 0.0:
        return copysign(pow(2.0, -1074.0), y)

    step: float = ulp(x)
    if y < x:
        step = 0.0 - step
    return x + step
