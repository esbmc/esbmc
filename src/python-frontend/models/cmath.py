import math


pi: float = math.pi
e: float = math.e
tau: float = math.tau
inf: float = math.inf
nan: float = math.nan
infj: complex = complex(0.0, inf)
nanj: complex = complex(0.0, nan)
_I: complex = complex(0.0, 1.0)
_NEG_I: complex = complex(0.0, -1.0)
_MIN_POSITIVE: float = 2.2250738585072014e-308
_LN2: float = 0.6931471805599453
_LN10: float = 2.302585092994046


def phase(z: complex) -> float:
    if z.real == 0.0 and z.imag == 0.0:
        return 0.0
    return math.atan2(z.imag, z.real)


def _safe_sqrt_nonnegative(x: float) -> float:
    if x <= 0.0:
        return 0.0
    return math.sqrt(x)


def _sign(x: float) -> float:
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return -1.0
    return 0.0


def _clamp_unit(x: float) -> float:
    if x > 1.0:
        return 1.0
    if x < -1.0:
        return -1.0
    return x


def _safe_acosh_arg(x: float) -> float:
    if x < 1.0:
        return 1.0
    return x


def _norm2(a: float, b: float) -> float:
    return _safe_sqrt_nonnegative(a * a + b * b)


def _real_asin_unit(x: float) -> float:
    x = _clamp_unit(x)
    return math.atan2(x, _safe_sqrt_nonnegative(1.0 - x * x))


def _real_acos_unit(x: float) -> float:
    x = _clamp_unit(x)
    return math.atan2(_safe_sqrt_nonnegative(1.0 - x * x), x)


def _real_acosh_ge1(x: float) -> float:
    x = _safe_acosh_arg(x)
    return math.log(x + _safe_sqrt_nonnegative(x - 1.0) * _safe_sqrt_nonnegative(x + 1.0))


def _abs_complex(z: complex) -> float:
    mag2 = z.real * z.real + z.imag * z.imag
    return _safe_sqrt_nonnegative(mag2)


def polar(z: complex) -> tuple[float, float]:
    if z.real == 0.0 and z.imag == 0.0:
        return (0.0, 0.0)
    if z.imag == 0.0 and z.real > 0.0:
        return (z.real, 0.0)
    return (_abs_complex(z), phase(z))


def rect(r: float, phi: float) -> complex:
    return complex(r * math.cos(phi), r * math.sin(phi))


def exp(z: complex) -> complex:
    ea = math.exp(z.real)
    return complex(ea * math.cos(z.imag), ea * math.sin(z.imag))


def _safe_positive(x: float) -> float:
    if x > 0.0:
        return x
    return _MIN_POSITIVE


def _real_log_positive(x: float) -> float:
    x1 = x
    k = 0

    if x1 > 2.0:
        x1 = x1 * 0.5
        k = k + 1
    if x1 > 2.0:
        x1 = x1 * 0.5
        k = k + 1
    if x1 > 2.0:
        x1 = x1 * 0.5
        k = k + 1
    if x1 > 2.0:
        x1 = x1 * 0.5
        k = k + 1

    if x1 < 0.5:
        x1 = x1 * 2.0
        k = k - 1
    if x1 < 0.5:
        x1 = x1 * 2.0
        k = k - 1
    if x1 < 0.5:
        x1 = x1 * 2.0
        k = k - 1
    if x1 < 0.5:
        x1 = x1 * 2.0
        k = k - 1

    y = (x1 - 1.0) / (x1 + 1.0)
    y2 = y * y
    y3 = y * y2
    y5 = y3 * y2
    y7 = y5 * y2
    y9 = y7 * y2
    y11 = y9 * y2

    series = y + (y3 / 3.0) + (y5 / 5.0) + (y7 / 7.0) + (y9 / 9.0) + (y11 / 11.0)
    return (2.0 * series) + (float(k) * _LN2)


def log(z: complex, base: float = 0.0) -> complex:
    if z.real == 1.0 and z.imag == 0.0:
        v = complex(0.0, 0.0)
    elif z.imag == 0.0 and z.real > 0.0:
        v = complex(_real_log_positive(z.real), 0.0)
    else:
        mag = _abs_complex(z)
        v = complex(_real_log_positive(_safe_positive(mag)), phase(z))
    if base == 0.0:
        return v
    return v / _real_log_positive(_safe_positive(base))


def log10(z: complex) -> complex:
    if z.real == 1.0 and z.imag == 0.0:
        return complex(0.0, 0.0)
    return log(z) / _LN10


def sqrt(z: complex) -> complex:
    if z.real == 0.0 and z.imag == 0.0:
        return complex(0.0, 0.0)

    r = _abs_complex(z)
    real_term = (r + z.real) / 2.0
    if real_term < 0.0:
        real_term = 0.0

    imag_term = (r - z.real) / 2.0
    if imag_term < 0.0:
        imag_term = 0.0

    real = _safe_sqrt_nonnegative(real_term)
    imag_mag = _safe_sqrt_nonnegative(imag_term)
    imag = imag_mag if z.imag >= 0.0 else -imag_mag
    return complex(real, imag)


def sin(z: complex) -> complex:
    a = z.real
    b = z.imag
    return complex(math.sin(a) * math.cosh(b), math.cos(a) * math.sinh(b))


def cos(z: complex) -> complex:
    a = z.real
    b = z.imag
    return complex(math.cos(a) * math.cosh(b), -math.sin(a) * math.sinh(b))


def tan(z: complex) -> complex:
    return sin(z) / cos(z)


def sinh(z: complex) -> complex:
    a = z.real
    b = z.imag
    return complex(math.sinh(a) * math.cos(b), math.cosh(a) * math.sin(b))


def cosh(z: complex) -> complex:
    a = z.real
    b = z.imag
    return complex(math.cosh(a) * math.cos(b), math.sinh(a) * math.sin(b))


def tanh(z: complex) -> complex:
    return sinh(z) / cosh(z)


def asin(z: complex) -> complex:
    if z.real == 0.0 and z.imag == 0.0:
        return complex(0.0, 0.0)
    if z.real == 0.0:
        return complex(0.0, math.asinh(z.imag))
    if z.imag == 0.0 and z.real >= -1.0 and z.real <= 1.0:
        return complex(math.asin(z.real), 0.0)
    if z.imag == 0.0 and z.real > 1.0:
        return complex(pi / 2.0, math.acosh(z.real))
    if z.imag == 0.0 and z.real < -1.0:
        return complex(-pi / 2.0, math.acosh(0.0 - z.real))

    a = math.sqrt((z.real + 1.0) * (z.real + 1.0) + z.imag * z.imag)
    b = math.sqrt((z.real - 1.0) * (z.real - 1.0) + z.imag * z.imag)
    t = (a - b) / 2.0
    if t > 1.0:
        t = 1.0
    if t < -1.0:
        t = -1.0
    u = (a + b) / 2.0
    if u < 1.0:
        u = 1.0
    real = math.asin(t)
    imag = math.copysign(1.0, z.imag) * math.acosh(u)
    return complex(real, imag)


def acos(z: complex) -> complex:
    if z.real == 0.0 and z.imag == 0.0:
        return complex(pi / 2.0, 0.0)
    if z.real == 0.0:
        return complex(pi / 2.0, -math.asinh(z.imag))
    if z.imag == 0.0 and z.real >= -1.0 and z.real <= 1.0:
        return complex(math.acos(z.real), 0.0)
    if z.imag == 0.0 and z.real > 1.0:
        return complex(0.0, -math.acosh(z.real))
    if z.imag == 0.0 and z.real < -1.0:
        return complex(pi, -math.acosh(0.0 - z.real))

    a = math.sqrt((z.real + 1.0) * (z.real + 1.0) + z.imag * z.imag)
    b = math.sqrt((z.real - 1.0) * (z.real - 1.0) + z.imag * z.imag)
    t = (a - b) / 2.0
    if t > 1.0:
        t = 1.0
    if t < -1.0:
        t = -1.0
    u = (a + b) / 2.0
    if u < 1.0:
        u = 1.0
    real = (pi / 2.0) - math.asin(t)
    imag = -math.copysign(1.0, z.imag) * math.acosh(u)
    return complex(real, imag)


def atan(z: complex) -> complex:
    if z.real == 0.0 and z.imag == 0.0:
        return complex(0.0, 0.0)
    if z.real == 0.0:
        return complex(0.0, math.atanh(z.imag))
    if z.imag == 0.0:
        return complex(math.atan(z.real), 0.0)
    xx = z.real * z.real
    yy = z.imag * z.imag
    numerator = 2.0 * z.real
    denominator_real = 1.0 - xx - yy
    if denominator_real == 0.0:
        if numerator > 0.0:
            real = pi / 4.0
        elif numerator < 0.0:
            real = -pi / 4.0
        else:
            real = 0.0
    else:
        real = 0.5 * math.atan(numerator / denominator_real)
        if denominator_real < 0.0 and numerator >= 0.0:
            real = real + (pi / 2.0)
        elif denominator_real < 0.0 and numerator < 0.0:
            real = real - (pi / 2.0)
    denominator = xx + (z.imag - 1.0) * (z.imag - 1.0)
    if denominator <= 0.0:
        denominator = _MIN_POSITIVE
    ratio = (xx + (z.imag + 1.0) * (z.imag + 1.0)) / denominator
    if ratio <= 0.0:
        ratio = _MIN_POSITIVE
    imag = 0.25 * math.log(ratio)
    return complex(real, imag)


def asinh(z: complex) -> complex:
    if z.real == 0.0 and z.imag == 0.0:
        return complex(0.0, 0.0)
    if z.real == 0.0:
        return complex(0.0, _real_asin_unit(z.imag))
    if z.imag == 0.0:
        return complex(math.asinh(z.real), 0.0)

    a = math.sqrt(z.real * z.real + (z.imag + 1.0) * (z.imag + 1.0))
    b = math.sqrt(z.real * z.real + (z.imag - 1.0) * (z.imag - 1.0))
    t = (a - b) / 2.0
    if t > 1.0:
        t = 1.0
    if t < -1.0:
        t = -1.0
    u = (a + b) / 2.0
    if u < 1.0:
        u = 1.0
    real = math.copysign(1.0, z.real) * math.acosh(u)
    imag = math.copysign(1.0, z.imag) * math.asin(t)
    return complex(real, imag)


def acosh(z: complex) -> complex:
    if z.real == 1.0 and z.imag == 0.0:
        return complex(0.0, 0.0)
    if z.imag == 0.0 and z.real >= 1.0:
        return complex(math.acosh(z.real), 0.0)
    if z.imag == 0.0 and z.real >= -1.0 and z.real < 1.0:
        return complex(0.0, math.acos(z.real))
    if z.imag == 0.0 and z.real < -1.0:
        return complex(math.acosh(0.0 - z.real), pi)

    a = math.sqrt((z.real + 1.0) * (z.real + 1.0) + z.imag * z.imag)
    b = math.sqrt((z.real - 1.0) * (z.real - 1.0) + z.imag * z.imag)
    t = (a - b) / 2.0
    if t > 1.0:
        t = 1.0
    if t < -1.0:
        t = -1.0
    u = (a + b) / 2.0
    if u < 1.0:
        u = 1.0
    real = math.acosh(u)
    imag = _sign(z.imag) * ((pi / 2.0) - math.asin(t))
    return complex(real, imag)


def atanh(z: complex) -> complex:
    if z.real == 0.0 and z.imag == 0.0:
        return complex(0.0, 0.0)
    if z.imag == 0.0 and (z.real == 1.0 or z.real == -1.0):
        raise ValueError("math domain error")
    if z.real == 0.0:
        return complex(0.0, math.atan(z.imag))
    if z.imag == 0.0 and z.real > -1.0 and z.real < 1.0:
        return complex(math.atanh(z.real), 0.0)
    yy = z.imag * z.imag
    denominator = (1.0 - z.real) * (1.0 - z.real) + yy
    if denominator <= 0.0:
        denominator = _MIN_POSITIVE
    ratio = ((1.0 + z.real) * (1.0 + z.real) + yy) / denominator
    if ratio <= 0.0:
        ratio = _MIN_POSITIVE
    real = 0.25 * math.log(ratio)
    imag = 0.5 * math.atan2(2.0 * z.imag, 1.0 - z.real * z.real - yy)
    return complex(real, imag)


def isnan(z: complex) -> bool:
    return math.isnan(z.real) or math.isnan(z.imag)


def isinf(z: complex) -> bool:
    return math.isinf(z.real) or math.isinf(z.imag)


def isfinite(z: complex) -> bool:
    return math.isfinite(z.real) and math.isfinite(z.imag)


def isclose(a: complex, b: complex, rel_tol: float = 1e-09, abs_tol: float = 0.0) -> bool:
    if rel_tol < 0.0 or abs_tol < 0.0:
        raise ValueError("tolerances must be non-negative")

    if a == b:
        return True

    if isinf(a) or isinf(b):
        return False

    diff = a - b
    diff_abs = _abs_complex(diff)
    a_abs = _abs_complex(a)
    b_abs = _abs_complex(b)
    return diff_abs <= max(rel_tol * max(a_abs, b_abs), abs_tol)
