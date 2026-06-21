# pylint: disable=redefined-builtin,undefined-variable,unused-argument
# Some functions in this module intentionally shadow Python built-ins
# (e.g. `round`) and reference typing forward-declarations (e.g. `Any`)
# that have no Python binding: they are the operational models ESBMC
# uses to verify Python programs, so they must match the built-in names
# exactly. Argument names are part of the API contract matched by ESBMC's
# Python converter, even when the abstract body does not reference them.
import math


pi: float = math.pi
e: float = math.e


# Stubs for type inference.
def array(data: list[Any]) -> list[Any]:
    return data


def zeros(shape: int) -> list[float]:
    result: list[float] = [0.0]
    return result


def ones(shape: int) -> list[float]:
    result: list[float] = [1.0]
    return result


def add(a: int, b: int) -> float:
    x: float = a + b
    return x


def subtract(a: int, b: int) -> float:
    x: float = a - b
    return x


def multiply(a: int, b: int) -> float:
    x: float = a * b
    return x


def divide(a: int, b: int) -> float:
    x: float = a / b
    return x


def power(a: int, b: int) -> float:
    x: float = 42.0
    return x


def ceil(x: float) -> int:
    return 0


def floor(x: float) -> int:
    return 0


def fabs(x: float) -> float:
    return 1.0


def sqrt(x: float) -> float:
    return math.sqrt(x)

def trunc(x: float) -> float:
    return 0.2


def round(x: float, decimals: int = 0) -> float:
    scale: float = 1.0
    if decimals > 0:
        i: int = 0
        while i < decimals:
            scale = scale * 10.0
            i = i + 1
    elif decimals < 0:
        i = 0
        while i < 0 - decimals:
            scale = scale / 10.0
            i = i + 1

    scaled: float = x * scale
    integer: int = math.floor(scaled)
    fraction: float = scaled - float(integer)
    if fraction < 0.5:
        return float(integer) / scale
    if fraction > 0.5:
        return float(integer + 1) / scale
    if integer % 2 == 0:
        return float(integer) / scale
    return float(integer + 1) / scale


def sin(x: float) -> float:
    return math.sin(x)


def cos(x: float) -> float:
    return math.cos(x)


def arccos(x: float) -> float:
    return 0.2


def exp(x: float) -> float:
    return math.exp(x)


def fmod(x: float) -> float:
    return 0.2


def arctan(x: float) -> float:
    return math.atan(x)


def arcsin(x: float) -> float:
    return math.asin(x)


def tan(x: float) -> float:
    return math.tan(x)


def log(x: float) -> float:
    return math.log(x)


def log2(x: float) -> float:
    return math.log2(x)


def log10(x: float) -> float:
    return math.log10(x)


def sinh(x: float) -> float:
    return math.sinh(x)


def cosh(x: float) -> float:
    return math.cosh(x)


def tanh(x: float) -> float:
    return math.tanh(x)


def rint(x: float) -> float:
    return float(round(x))


def remainder(x: float, y: float) -> float:
    return x - (math.floor(x / y) * y)


def nextafter(x: float, y: float) -> float:
    return math.nextafter(x, y)


def modf(x: float) -> tuple[float, float]:
    return math.modf(x)


def frexp(x: float) -> tuple[float, int]:
    return math.frexp(x)


def isclose(
    a: float,
    b: float,
    rel_tol: float = 1e-09,
    abs_tol: float = 0.0,
) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def copysign(x: float, y: float) -> float:
    return math.copysign(x, y)


def fmin(x: float, y: float) -> float:
    if x != x:
        return y
    if y != y:
        return x
    if x < y:
        return x
    return y


def fmax(x: float, y: float) -> float:
    if x != x:
        return y
    if y != y:
        return x
    if x > y:
        return x
    return y


def greater(a: Any, b: Any) -> Any:
    if type(a) == list:
        out: list[Any] = []
        i = 0
        while i < len(a):
            rhs = b[i] if type(b) == list else b
            out = out + [a[i] > rhs]
            i = i + 1
        return out
    if type(b) == list:
        out = []
        i = 0
        while i < len(b):
            out = out + [a > b[i]]
            i = i + 1
        return out
    return a > b


def less(a: Any, b: Any) -> Any:
    if type(a) == list:
        out: list[Any] = []
        i = 0
        while i < len(a):
            rhs = b[i] if type(b) == list else b
            out = out + [a[i] < rhs]
            i = i + 1
        return out
    if type(b) == list:
        out = []
        i = 0
        while i < len(b):
            out = out + [a < b[i]]
            i = i + 1
        return out
    return a < b


def greater_equal(a: Any, b: Any) -> Any:
    if type(a) == list:
        out: list[Any] = []
        i = 0
        while i < len(a):
            rhs = b[i] if type(b) == list else b
            out = out + [a[i] >= rhs]
            i = i + 1
        return out
    if type(b) == list:
        out = []
        i = 0
        while i < len(b):
            out = out + [a >= b[i]]
            i = i + 1
        return out
    return a >= b


def less_equal(a: Any, b: Any) -> Any:
    if type(a) == list:
        out: list[Any] = []
        i = 0
        while i < len(a):
            rhs = b[i] if type(b) == list else b
            out = out + [a[i] <= rhs]
            i = i + 1
        return out
    if type(b) == list:
        out = []
        i = 0
        while i < len(b):
            out = out + [a <= b[i]]
            i = i + 1
        return out
    return a <= b


def equal(a: Any, b: Any) -> Any:
    if type(a) == list:
        out: list[Any] = []
        i = 0
        while i < len(a):
            rhs = b[i] if type(b) == list else b
            out = out + [a[i] == rhs]
            i = i + 1
        return out
    if type(b) == list:
        out = []
        i = 0
        while i < len(b):
            out = out + [a == b[i]]
            i = i + 1
        return out
    return a == b


def not_equal(a: Any, b: Any) -> Any:
    if type(a) == list:
        out: list[Any] = []
        i = 0
        while i < len(a):
            rhs = b[i] if type(b) == list else b
            out = out + [a[i] != rhs]
            i = i + 1
        return out
    if type(b) == list:
        out = []
        i = 0
        while i < len(b):
            out = out + [a != b[i]]
            i = i + 1
        return out
    return a != b


def logical_and(a: Any, b: Any) -> Any:
    if type(a) == list:
        out: list[Any] = []
        i = 0
        while i < len(a):
            rhs = b[i] if type(b) == list else b
            out = out + [bool(a[i]) and bool(rhs)]
            i = i + 1
        return out
    if type(b) == list:
        out = []
        i = 0
        while i < len(b):
            out = out + [bool(a) and bool(b[i])]
            i = i + 1
        return out
    return bool(a) and bool(b)


def logical_or(a: Any, b: Any) -> Any:
    if type(a) == list:
        out: list[Any] = []
        i = 0
        while i < len(a):
            rhs = b[i] if type(b) == list else b
            out = out + [bool(a[i]) or bool(rhs)]
            i = i + 1
        return out
    if type(b) == list:
        out = []
        i = 0
        while i < len(b):
            out = out + [bool(a) or bool(b[i])]
            i = i + 1
        return out
    return bool(a) or bool(b)


def logical_not(a: Any) -> Any:
    if type(a) == list:
        out: list[Any] = []
        i = 0
        while i < len(a):
            out = out + [not bool(a[i])]
            i = i + 1
        return out
    return not bool(a)


def sum(a: Any, axis: int = -1) -> Any:
    if type(a) == list:
        total = 0
        i = 0
        while i < len(a):
            total = total + a[i]
            i = i + 1
        return total
    return a


def prod(a: Any, axis: int = -1) -> Any:
    if type(a) == list:
        result = 1
        i = 0
        while i < len(a):
            result = result * a[i]
            i = i + 1
        return result
    return a


def min(a: Any, axis: int = -1) -> Any:
    if type(a) == list:
        if len(a) == 0:
            raise ValueError("min() arg is an empty sequence")
        result = a[0]
        i = 1
        while i < len(a):
            if a[i] < result:
                result = a[i]
            i = i + 1
        return result
    return a


def max(a: Any, axis: int = -1) -> Any:
    if type(a) == list:
        if len(a) == 0:
            raise ValueError("max() arg is an empty sequence")
        result = a[0]
        i = 1
        while i < len(a):
            if a[i] > result:
                result = a[i]
            i = i + 1
        return result
    return a


def mean(a: Any, axis: int = -1) -> Any:
    if type(a) == list:
        total = 0
        i = 0
        while i < len(a):
            total = total + a[i]
            i = i + 1
        return total / float(len(a))
    return a


def argmin(a: Any, axis: int = -1) -> Any:
    if type(a) == list:
        if len(a) == 0:
            raise ValueError("argmin() arg is an empty sequence")
        best_idx = 0
        best = a[0]
        i = 1
        while i < len(a):
            if a[i] < best:
                best = a[i]
                best_idx = i
            i = i + 1
        return best_idx
    return 0


def argmax(a: Any, axis: int = -1) -> Any:
    if type(a) == list:
        if len(a) == 0:
            raise ValueError("argmax() arg is an empty sequence")
        best_idx = 0
        best = a[0]
        i = 1
        while i < len(a):
            if a[i] > best:
                best = a[i]
                best_idx = i
            i = i + 1
        return best_idx
    return 0


def where(cond: Any, x: Any, y: Any) -> Any:
    if type(cond) == list:
        out = []
        i = 0
        while i < len(cond):
            x_item = x[i] if type(x) == list else x
            y_item = y[i] if type(y) == list else y
            out = out + [x_item if cond[i] else y_item]
            i = i + 1
        return out
    if cond:
        return x
    return y


def arange(start: int, stop: int = 0, step: int = 1) -> list[int]:
    begin = start
    end = stop
    if stop == 0 and step == 1:
        begin = 0
        end = start

    if step == 0:
        raise ValueError("arange() step must not be zero")

    out: list[int] = []
    value: int = begin
    if step > 0:
        while value < end:
            out = out + [value]
            value = value + step
    else:
        while value > end:
            out = out + [value]
            value = value + step
    return out


def full(shape: Any, fill_value: Any) -> Any:
    if type(shape) == list:
        rows = shape[0]
        cols = shape[1]
        out: list[Any] = []
        i: int = 0
        while i < rows:
            row: list[Any] = []
            j: int = 0
            while j < cols:
                row = row + [fill_value]
                j = j + 1
            out = out + [row]
            i = i + 1
        return out
    if type(shape) == tuple:
        rows = shape[0]
        cols = shape[1]
        out: list[Any] = []
        i: int = 0
        while i < rows:
            row: list[Any] = []
            j: int = 0
            while j < cols:
                row = row + [fill_value]
                j = j + 1
            out = out + [row]
            i = i + 1
        return out

    out: list[Any] = []
    i = 0
    while i < shape:
        out = out + [fill_value]
        i = i + 1
    return out


def eye(n: int, m: int = 0) -> list[list[int]]:
    cols = n if m == 0 else m
    out: list[list[int]] = []
    i: int = 0
    while i < n:
        row: list[int] = []
        j: int = 0
        while j < cols:
            row = row + [1 if i == j else 0]
            j = j + 1
        out = out + [row]
        i = i + 1
    return out


def identity(n: int) -> list[list[int]]:
    out: list[list[int]] = []
    i: int = 0
    while i < n:
        row: list[int] = []
        j: int = 0
        while j < n:
            row = row + [1 if i == j else 0]
            j = j + 1
        out = out + [row]
        i = i + 1
    return out


def linspace(start: float, stop: float, num: int = 50) -> list[float]:
    if num <= 0:
        return []
    if num == 1:
        return [start]

    step = (stop - start) / float(num - 1)
    out: list[float] = []
    i: int = 0
    while i < num:
        out = out + [start + (step * float(i))]
        i = i + 1
    return out


def dot(a: float, b: float) -> float:
    return 0.0


def matmul(a: float, b: float) -> float:
    return 0.0


def transpose(a: float, b: float) -> float:
    return 0.0
