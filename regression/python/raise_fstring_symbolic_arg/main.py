def f(x: int) -> int:
    if x < 1:
        raise ValueError(f"f expects a positive value, x={x}")
    return x

assert f(5) == 5
