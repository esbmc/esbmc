def get_float() -> float:
    return 3.14

def double_float(x: float) -> float:
    return x * 2.0

assert abs(double_float(get_float()) - 6.28) < 0.01
