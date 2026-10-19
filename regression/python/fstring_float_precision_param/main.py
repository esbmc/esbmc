def f(v: float) -> str:
    return f"v {v:.6f}"


s = f(2.5)
assert len(s) >= 0
