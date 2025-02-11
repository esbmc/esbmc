def saturating_sub(a: int, b: int) -> int:
    """
    Computes a - b, saturating at numeric bounds.
    """
    return a - b if a > b else 0

x: int = saturating_sub(3, 1)
assert(x == 2)
