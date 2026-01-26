def add_positive(a: int, b: int) -> int:
    """Add two positive numbers"""
    if a < 0 or b < 0:
        a = a + b
    return a + b

add_positive(__VERIFIER_nondet_int(), __VERIFIER_nondet_int())