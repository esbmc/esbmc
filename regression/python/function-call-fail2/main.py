# Function returning zero — should trigger divide-by-zero
def zero() -> int:
    return 0

z: float = 1.0 / zero()  # should fail
