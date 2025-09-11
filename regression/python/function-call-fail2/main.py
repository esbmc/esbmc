# Function returning zero — should trigger divide-by-zero
def zero() -> int:
    return 0


# The division by zero yields infinity (inf)
z: float = 1.0 / zero()  # should fail
