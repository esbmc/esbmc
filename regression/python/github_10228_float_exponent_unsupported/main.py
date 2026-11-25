# Issue #10228: a float exponent keeps the refusal
def g(k: int) -> float:
    return k


x = 1.5
g(2 ** x)
