# Issue #10228: the same conversion for a keyword argument
def g(k: int) -> int:
    return k


n = nondet_int()
if 0 <= n <= 62:
    assert g(k=2 ** n) >= 1
    assert g(2 ** n) % 2 == 0 or n == 0
