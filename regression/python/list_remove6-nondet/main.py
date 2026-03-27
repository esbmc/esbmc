# Verify that after remove, the element count decreases by exactly 1
def nondet_int() -> int:
    ...


x: int = nondet_int()
l = [1, 2, 3]

# Only remove if present to avoid ValueError
if x == 1 or x == 2 or x == 3:
    l.remove(x)
    assert len(l) == 2
