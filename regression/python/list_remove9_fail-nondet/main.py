x: int = nondet_int()

l = [1, 2, 3]

if x != 1 and x != 2 and x != 3:
    # This must fail
    l.remove(x)
    assert False  # unreachable, ESBMC should report violation
