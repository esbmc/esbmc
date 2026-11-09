s: str = nondet_str()
if len(s) <= 2:
    n = 0
    for c in s:
        n += 1
    assert n == len(s)
