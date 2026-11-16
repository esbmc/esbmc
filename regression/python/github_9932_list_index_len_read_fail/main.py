l = [5, 6, 7]
b: str = nondet_str()
if len(b) <= 2:
    x = l[len(b)]
    assert x != 7
