l = [0, 1, 2]
b: str = nondet_str()
if len(b) <= 2:
    assert l[len(b)] == 0
