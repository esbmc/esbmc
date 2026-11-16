l = [0, 0, 0]
b: str = nondet_str()
if len(b) <= 2:
    assert l[len(b)] == 0
