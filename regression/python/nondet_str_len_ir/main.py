s: str = nondet_str()
if len(s) <= 2:
    assert len(s[:]) == len(s)
