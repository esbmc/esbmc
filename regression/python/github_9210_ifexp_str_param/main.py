# A str conditional expression passed to a str parameter is materialized (#9210).
d = {"a": 0, "b": 1}


def pick(k: str) -> int:
    return d[k]


c = nondet_bool()
r = pick("a" if c else "b")
assert r == (0 if c else 1)
