def pair() -> tuple[str, str]:
    return ("x", nondet_str())


a, b = pair()
assert a == "x"
