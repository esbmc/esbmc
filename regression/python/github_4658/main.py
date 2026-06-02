def make(s: str):
    return complex(s)


c = make("1+2j")
assert c.real == 1.0
