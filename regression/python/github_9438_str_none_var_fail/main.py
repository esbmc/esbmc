# github issue 9438: str() of a name bound to None is "None", not any string
x = None
assert str(x) == "None"
assert str(x) == "none"
