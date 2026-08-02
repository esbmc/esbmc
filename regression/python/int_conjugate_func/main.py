# Regression for the --function path: a function calling x.conjugate() must
# be recognised as an int model method, not resolved as a user function
# (which previously crashed with an uncaught json type_error).
def check(n: int) -> None:
    assert n.conjugate() == n


check(5)
