# The argument's int type reaches the lambda parameter (#7745).
g = lambda n: n + 0
x: int = 9007199254740993
assert g(x) == 9007199254740992
