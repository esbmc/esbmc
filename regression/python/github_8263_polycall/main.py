# ident's return type is inferred once for the whole program, from one of
# its call sites, so `ident(n) == n` was folded to False against the str
# type and a correct program was reported as failing. It is refused now.
def ident(x):
    return x


def main():
    n = nondet_int()
    assert ident(n) == n
    assert ident("s") == "s"


main()
