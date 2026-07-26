# Exercises the --python-irep2-adjust-only function->pointer decay arm: a
# factory returns a bare code-typed designator, which must decay to `&triple`
# (C11 6.3.2.1p4) before symex, or the indirect call has no target and SMT
# encoding aborts on "Unexpected type in int/ptr typecast". No free variable is
# captured -- closure capture is a separate, pre-existing gap on both paths.
def get_tripler():
    def triple(x: int) -> int:
        return x * 3

    return triple


f = get_tripler()
assert f(4) == 12
