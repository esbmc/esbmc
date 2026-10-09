# Both branches of the conditional expression bind the same kind, so the
# rewrite that #8263 added must not fire and `v + 1` stays well typed.
def main():
    i = nondet_int()
    if not (0 <= i and i <= 1):
        return
    v = 1 if i == 0 else 2
    y = v + 1
    assert y == 2 or y == 3


main()
