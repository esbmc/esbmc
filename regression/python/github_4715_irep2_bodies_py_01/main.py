# Exercises structured-CF code kinds (if/while/for/break/continue) from the
# Python frontend under --irep2-bodies (V.4.3, esbmc/esbmc#4715).
# Verdict must be VERIFICATION SUCCESSFUL with and without the flag.


def test():
    x = 0
    y = 5

    # code_ifthenelse2t
    if y > 0:
        x = 1
    else:
        x = 2

    # code_while2t + code_break2t
    w = 0
    while w < 3:
        if w == 2:
            break
        w += 1

    # code_for2t (via range) + code_continue2t
    s = 0
    for i in range(4):
        if i == 2:
            continue
        s += i

    assert x == 1, "x should be 1 after if-then-else"
    assert w == 2, "while loop with break should stop at w==2"
    assert s == 4, "for loop with continue: 0+1+3==4"


test()
