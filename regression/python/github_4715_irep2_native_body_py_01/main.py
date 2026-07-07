# Pins the experimental --irep2-native-body flag (W1-loc spike Phase C,
# esbmc/esbmc#4715). Until the IREP2-native goto_convert dispatcher supports a
# body's statement kinds it falls back to the legacy round-trip, so the verdict
# must stay VERIFICATION SUCCESSFUL, byte-identical to a run without the flag.


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
