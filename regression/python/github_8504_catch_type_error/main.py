# A TypeError on a name that holds an int on one path and a str on the other
# is caught by except TypeError on the str path (#8504).
def main() -> None:
    if nondet_bool():
        a = 1
    else:
        a = "ab"
    if nondet_bool():
        b = 3
    else:
        b = "cd"

    try:
        d = a / 2
    except TypeError:
        d = 0.0
    try:
        s = a - 2
    except TypeError:
        s = 100
    try:
        lt = a < 2
    except TypeError:
        lt = True
    try:
        t = a - b
    except TypeError:
        t = 50
    try:
        n = -a
    except TypeError:
        n = 7

    assert d == 0.5 or d == 0.0
    assert s == -1 or s == 100
    assert lt
    assert t == -2 or t == 50
    assert n == -1 or n == 7


main()
