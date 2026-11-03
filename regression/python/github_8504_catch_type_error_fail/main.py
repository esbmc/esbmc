# On the str path a - 2 raises TypeError, so the handler's value reaches the
# assertion and it fails there (#8504).
def main() -> None:
    if nondet_bool():
        a = 1
    else:
        a = "ab"
    try:
        s = a - 2
    except TypeError:
        s = 100
    assert s == -1


main()
