def over_a(s):
    for c in s:
        if c != "a":
            return False
    return True


def main() -> None:
    t1 = "a" if nondet_bool() else "b"
    t2 = "a" if nondet_bool() else "b"
    t = t1 + t2
    assert over_a(t)


main()
