def over_ab(s):
    for c in s:
        if c != "a" and c != "b":
            return False
    return True


def main() -> None:
    t1 = "a" if nondet_bool() else "b"
    t2 = "a" if nondet_bool() else "b"
    t = t1 + t2
    assert over_ab(t)


main()
