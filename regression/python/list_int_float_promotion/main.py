# A list literal that mixes int and float must promote to float (Python widens
# int to float). The int element 3 appears before the float 3.5, which used to
# force the list to list[int] and truncate 3.5 -> 3 (issue #5156).


def grade(gpas):
    out = []
    for g in gpas:
        if g > 3.0:
            out.append("hi")
        else:
            out.append("lo")
    return out


if __name__ == "__main__":
    # 3 (int) is not > 3.0 -> "lo"; 3.5 (float) must stay 3.5 -> "hi".
    assert grade([3, 3.5]) == ["lo", "hi"]
    # bool is int-like in Python; a bool/float mix also promotes to float.
    assert sum([True, 1.5, 2.5]) == 5.0
