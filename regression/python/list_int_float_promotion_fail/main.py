# Negative variant of list_int_float_promotion: the float 3.5 IS > 3.0, so the
# expected list is wrong and the assertion must be violated. This guards against
# a regression that truncates 3.5 -> 3 (which would make the wrong expectation
# hold and the assertion pass).


def grade(gpas):
    out = []
    for g in gpas:
        if g > 3.0:
            out.append("hi")
        else:
            out.append("lo")
    return out


if __name__ == "__main__":
    # Wrong on purpose: 3.5 > 3.0 yields "hi", not "lo".
    assert grade([3, 3.5]) == ["lo", "lo"]
