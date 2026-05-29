# Regression for issue #4819: float() applied to a *string variable* must parse
# the value at runtime (via the __python_str_to_float operational model) instead
# of unconditionally raising ValueError. Before the fix, any float(<str var>)
# raised ValueError even when the string was a valid number.


def to_float(value):
    return float(value)


def closest_integer(value):
    # Mirrors the humaneval_99 composition that originally exposed the bug.
    return int(round(float(value)))


if __name__ == "__main__":
    assert to_float("10") == 10.0
    assert to_float("-5") == -5.0
    assert to_float("2.5") == 2.5
    # Parsing a variable must agree bit-for-bit with the literal/strtod path,
    # including fractional values that are not exactly representable.
    assert to_float("0.3") == 0.3
    assert to_float("12.34") == 12.34
    assert closest_integer("10") == 10
    print("ok")
