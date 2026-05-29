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
    assert closest_integer("10") == 10
    print("ok")
