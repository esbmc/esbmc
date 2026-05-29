# Regression for issue #4819: float() of a string variable that is not a valid
# float literal must still raise ValueError. This exercises the runtime
# validity guard (__python_str_is_float) on the invalid path.


def to_float(value):
    return float(value)


if __name__ == "__main__":
    x = to_float("abc")
    print(x)
