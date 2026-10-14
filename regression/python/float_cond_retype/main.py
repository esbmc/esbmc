# Regression for #5161: float() applied to a numeric variable that is only
# conditionally reassigned a string (e.g. inside `if isinstance(x, str)`) must
# emit a numeric cast, not fold to 0.0 from the variable's stale string value.


def larger(a, b):
    t = a
    if isinstance(t, str):
        t = t.replace(',', '.')
    return a if float(t) > float(b) else b


if __name__ == "__main__":
    assert larger(1, 2) == 2
    assert larger(5, 3) == 5
