# Negative variant of #5161: pin the checker boundary. larger(1, 2) computes 2,
# so asserting it equals 1 must report VERIFICATION FAILED. If float() ever
# regresses to folding to 0.0 (making both branches collapse), this guards it.


def larger(a, b):
    t = a
    if isinstance(t, str):
        t = t.replace(',', '.')
    return a if float(t) > float(b) else b


if __name__ == "__main__":
    assert larger(1, 2) == 1
