# HumanEval/144 - non-vacuity counterpart of humaneval_144.
# simplify("1/5", "5/1") computes 1*5 / (5*1) == 1.0, a whole number, so the
# real result is True. Asserting the opposite must FAIL with a counterexample,
# proving the passing test verifies the genuine computed value rather than
# passing vacuously (the str.split()/int() models are actually exercised).


def simplify(x, n):
    a, b = x.split("/")
    c, d = n.split("/")
    numerator = int(a) * int(c)
    denom = int(b) * int(d)
    if (numerator/denom == int(numerator/denom)):
        return True
    return False


if __name__ == "__main__":
    assert simplify("1/5", "5/1") == False, 'non-vacuity: real result is True'
    print("unreachable")
