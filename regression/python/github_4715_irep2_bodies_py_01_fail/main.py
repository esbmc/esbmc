# Negative counterpart to github_4715_irep2_bodies_py_01.
# The assertion is intentionally wrong so ESBMC produces VERIFICATION FAILED,
# proving the Python frontend's --irep2-bodies path still catches bugs.


def test():
    x = 0
    if True:
        x = 1

    assert x == 0, "wrong: x is 1 after the if branch"


test()
