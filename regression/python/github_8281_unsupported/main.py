from _sv_verifier import check_type


def f(hint):
    check_type(1, hint)


f(int)
