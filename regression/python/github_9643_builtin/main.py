import _sv_verifier


def f(list):
    return list


x = [1]
_sv_verifier.check_type(x, list)
