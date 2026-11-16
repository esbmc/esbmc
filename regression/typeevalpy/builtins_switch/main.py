# Adapted from TypeEvalPy (https://github.com/secure-software-engineering/TypeEvalPy
# @ 3719de1), micro-benchmark/python_features/builtins/switch/main.py
#
# The matched value is chosen nondeterministically among the three cases,
# and the returned value is used afterwards.

#  A function is defined with switch statement in it.

def func(value):
    match value:
        case "case1":
            return 42
        case "case2":
            return "hello this is case2"
        case _:
            return "unknown type"


def main() -> None:
    if nondet_bool():
        v = "case1"
    elif nondet_bool():
        v = "case2"
    else:
        v = "case3"
    r = func(v)
    n = r + 1


main()
