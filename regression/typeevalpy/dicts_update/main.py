# Adapted from TypeEvalPy (https://github.com/secure-software-engineering/TypeEvalPy
# @ 3719de1), micro-benchmark/python_features/dicts/update/main.py
#
# The update() call is made conditional, and the value returned by the
# stored function is used afterwards.

# The update method of dictionaries is used.

def func1():
    return 42


def func2():
    return "Hello from func2"


def main() -> None:
    d = {"a": func1}
    if nondet_bool():
        d.update({"a": func2})
    e = d["a"]()
    f = e + 1


main()
