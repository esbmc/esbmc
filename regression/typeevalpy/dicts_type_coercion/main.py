# Adapted from TypeEvalPy (https://github.com/secure-software-engineering/TypeEvalPy
# @ 3719de1), micro-benchmark/python_features/dicts/type_coercion/main.py
#
# Each branch of a nondeterministic choice looks up one of the two keys
# and uses the value returned by the stored function.

# Check if tool type coerces integer and string values.

def func1():
    return "Hello from func1"


def func2():
    return 42


def main() -> None:
    d = {1: func1, "1": func2}
    if nondet_bool():
        x = d[1]().upper()
    else:
        x = d["1"]() + 1


main()
