# Adapted from TypeEvalPy (https://github.com/secure-software-engineering/TypeEvalPy
# @ 3719de1), micro-benchmark/python_features/generators/yield_next/main.py
#
# The accumulator is conditionally initialised with a str before the loop.

# Call of a function which was yielded.

def squares():
    n = 1
    while True:
        yield n**2
        n += 1


def main() -> None:
    gen = squares()
    if nondet_bool():
        a = "total: "
    for i in range(5):
        try:
            a += next(gen)
        except NameError:
            a = next(gen)


main()
