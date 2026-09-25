# Exercises the --python-irep2-adjust-only binary-arithmetic arm. Under
# --fixedbv the converter emits `n * x` with a signedbv and a fixedbv operand;
# without the usual arithmetic conversions the hop-off evaluates the product
# over unreconciled operands and reports a false alarm on an assertion legacy
# proves.


def mix(n: int, x: float) -> float:
    return n * x + n


assert mix(2, 1.5) == 5.0
