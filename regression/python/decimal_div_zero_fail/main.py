from decimal import Decimal


def f(a: Decimal, b: Decimal) -> Decimal:
    return a / b


f(Decimal(1), Decimal(0))
