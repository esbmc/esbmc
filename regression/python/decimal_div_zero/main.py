from decimal import Decimal


def div(a: Decimal, b: Decimal) -> Decimal:
    return a / b


def floordiv(a: Decimal, b: Decimal) -> Decimal:
    return a // b


div(Decimal(1), Decimal(2))
floordiv(Decimal(1), Decimal(2))

caught: int = 0
try:
    div(Decimal(1), Decimal(0))
except ZeroDivisionError:
    caught = caught + 1
try:
    floordiv(Decimal(-3), Decimal(0))
except ZeroDivisionError:
    caught = caught + 1
assert caught == 2
