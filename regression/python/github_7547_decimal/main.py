from decimal import Decimal


def f(s: str) -> None:
    d = Decimal(s)
    assert d == d
