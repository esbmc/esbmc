# Falsification harness for decimal.Decimal equality
# (src/python-frontend/models/decimal.py).
#
# Decimal("1.5") and Decimal("2.5") are distinct values, so asserting their
# equality must be falsifiable.
#
# WRONG PROPERTY (expected to be falsified):
#   F1: Decimal("1.5") == Decimal("2.5").
from decimal import Decimal

a: Decimal = Decimal("1.5")
b: Decimal = Decimal("2.5")

assert a == b       # F1 — falsifiable (distinct values)
