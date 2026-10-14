# Falsification harness for enum.Enum equality
# (src/python-frontend/models/enum.py).
#
# Distinct enum members are never equal, so asserting equality between two of
# them must be falsifiable.
#
# WRONG PROPERTY (expected to be falsified):
#   F1: Color.RED == Color.GREEN.  They are distinct members.
from enum import Enum


class Color(Enum):
    RED = 1
    GREEN = 2


assert Color.RED == Color.GREEN  # F1 — falsifiable (distinct members)
