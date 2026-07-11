# Verification harness for @dataclass default field values
# (preprocessor src/python-frontend/preprocessor/dataclass_mixin.py).
#
# A field with a default is filled from that default when the argument is
# omitted, and overridden when supplied.
#
# REQUIRES:
#   R1: a nondet integer for the required field.
#
# ENSURES (for Config with a: int, b: int = 10):
#   E1: the required field takes the positional argument
#   E2: an omitted field takes its default (10)
#   E3: a supplied field overrides the default
from dataclasses import dataclass


@dataclass
class Config:
    a: int
    b: int = 10


x: int = nondet_int()

c = Config(x)
assert c.a == x         # E1
assert c.b == 10        # E2

c2 = Config(x, 20)
assert c2.b == 20       # E3
