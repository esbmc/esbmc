# End-to-end check that ``from dataclasses import dataclass as <alias>`` and
# ``from dataclasses import field as <alias>`` are recognized by the
# preprocessor's dataclass expansion (Marco D + alias polish).
from dataclasses import dataclass as dc, field as fld


@dc
class C:
    a: int = 5
    b: int = fld(default=7)


c = C()
assert c.a == 5
assert c.b == 7

c2 = C(a=10, b=20)
assert c2.a == 10
assert c2.b == 20
