from dataclasses import dataclass, replace


@dataclass
class C:
    x: int


c = C(1)

# unknown field name must raise; if replace does not raise this assertion fails
failed = False
try:
    replace(c, y=2)
except TypeError:
    failed = True

assert failed == True
