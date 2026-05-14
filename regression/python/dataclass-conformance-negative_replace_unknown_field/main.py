from dataclasses import dataclass, replace


@dataclass
class Pair:
    left: int
    right: int


obj = Pair(1, 2)
exception_raised = False
try:
    replace(obj, missing=3)
except TypeError as exc:
    exception_raised = True

assert exception_raised
