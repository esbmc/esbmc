# A `List[Tuple[int, int]]` annotation yields only the element's head name, so
# the loop value was annotated with a component-less `Tuple` and read back as
# an opaque pointer that `a, b = tpl` could not unpack -- strictly worse than
# the unannotated case, which types the value `Any` and works.
from typing import List, Tuple

pairs: List[Tuple[int, int]] = []
pairs.append((1, 2))
pairs.append((3, 4))

total = 0
for i, tpl in enumerate(pairs):
    a, b = tpl
    assert a < b
    total = total + a + b
assert total == 10
assert i == 1
