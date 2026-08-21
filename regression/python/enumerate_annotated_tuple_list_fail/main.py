# The unpacked components are really read, not assumed.
from typing import List, Tuple

pairs: List[Tuple[int, int]] = []
pairs.append((1, 2))

for i, tpl in enumerate(pairs):
    a, b = tpl
    assert a > b
