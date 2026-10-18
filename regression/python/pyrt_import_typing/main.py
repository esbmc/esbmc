# Annotations are discarded, so typing has no contents to bring in: the names
# it binds only ever appear in annotations, which produce no claim either way.
import typing
from typing import Any, List, Optional
from typing import Dict as D


x: List = [1, 2]
y: Optional = None
z: D = {"a": 1}


def f(a: Any) -> Any:
    return a


assert len(x) == 2
assert y is None
assert len(z) == 1
assert f(4) == 4
