# GitHub #9079: the TypeError is raised only on the path where s is None.
from typing import Optional


def match(stderr: Optional[str]) -> bool:
    return 'ab' in stderr


c = nondet_bool()
s: Optional[str] = None
if c:
    s = "xab"
try:
    match(s)
except TypeError:
    assert c
