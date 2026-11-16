# Issue #9312: m.E is not pkg.m.E, though both share a class symbol.
import m
from pkg.m import E


def run() -> int:
    try:
        raise E("pkg")
    except m.E:
        return 1
    return 0


assert run() == 1
