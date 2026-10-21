import helper
from helper import mk


def via_attribute(c: bool) -> None:
    if c:
        x = 1
    else:
        x = helper.mk()
    if not c:
        assert isinstance(x, str)


def via_from_import(c: bool) -> None:
    if c:
        x = 1
    else:
        x = mk()
    if not c:
        assert x == "a"


def concatenated(c: bool) -> None:
    if c:
        x = 1
    else:
        x = helper.mk()
    if not c:
        assert x + "b" == "ab"


def qualified_nondet(c: bool) -> None:
    if c:
        x = 1
    else:
        x = helper.nondet_str()
    if not c:
        assert isinstance(x, str)


c = helper.nondet_bool()
via_attribute(c)
via_from_import(c)
concatenated(c)
qualified_nondet(c)
