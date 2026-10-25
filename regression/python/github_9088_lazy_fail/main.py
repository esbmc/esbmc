# GitHub #9088: in an and/or tail the TypeError is raised only when the tail
# runs.
def f(c: bool) -> bool:
    try:
        return c and len([1] + (2, )) > 0
    except TypeError:
        assert not c
        return False


f(nondet_bool())
