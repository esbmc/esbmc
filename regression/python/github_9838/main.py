# Issue #9838: a missing attribute as a later and/or operand raises
# AttributeError only where the operand is evaluated.
class C:
    pass


def assigned(c, b: bool) -> int:
    try:
        x = b or c.limit
    except AttributeError:
        return 1
    return 2


def tested(c, b: bool) -> int:
    try:
        if b and c.limit:
            return 3
    except AttributeError:
        return 1
    return 2


def assigned_first(c) -> int:
    try:
        x = c.limit or True
    except AttributeError:
        return 1
    return 2


def tested_first(c) -> int:
    try:
        if c.limit or True:
            return 3
    except AttributeError:
        return 1
    return 2


def assigned_not(c) -> int:
    try:
        x = False or not c.limit
    except AttributeError:
        return 1
    return 2


def comprehended(c) -> int:
    try:
        d = {i: (c.limit or i) for i in range(2)}
    except AttributeError:
        return 1
    return 2


c = C()
assert assigned(c, False) == 1
assert assigned(c, True) == 2
assert tested(c, True) == 1
assert tested(c, False) == 2
assert assigned_first(c) == 1
assert tested_first(c) == 1
assert assigned_not(c) == 1
assert comprehended(c) == 1
