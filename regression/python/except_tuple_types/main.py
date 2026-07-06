# `except (A, B):` catches any of the listed exception types. This previously
# crashed ESBMC (a json operator[] assertion, because the tuple type node has
# no "id"). Covers both listed types, a third type, the `as e` binding, body
# execution, and dispatch to a following handler when the tuple does not match.
caught = 0
try:
    raise KeyError()
except (ValueError, KeyError):
    caught = 1
assert caught == 1

try:
    raise ValueError()
except (ValueError, KeyError, IndexError):
    pass

try:
    raise IndexError()
except (ValueError, KeyError):
    assert False   # not one of the listed types here...
except IndexError:
    caught = 2
assert caught == 2

try:
    raise ValueError("boom")
except (ValueError, KeyError) as e:
    assert str(e) == "boom"
