# len() of a dict literal used to fall through to strlen over the dict struct
# and report a wrong size. A dict bound to a name was never affected: it
# reaches the dict-aware path by type. List and tuple literals were already
# routed correctly.

assert len({"a": 1, "b": 2}) == 2
assert len({"a": 1}) == 1
assert len({}) == 0

d = {"a": 1, "b": 2}
assert len(d) == 2

assert len(dict(a=1, b=2)) == 2
assert len([1, 2, 3]) == 3
assert len((1, 2)) == 2
assert len("abc") == 3
