# list(d.items()) on a non-empty dict should not equal []
d: dict[str, int] = {"a": 1, "b": 2}
assert list(d.items()) != []
