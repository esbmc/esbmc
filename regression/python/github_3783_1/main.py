# Basic dict.popitem() with string keys and int values
d: dict[str, int] = {"a": 1, "b": 2, "c": 3}
key, value = d.popitem()
# LIFO: last inserted is "c": 3
assert key == "c"
assert value == 3
assert len(d) == 2
