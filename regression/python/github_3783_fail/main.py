# dict.popitem() on empty dict raises KeyError
d: dict[str, int] = {}
key, value = d.popitem()
