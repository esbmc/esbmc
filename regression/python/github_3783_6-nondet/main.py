# popitem() on a dict whose value is a nondet_bool preserves the value
# and the dict is empty afterwards
flag: bool = nondet_bool()
d: dict[str, bool] = {"active": flag}
key, value = d.popitem()
assert key == "active"
assert value == flag
