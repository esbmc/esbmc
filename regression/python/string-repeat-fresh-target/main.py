# The working counterpart of string-augmented-repeat: with a fresh target the
# source keeps its array type, so __python_str_repeat gets a static length.
s = "ab"
t = s * 3
assert t == "ababab"
assert len(t) == 6
assert s == "ab"
