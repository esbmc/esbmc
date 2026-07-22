# Regression guard for #6243's fix, dict variant of
# github_6243_tuple_placeholder_no_crash: a dict that already had a member
# read against it (`y = d["a"]`) is a struct-shaped existing type, so the
# fix's allowlist correctly leaves it unretyped when later rebound to a
# constructor call — this only pins that doing so does not crash (it
# surfaces as a clean AttributeError verdict instead, a different shape
# than the tuple case's plain assertion mismatch, so it needs its own test).
class Service:
    def __init__(self, name):
        self._name = name


d = {"a": 1}
y = d["a"]
d = Service("hi")
assert d._name == "hi"
