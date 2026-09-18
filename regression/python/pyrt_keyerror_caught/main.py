# A KeyError the dict model raises is catchable inside a try, and the
# hierarchy is CPython's: KeyError -> LookupError -> Exception.
d = {"a": 1}

c = 0
try:
    v = d["zz"]
    c = 99
except KeyError:
    c = 1
assert c == 1

c = 0
try:
    v = d["zz"]
except LookupError:
    c = 1
assert c == 1

c = 0
try:
    v = d["zz"]
except Exception:
    c = 1
assert c == 1

# A handler for a sibling class does not match; the next one does.
c = 0
try:
    v = d["zz"]
except TypeError:
    c = 99
except KeyError:
    c = 1
assert c == 1

# A lookup that succeeds raises nothing.
c = 0
try:
    c = d["a"]
except KeyError:
    c = 99
assert c == 1
