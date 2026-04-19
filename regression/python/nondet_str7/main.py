# Test 1: Basic nondet_str() with reassignment on empty
a = nondet_str()
if a == "":
    a = "default"

# Test 2: Multiple independent nondet_str() allocations
b = nondet_str()
c = nondet_str()

# Test 3: String reassignment
d = nondet_str()
d = "fixed"
assert d == "fixed"

# Test 4: Comparison with non-empty constant
e = nondet_str()
if e == "test":
    e = "matched"
else:
    e = "other"

# Test 5: List storage and retrieval (regression test)
str1 = nondet_str()
str2 = nondet_str()
str3 = nondet_str()
l = [str1, str2, str3]
assert l[0] == str1
assert l[1] == str2
assert l[2] == str3

# Test 6: Multiple reassignments
f = nondet_str()
f = "first"
f = "second"
assert f == "second"

# Test 7: Empty string edge case
g = nondet_str()
if g == "":
    g = "was_empty"
    assert g == "was_empty"

# Test 8: Inequality check
h = nondet_str()
i = nondet_str()
result = (h == i)
