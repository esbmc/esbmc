# list.extend(other) appends every element of `other` in place. The operational
# model previously copied each element with an inline alloca+memcpy whose
# per-byte loop, nested inside the extend loop, left the resulting list size
# unconstrained in the SMT model -- so even len() after a non-empty extend was
# non-deterministic and correct assertions failed.
a = [1, 2]
a.extend([3, 4])
assert len(a) == 4
assert a == [1, 2, 3, 4]
assert a[2] == 3

# extending by a variable, and chaining
b = [5]
c = [6, 7]
b.extend(c)
b.extend([8])
assert b == [5, 6, 7, 8]

# empty extend is a no-op; float elements keep their value
d = [1.5]
d.extend([])
d.extend([2.5, 3.5])
assert len(d) == 3
assert d[1] == 2.5 and d[2] == 3.5

# string elements
e = ["x"]
e.extend(["y", "z"])
assert e[2] == "z"
