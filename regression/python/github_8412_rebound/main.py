# x is rebound after y = x, so y keeps the first object and x holds another.
x = "a"
y = x
x = "b"

assert y is not x
