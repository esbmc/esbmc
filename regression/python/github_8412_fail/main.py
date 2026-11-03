# c names the object v names, so `is not` is False and the assertion fails.
v = "hello"
b = v
c = b

assert c is not v
