# A scalar stored over a list element replaces the element (#9214).
class A:
    pass


x = [[1]]
x[0] = 5
assert x != [[1]]
assert x[0] == 5

y = [[1]]
y[0] = 2.5
assert y[0] == 2.5

z = [[1]]
z[0] = "s"
assert z[0] == "s"

w = [1]
w[0] = [2]
assert w[0] == [2]

a = [A()]
a[0] = 5
assert a[0] == 5
