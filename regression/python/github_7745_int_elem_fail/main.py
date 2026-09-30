# github #7745: the failing twin of github_7745_int_elem.
f = lambda c: c + 1
xs = [1]
x = f(xs[0])
assert x == 99
