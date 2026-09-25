# github #7745: a non-class element keeps the registry's verdict. The AST
# fallback added for #7745 must answer only for class instances -- this is the
# shape that would break if it ever started answering for anything else.
f = lambda c: c + 1
xs = [1]
x = f(xs[0])
assert x == 2
