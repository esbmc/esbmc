# tuple(iterable) over lists. The tuple is modelled as the underlying
# sequence object, so equality routes through the list machinery (#4807,
# humaneval_33 pattern). Copy semantics live in tuple-from-list-copy,
# len/subscript/iteration in tuple-from-list-access and a subscript of a
# list-returning call in tuple-from-list-call-index: together they exceed the
# 120 s CI cap.

def ident(l: list):
    l = list(l)
    return l

# tuple of a list literal
assert tuple([1, 2, 3]) == tuple([1, 2, 3])

# tuple of a list variable
a = [1, 2, 3]
assert tuple(a) == tuple(a)

# tuple of a list-returning call (the humaneval_33 assertion shape)
assert tuple(ident([1, 2, 3])) == tuple(ident([1, 2, 3]))

print("ok")
