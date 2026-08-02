# tuple(iterable) over lists. The tuple is modelled as the underlying
# sequence object, so equality routes through the list machinery (#4807,
# humaneval_33 pattern). Copy semantics live in tuple-from-list-copy and
# len/subscript/iteration in tuple-from-list-access; verifying all three in
# one program lands at ~85s on the macOS arm64 runner against a 120s cap.

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
assert tuple(ident([4, 5, 6]))[0] == 4

print("ok")
