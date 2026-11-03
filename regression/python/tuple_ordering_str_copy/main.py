x = "aa"
if nondet_bool():
    x = "cc"

assert (x, 1) != ("", 1)
assert (x, 1) < ("dd", 0)
assert (x, 1) == ("aa", 1) or (x, 1) == ("cc", 1)
