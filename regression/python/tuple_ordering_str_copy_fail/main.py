x = "aa"
if nondet_bool():
    x = "cc"

assert (x, 1) < ("bb", 0)
